"""
bars.py
-------
Bar fetching and SESSION-ALIGNED resampling.

Why this module exists
======================
The v3 scanner built its 4H bars by bucketing Alpaca's 1-Hour bars from a
09:30 ET anchor. Two things broke that:

  1. Alpaca stamps hourly bars on the hour (09:00, 10:00, 11:00...). The
     09:00 bar is the one that CONTAINS the 09:30 open -- and it was
     filtered out because 09:00 < 09:30. The resulting "09:30 4H bar"
     actually covered 10:00-14:00. The market open was missing entirely.

  2. Alpaca returns EXTENDED HOURS bars by default (04:00-20:00 ET). So the
     "13:30 4H bar" covered 14:00-18:00 (two hours of after-hours tape),
     and a phantom third bucket appeared at 17:30 made of pure evening data.

Every higher-timeframe signal computed on those bars was wrong.

The fix: build everything from 5-minute bars, explicitly filter to Regular
Trading Hours, and bucket from the session open. 5Min bars nest exactly into
15m / 30m / 1H / 2H / 4H boundaries, so no bar ever straddles a bucket edge.

Session buckets (standard 6.5h RTH session, 09:30-16:00 ET)
===========================================================
    4H:  09:30-13:30 | 13:30-16:00   (2nd bar is a 2.5h stub)
    2H:  09:30-11:30 | 11:30-13:30 | 13:30-15:30 | 15:30-16:00  (30m stub)
    1H:  09:30-10:30 | ... | 15:30-16:00  (30m stub)

The trailing stub bar is real -- it's what TradingView shows too, and it's
where a lot of afternoon resolution happens. It is NOT dropped.

Half-days (early 13:00 ET close) produce a shorter final bar. That's correct
and requires no special handling: we bucket whatever RTH data exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger("strat_scanner.bars")

ET = "America/New_York"

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)

# Bucket size in minutes for each session-aligned timeframe we build.
SESSION_TIMEFRAMES: dict[str, int] = {
    "15Min": 15,
    "30Min": 30,
    "1H": 60,
    "2H": 120,
    "4H": 240,
}

# How much history to pull (calendar days) to guarantee enough bars.
LOOKBACK_DAYS: dict[str, int] = {
    "5Min": 6,
    "15Min": 12,
    "30Min": 20,
    "1H": 30,
    "2H": 45,
    "4H": 60,
}

MIN_BARS_REQUIRED = 5


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------

def session_open(ts: pd.Timestamp) -> pd.Timestamp:
    """09:30 ET on the calendar day of `ts` (tz-aware, ET)."""
    return ts.normalize() + pd.Timedelta(hours=RTH_OPEN.hour, minutes=RTH_OPEN.minute)


def session_close(ts: pd.Timestamp) -> pd.Timestamp:
    """16:00 ET on the calendar day of `ts` (tz-aware, ET)."""
    return ts.normalize() + pd.Timedelta(hours=RTH_CLOSE.hour)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars whose START falls inside RTH.

    `between_time` is inclusive on both ends, so the upper bound is 15:55 --
    the last 5-minute bar of the session, which covers 15:55-16:00. Using
    16:00 would let the first after-hours bar in.
    """
    if df.empty:
        return df
    return df.between_time("09:30", "15:55")


def bucket_start(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """
    Which session-anchored bucket does bar-start `ts` belong to?

    Anchored at 09:30 of that day's session, NOT at midnight. This is the
    whole point of the module.
    """
    open_ = session_open(ts)
    elapsed = int((ts - open_).total_seconds() // 60)
    return open_ + pd.Timedelta(minutes=(elapsed // minutes) * minutes)


def bucket_end(start: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """
    When does the bucket beginning at `start` actually close?

    Clamped to the 16:00 session close, so the trailing stub bar reports its
    real end time rather than a time that never arrives. This matters: a
    bucket whose nominal end is 17:30 would never be considered "closed"
    and the scanner would sit there waiting forever.
    """
    nominal = start + pd.Timedelta(minutes=minutes)
    return min(nominal, session_close(start))


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def resample_session(df5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Aggregate RTH 5-minute bars into session-anchored buckets of `minutes`.

    Returns a DataFrame indexed by bucket start (ET) with columns:
        open, high, low, close, volume, bar_end
    """
    if df5.empty:
        return df5

    df = df5.copy()
    df["bucket"] = [bucket_start(ts, minutes) for ts in df.index]

    agg = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg.index.name = "timestamp"
    agg["bar_end"] = [bucket_end(ts, minutes) for ts in agg.index]
    return agg.sort_index()


def is_closed(bar_end: pd.Timestamp, now_et: Optional[pd.Timestamp] = None) -> bool:
    """Has this bar's period actually finished in real time?"""
    now_et = now_et or pd.Timestamp.now(tz=ET)
    return now_et >= bar_end


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

class BarProvider:
    """
    Fetches 5-minute RTH bars once per symbol per cycle, then derives every
    other timeframe from that single pull.

    v3 made one Alpaca request per (symbol, timeframe) -- 6 requests per
    symbol, 84 requests per scan for a 14-symbol watchlist. This makes 2
    (5Min + 1D). Everything intraday is a groupby on data we already have,
    which also guarantees the timeframes are mutually consistent.
    """

    def __init__(self, api_key: str, secret_key: str, data_feed: str = "iex"):
        self.client = StockHistoricalDataClient(api_key, secret_key)
        feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
        self.feed = feed_map.get(data_feed.lower(), DataFeed.IEX)
        if data_feed.lower() not in feed_map:
            logger.warning("Unknown ALPACA_DATA_FEED %r; defaulting to IEX.", data_feed)

    def _request(self, symbol: str, timeframe: TimeFrame, days: int) -> pd.DataFrame:
        start = datetime.now() - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            feed=self.feed,
        )
        df = self.client.get_stock_bars(req).df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df.index = df.index.tz_convert(ET)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def fetch(self, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Returns {timeframe_label: DataFrame}, all session-aligned and
        RTH-only, plus "5Min" (the base) and "1D".

        Every intraday frame here is derived from the same 5Min pull, so a
        15m close and a 4H high can never disagree with each other.
        """
        raw5 = self._request(symbol, TimeFrame(5, TimeFrameUnit.Minute), LOOKBACK_DAYS["4H"])
        if raw5.empty:
            logger.warning("No 5Min data returned for %s", symbol)
            return {}

        df5 = filter_rth(raw5)
        if df5.empty:
            logger.warning("No RTH 5Min data for %s", symbol)
            return {}

        out: dict[str, pd.DataFrame] = {}

        # Base timeframe. bar_end is trivially +5min (never straddles the close,
        # because 15:55 + 5m == 16:00 exactly).
        base = df5.copy()
        base["bar_end"] = base.index + pd.Timedelta(minutes=5)
        out["5Min"] = base

        for label, minutes in SESSION_TIMEFRAMES.items():
            resampled = resample_session(df5, minutes)
            if len(resampled) >= MIN_BARS_REQUIRED:
                out[label] = resampled

        daily = self._request(symbol, TimeFrame(1, TimeFrameUnit.Day), 180)
        if not daily.empty:
            daily = daily.copy()
            daily["bar_end"] = [session_close(ts) for ts in daily.index]
            out["1D"] = daily

        return out
