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

# A trailing session bucket shorter than this fraction of its nominal length
# is folded into the bar before it. 0.5 keeps the 4H 13:30 bar (2.5h = 62%)
# and eliminates the 2H 15:30 bar (0.5h = 25%).
MIN_BAR_COMPLETENESS = 0.5


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

def resample_session(df5: pd.DataFrame, minutes: int, merge_stub: bool = True) -> pd.DataFrame:
    """
    Aggregate RTH 5-minute bars into session-anchored buckets of `minutes`.

    THE STUB PROBLEM
    ----------------
    RTH is 6.5 hours, which divides evenly into nothing useful. So the final
    bucket of each session is short:

        4H:  13:30-16:00  =  2.5h of a 4h bucket   (62% -- a real bar)
        2H:  15:30-16:00  =  0.5h of a 2h bucket   (25% -- NOT a real bar)

    A 30-minute candle has a 30-minute range. It is therefore almost always
    strictly INSIDE the 2-hour bar before it -- which means it arms levels
    constantly, off a range so tight the target sits practically on top of
    the trigger. Observed live on QQQ: a "2H inside bar" that produced a
    0.12 reward-to-risk. That's not a setup, it's an artifact of the clock.

    So: if the trailing bucket is under `MIN_BAR_COMPLETENESS` of its nominal
    length, fold it into the previous bucket of the same session.

        2H final bar becomes 13:30-16:00 (2.5h). 4H is unaffected.

    NOTE FOR CHART COMPARISON: TradingView WILL show that separate 15:30
    half-candle on a 2H chart. The scanner deliberately does not. Every other
    bar matches exactly -- only the last 2H bar of the session differs, and it
    differs on purpose. Don't be alarmed when you're eyeballing the two.
    """
    if df5.empty:
        return df5

    df = df5.copy()
    df["bucket"] = [bucket_start(ts, minutes) for ts in df.index]

    absorbed: set = set()
    if merge_stub:
        df["bucket"], absorbed = _merge_trailing_stubs(df["bucket"], minutes)

    agg = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg.index.name = "timestamp"
    agg = agg.sort_index()
    agg["bar_end"] = _bar_ends(agg.index, minutes, absorbed)
    return agg


def _bar_ends(starts, minutes: int, absorbed: set) -> list[pd.Timestamp]:
    """
    When each bucket actually closes.

    Normally: start + minutes, clamped to the session close.

    But a bucket that ABSORBED a trailing stub genuinely runs to 16:00, and
    must say so. If it reported its nominal end instead, it would be marked
    "closed" at 15:30 while still swallowing 15:30-16:00 bars -- the scanner
    would arm off a bar that was still changing underneath it.

    The absorbed set is passed in rather than inferred from "is this the last
    bucket of the session", because mid-session the last bucket present in
    the data is simply the one in progress -- at 11:00 the 09:30 4H bar is
    the only bucket that exists, and it ends at 13:30, not at the close.
    """
    ends = []
    for ts in starts:
        if ts in absorbed:
            ends.append(session_close(ts))
        else:
            ends.append(min(ts + pd.Timedelta(minutes=minutes), session_close(ts)))
    return ends


def _merge_trailing_stubs(buckets: pd.Series, minutes: int) -> tuple[pd.Series, set]:
    """
    Reassign any too-short final bucket of a session to the bucket before it.

    Returns (remapped_labels, set_of_buckets_that_absorbed_a_stub).

    Works on the bucket-label column directly: relabel the stub's rows with
    the previous bucket's start, and the groupby that follows merges them for
    free.
    """
    remapped = buckets.copy()
    absorbed: set = set()

    for _, day_buckets in buckets.groupby(buckets.dt.date):
        starts = sorted(day_buckets.unique())
        if len(starts) < 2:
            continue

        last = starts[-1]
        span_minutes = (bucket_end(last, minutes) - last).total_seconds() / 60
        if span_minutes < minutes * MIN_BAR_COMPLETENESS:
            remapped[remapped == last] = starts[-2]
            absorbed.add(starts[-2])

    return remapped, absorbed


def resample_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly bars from daily bars. The top rung of the magnitude ladder.

    Weeks are Monday-Friday and are labelled by their MONDAY, so the index is
    a real trading timestamp rather than a pandas period boundary (W-FRI starts
    on a Saturday, which is not a thing).

    bar_end is always Friday 16:00 of that week, computed from the label rather
    than from the last daily bar present. That matters: on Wednesday the current
    week only HAS three daily bars, and if bar_end were derived from them the
    week would be marked closed while still absorbing Thursday and Friday --
    the exact class of bug that corrupted v3's 4H bars.
    """
    if daily.empty:
        return daily

    df = daily.copy()
    mondays = [ts.normalize() - pd.Timedelta(days=ts.weekday()) for ts in df.index]
    df["_week"] = mondays

    agg = df.groupby("_week").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg.index.name = "timestamp"
    agg = agg.sort_index()
    agg["bar_end"] = [session_close(ts + pd.Timedelta(days=4)) for ts in agg.index]
    return agg


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

    def fetch(self, symbol: str, lookback_days: Optional[int] = None) -> dict[str, pd.DataFrame]:
        """
        Returns {timeframe_label: DataFrame}, all session-aligned and
        RTH-only, plus "5Min" (the base) and "1D".

        Every intraday frame here is derived from the same 5Min pull, so a
        15m close and a 4H high can never disagree with each other.

        lookback_days overrides how much 5-minute history to pull (and hence
        every intraday frame). The live scanner leaves it None -- pulling 60
        days every cycle is right for alerting. A backtest passes a big number
        (e.g. 365) to validate out-of-sample without changing production fetch.
        """
        days5 = lookback_days or LOOKBACK_DAYS["4H"]
        raw5 = self._request(symbol, TimeFrame(5, TimeFrameUnit.Minute), days5)
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

        # The LAST TRADED PRICE, extended hours included.
        #
        # Bars are RTH-only and must stay that way -- letting premarket tape into
        # the buckets is exactly what corrupted v3's higher timeframes. But
        # "where is price right now" is a different question from "what shape is
        # the candle", and conflating them made premarket ARM alerts quote
        # Friday's 4pm close as though it were live. The levels were right and
        # the distances were fiction.
        #
        # So: candles from RTH, current price from the full tape.
        out["_last"] = raw5.tail(1)

        for label, minutes in SESSION_TIMEFRAMES.items():
            resampled = resample_session(df5, minutes)
            if len(resampled) >= MIN_BARS_REQUIRED:
                out[label] = resampled

        daily = self._request(symbol, TimeFrame(1, TimeFrameUnit.Day), 400)
        if not daily.empty:
            daily = daily.copy()
            daily["bar_end"] = [session_close(ts) for ts in daily.index]
            out["1D"] = daily

            # Weekly: the top rung of the magnitude ladder. Without it, a setup
            # that has cleared every daily level looks like it has nowhere left
            # to go, when in fact the week's prior high may be 4R above.
            weekly = resample_weekly(daily)
            if len(weekly) >= 3:
                out["1W"] = weekly

        return out
