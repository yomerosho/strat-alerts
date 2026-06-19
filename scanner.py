"""
scanner.py
----------
"The Strat" (Rob Smith methodology) pattern detection on top of Alpaca
market data, for the 4-Hour, Daily, Weekly, and Monthly timeframes.

Strat bar labels (relative to the PRIOR completed bar):
    1   Inside bar   -> high <= prev.high and low  >= prev.low
    2U  Directional  -> high  > prev.high and low  >= prev.low
    2D  Directional  -> low  < prev.low  and high <= prev.high
    3   Outside bar  -> high  > prev.high and low  <  prev.low

A "trigger" is a live breach of the most recently COMPLETED bar's high/low
by the bar currently still forming:
    bullish trigger -> current price > last_completed_bar.high
    bearish trigger -> current price < last_completed_bar.low

4-Hour bars are not native to most feeds, so they're built by resampling
1-Hour bars into 4-hour buckets anchored to the 9:30 ET market open
(9:30-13:30, 13:30-16:00 for a standard 6.5h session).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logger = logging.getLogger("strat_scanner.scanner")

STRAT_INSIDE = "1"
STRAT_2U = "2U"
STRAT_2D = "2D"
STRAT_OUTSIDE = "3"

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


@dataclass(frozen=True)
class StratState:
    """Snapshot of a symbol's Strat status on one timeframe."""
    symbol: str
    timeframe: str
    last_bar_time: pd.Timestamp
    last_three_labels: tuple[str, str, str]  # most recent 3 completed bars, oldest->newest
    last_completed_high: float
    last_completed_low: float
    current_price: float
    trigger: Optional[str]  # "bullish_trigger" | "bearish_trigger" | None

    @property
    def setup_key(self) -> str:
        """A compact string used for debounce comparisons -- changes only
        when something alert-worthy actually changes."""
        trig = self.trigger or "none"
        return f"{self.last_three_labels}|{trig}"

    def to_dict(self) -> dict:
        """JSON-serializable snapshot, used for the dashboard's data feed."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "last_bar_time": self.last_bar_time.isoformat(),
            "last_three_labels": list(self.last_three_labels),
            "last_completed_high": self.last_completed_high,
            "last_completed_low": self.last_completed_low,
            "current_price": self.current_price,
            "trigger": self.trigger,
        }

    @property
    def direction(self) -> str:
        """bull / bear / neutral -- a live trigger wins, otherwise fall back
        to the last completed bar's label. Shared by FTFC computation in
        main.py and by the dashboard so both apply the same rule."""
        if self.trigger == "bullish_trigger":
            return "bull"
        if self.trigger == "bearish_trigger":
            return "bear"
        last_label = self.last_three_labels[-1] if self.last_three_labels else None
        if last_label == "2U":
            return "bull"
        if last_label == "2D":
            return "bear"
        return "neutral"


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'strat_label' column to an OHLC dataframe (needs prior bar)."""
    labels = [None]
    for i in range(1, len(df)):
        prev_high, prev_low = df["high"].iloc[i - 1], df["low"].iloc[i - 1]
        high, low = df["high"].iloc[i], df["low"].iloc[i]

        broke_up = high > prev_high
        broke_down = low < prev_low

        if broke_up and broke_down:
            labels.append(STRAT_OUTSIDE)
        elif broke_up:
            labels.append(STRAT_2U)
        elif broke_down:
            labels.append(STRAT_2D)
        else:
            labels.append(STRAT_INSIDE)

    out = df.copy()
    out["strat_label"] = labels
    return out


def detect_trigger(df: pd.DataFrame, current_price: float) -> Optional[str]:
    """Compare current price to the last COMPLETED bar's high/low."""
    if len(df) == 0:
        return None
    last = df.iloc[-1]
    if current_price > last["high"]:
        return "bullish_trigger"
    if current_price < last["low"]:
        return "bearish_trigger"
    return None


INTRADAY_DURATIONS = {
    "5Min": pd.Timedelta(minutes=5),
    "15Min": pd.Timedelta(minutes=15),
    "30Min": pd.Timedelta(minutes=30),
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
}


def bar_period_has_closed(bar_time: pd.Timestamp, timeframe_label: str) -> bool:
    """
    Whether a bar's own time period has actually finished in real time.

    The most recent bar Alpaca returns isn't always still "forming" -- if
    the scan runs after market close, that day's Daily bar (or this 4H
    bucket) is already complete. Treating it as forming anyway silently
    discards a fully real bar and makes everything look one period stale
    (e.g. always showing yesterday's date on the Daily tab, even at 10pm).
    """
    now_et = pd.Timestamp.now(tz="US/Eastern")
    if timeframe_label in INTRADAY_DURATIONS:
        return now_et >= bar_time + INTRADAY_DURATIONS[timeframe_label]
    if timeframe_label == "1D":
        # Alpaca's daily bar timestamp lands at midnight ET of the trading
        # day; that session's data is final once the 16:00 ET close passes.
        close_time = bar_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return now_et >= close_time
    if timeframe_label == "1W":
        return now_et.isocalendar()[:2] > bar_time.isocalendar()[:2]
    if timeframe_label == "1M":
        return (now_et.year, now_et.month) > (bar_time.year, bar_time.month)
    return True


def resample_to_4h(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 4-hour bars anchored to market open (9:30 ET) from 1-hour bars.

    Assumes hourly_df.index is a tz-aware DatetimeIndex already converted to
    US/Eastern. Buckets per trading day: 09:30-13:30, 13:30-17:30 (the
    second bucket naturally truncates at the 16:00 close since no bars exist
    past it).
    """
    if hourly_df.empty:
        return hourly_df

    df = hourly_df.copy()
    df["session_date"] = df.index.date

    rows = []
    for _, day_df in df.groupby("session_date"):
        day_df = day_df.sort_index()
        anchor = day_df.index[0].normalize() + pd.Timedelta(
            hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE
        )
        bucket_edges = [anchor + pd.Timedelta(hours=4 * i) for i in range(0, 3)]
        for start, end in zip(bucket_edges, bucket_edges[1:] + [anchor + pd.Timedelta(hours=12)]):
            bucket = day_df[(day_df.index >= start) & (day_df.index < end)]
            if bucket.empty:
                continue
            rows.append({
                "timestamp": start,
                "open": bucket["open"].iloc[0],
                "high": bucket["high"].max(),
                "low": bucket["low"].min(),
                "close": bucket["close"].iloc[-1],
                "volume": bucket["volume"].sum(),
            })

    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out


class StratScanner:
    """Fetches bars from Alpaca and computes Strat state per symbol/timeframe."""

    def __init__(self, api_key: str, secret_key: str, data_feed: str = "iex"):
        self.client = StockHistoricalDataClient(api_key, secret_key)
        feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
        self.feed = feed_map.get(data_feed.lower(), DataFeed.IEX)
        if data_feed.lower() not in feed_map:
            logger.warning("Unknown ALPACA_DATA_FEED '%s', defaulting to IEX.", data_feed)

    def _fetch(self, symbol: str, timeframe: TimeFrame, start: datetime) -> pd.DataFrame:
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start, feed=self.feed
        )
        bars = self.client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return df
        # alpaca-py returns a MultiIndex (symbol, timestamp) when multiple
        # symbols are requested; normalize to a plain DatetimeIndex.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df.index = df.index.tz_convert("US/Eastern")
        return df[["open", "high", "low", "close", "volume"]]

    def get_state(self, symbol: str, timeframe_label: str) -> Optional[StratState]:
        """timeframe_label is one of '5Min', '15Min', '30Min', '1H', '4H', '1D'."""
        now = datetime.utcnow()

        if timeframe_label == "5Min":
            df = self._fetch(symbol, TimeFrame(5, TimeFrameUnit.Minute), now - timedelta(days=5))
        elif timeframe_label == "15Min":
            df = self._fetch(symbol, TimeFrame(15, TimeFrameUnit.Minute), now - timedelta(days=10))
        elif timeframe_label == "30Min":
            df = self._fetch(symbol, TimeFrame(30, TimeFrameUnit.Minute), now - timedelta(days=15))
        elif timeframe_label == "1H":
            df = self._fetch(symbol, TimeFrame(1, TimeFrameUnit.Hour), now - timedelta(days=20))
        elif timeframe_label == "4H":
            hourly = self._fetch(symbol, TimeFrame(1, TimeFrameUnit.Hour), now - timedelta(days=20))
            df = resample_to_4h(hourly)
        elif timeframe_label == "1D":
            df = self._fetch(symbol, TimeFrame(1, TimeFrameUnit.Day), now - timedelta(days=180))
        else:
            raise ValueError(f"Unknown timeframe label: {timeframe_label}")

        if df is None or df.empty or len(df) < 4:
            logger.warning("Not enough %s bars for %s to evaluate.", timeframe_label, symbol)
            return None

        labeled = label_bars(df)
        last_bar_time = labeled.index[-1]

        if bar_period_has_closed(last_bar_time, timeframe_label):
            # The most recent bar is already final -- include it as the
            # latest completed bar rather than discarding it. There's no
            # live intrabar price beyond it (we only have historical bars
            # here), so there's nothing new to compare for a trigger.
            completed = labeled
            current_price = float(completed["close"].iloc[-1])
            trigger = None
        else:
            completed = labeled.iloc[:-1]
            forming = labeled.iloc[-1]
            current_price = float(forming["close"])
            trigger = detect_trigger(completed, current_price=current_price)

        if len(completed) < 4:
            logger.warning("Not enough completed %s bars for %s to evaluate.", timeframe_label, symbol)
            return None

        last_three = tuple(completed["strat_label"].iloc[-3:].fillna("?"))

        return StratState(
            symbol=symbol,
            timeframe=timeframe_label,
            last_bar_time=completed.index[-1],
            last_three_labels=last_three,  # type: ignore[arg-type]
            last_completed_high=float(completed["high"].iloc[-1]),
            last_completed_low=float(completed["low"].iloc[-1]),
            current_price=current_price,
            trigger=trigger,
        )
