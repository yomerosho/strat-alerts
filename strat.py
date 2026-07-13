"""
strat.py
--------
Strat bar classification. Pure functions over OHLC frames -- no I/O.

Bar labels, always relative to the PRIOR bar:
    1   Inside      high <= prev.high  AND  low >= prev.low
    2U  Directional high  > prev.high  AND  low >= prev.low
    2D  Directional low   < prev.low   AND  high <= prev.high
    3   Outside     high  > prev.high  AND  low  < prev.low
"""

from __future__ import annotations

import pandas as pd

INSIDE = "1"
UP = "2U"
DOWN = "2D"
OUTSIDE = "3"

BULL = "bull"
BEAR = "bear"
NEUTRAL = "neutral"


def label_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `label` column. First row is None (no prior bar to compare to)."""
    if df.empty:
        return df

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    labels: list[str | None] = [None]
    for i in range(1, len(df)):
        up = highs[i] > highs[i - 1]
        down = lows[i] < lows[i - 1]
        if up and down:
            labels.append(OUTSIDE)
        elif up:
            labels.append(UP)
        elif down:
            labels.append(DOWN)
        else:
            labels.append(INSIDE)

    out = df.copy()
    out["label"] = labels
    return out


def bar_direction(label: str | None) -> str:
    if label == UP:
        return BULL
    if label == DOWN:
        return BEAR
    return NEUTRAL


def continuity(closed_by_tf: dict[str, pd.DataFrame], timeframes: list[str], direction: str) -> tuple[int, int]:
    """
    FTFC as a score, not a gate.

    A timeframe "agrees" if its currently-forming bar is trading in
    `direction` relative to its own open -- i.e. green candle for bull,
    red for bear. That's the real Rob Smith definition of continuity:
    price above/below the open of the bar in progress.

    Falls back to the last closed bar's label if there's no forming bar
    (e.g. scanning after the close).

    Returns (agreeing, total).
    """
    agree = 0
    total = 0
    for tf in timeframes:
        df = closed_by_tf.get(tf)
        if df is None or df.empty:
            continue
        total += 1
        last = df.iloc[-1]
        if last["close"] > last["open"] and direction == BULL:
            agree += 1
        elif last["close"] < last["open"] and direction == BEAR:
            agree += 1
    return agree, total
