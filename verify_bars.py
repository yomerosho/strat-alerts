"""
verify_bars.py
--------------
THE check. Prints the actual 2H and 4H candles the scanner is building, from
live Alpaca data, so you can hold them next to your TradingView chart.

This exists because v3 silently built its 4H bars wrong -- the 09:30 candle
didn't contain the market open, and the afternoon candle was polluted with
after-hours tape. Every signal it produced was computed on candles that don't
exist on any chart you've ever looked at. Nothing in the code complained.

So: don't trust v4 because the tests pass. Trust it because you compared the
numbers below to your chart with your own eyes.

    python verify_bars.py           # SPY, last 3 sessions
    python verify_bars.py TSLA      # a specific symbol
    python verify_bars.py TSLA 5    # ...and more sessions

WHAT TO COMPARE
---------------
Open TradingView, set the symbol, set the timeframe to 4H, and make sure
"Extended Hours" (ETH) is OFF. The candles should line up: same timestamps,
same OHLC, same Strat labels. TradingView's own 4H bars for US equities are
session-anchored, which is exactly what we're now doing.

If the highs and lows match, the levels are real and everything downstream
is trustworthy. If they don't, STOP -- do not let it alert you.
"""

from __future__ import annotations

import sys

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bars import BarProvider, filter_rth, resample_session
from config import CONFIG
from levels import split_closed_forming
from strat import label_bars

ET = "America/New_York"


def show(symbol: str, sessions: int) -> None:
    if not CONFIG.alpaca_api_key:
        print("ERROR: ALPACA_API_KEY not set.")
        sys.exit(1)

    provider = BarProvider(
        CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key, CONFIG.alpaca_data_feed
    )
    now = pd.Timestamp.now(tz=ET)

    print(f"\n{'=' * 74}")
    print(f"  {symbol}   feed={CONFIG.alpaca_data_feed.upper()}   now={now:%Y-%m-%d %H:%M} ET")
    print(f"{'=' * 74}")

    raw5 = provider._request(
        symbol, TimeFrame(5, TimeFrameUnit.Minute), sessions * 3 + 10
    )
    if raw5.empty:
        print("  No data returned.")
        return

    df5 = filter_rth(raw5)

    # Sanity: prove extended-hours bars were actually stripped
    dropped = len(raw5) - len(df5)
    print(f"\n  5-minute bars: {len(raw5)} fetched, {dropped} extended-hours dropped, {len(df5)} RTH kept")
    if not df5.empty:
        earliest = min(t.strftime("%H:%M") for t in df5.index)
        latest = max(t.strftime("%H:%M") for t in df5.index)
        print(f"  RTH range in data: {earliest} - {latest}   (must be 09:30 - 15:55)")
        if earliest != "09:30":
            print("  *** WARNING: earliest RTH bar is not 09:30. The open may be missing. ***")

    days = sorted({t.date() for t in df5.index})[-sessions:]
    df5 = df5[[t.date() in days for t in df5.index]]

    for label, minutes in (("4H", 240), ("2H", 120)):
        bars = label_bars(resample_session(df5, minutes))
        closed, forming = split_closed_forming(bars, now)

        print(f"\n  {'-' * 70}")
        print(f"  {label} CANDLES   (compare to TradingView, Extended Hours OFF)")
        print(f"  {'-' * 70}")
        print(f"  {'BAR START':<18} {'OPEN':>9} {'HIGH':>9} {'LOW':>9} {'CLOSE':>9}  "
              f"{'STRAT':<6} STATE")

        for ts, r in bars.iterrows():
            is_forming = forming is not None and ts == forming.name
            state = "FORMING" if is_forming else "closed"
            lab = r["label"] if isinstance(r["label"], str) else "-"
            marker = " <-- ARMS HERE" if (lab == "1" and not is_forming
                                          and ts == closed.index[-1]) else ""
            print(
                f"  {ts:%Y-%m-%d %H:%M}   {r['open']:>9.2f} {r['high']:>9.2f} "
                f"{r['low']:>9.2f} {r['close']:>9.2f}  {lab:<6} {state}{marker}"
            )

        if len(closed) and closed["label"].iloc[-1] == "1":
            inside = closed.iloc[-1]
            print(f"\n  >> INSIDE BAR on {label}. Armed levels:")
            print(f"       BULL: close above {inside['high']:.2f}")
            print(f"       BEAR: close below {inside['low']:.2f}")
            print(f"     Put these two numbers on your chart. They should sit exactly")
            print(f"     on the high and low of that {label} inside candle.")

    print()


if __name__ == "__main__":
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    show(sym, n)
