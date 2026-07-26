"""
Fetch and cache 5-minute + daily bars for the ORB backtest.

5-min: 2021-11-01 -> present  (buffer before 2022-01-01 for 20-session RVOL
       and 14-day OR-average warmup)
Daily: 2021-01-01 -> present  (buffer for 14-day ATR + 6-month percentile
       window before the in-sample start)

Feed: SIP (consolidated tape -- required for RVOL to mean anything).
Prices: raw (unadjusted), which is what a live trader sees intraday.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ET = "America/New_York"
SYMBOLS = ["SPY", "QQQ", "IWM"]
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(Path(__file__).parents[2] / ".env")

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
)


def fetch(symbol: str, timeframe: TimeFrame, start: str) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=datetime.fromisoformat(start),
        feed=DataFeed.SIP,
        adjustment=Adjustment.RAW,
    )
    df = client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df.index = df.index.tz_convert(ET)
    return df[["open", "high", "low", "close", "volume"]].sort_index()


for sym in SYMBOLS:
    p5 = DATA_DIR / f"{sym}_5min.parquet"
    pd_daily = DATA_DIR / f"{sym}_daily.parquet"

    if not p5.exists():
        df5 = fetch(sym, TimeFrame(5, TimeFrameUnit.Minute), "2021-11-01")
        df5.to_parquet(p5)
        print(f"{sym} 5min: {len(df5)} bars  {df5.index[0]} -> {df5.index[-1]}")
    else:
        print(f"{sym} 5min: cached")

    if not pd_daily.exists():
        dfd = fetch(sym, TimeFrame(1, TimeFrameUnit.Day), "2021-01-01")
        dfd.to_parquet(pd_daily)
        print(f"{sym} daily: {len(dfd)} bars  {dfd.index[0]} -> {dfd.index[-1]}")
    else:
        print(f"{sym} daily: cached")

print("done")
