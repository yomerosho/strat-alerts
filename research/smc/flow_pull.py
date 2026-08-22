"""
Pull trade prints and turn them into signed order flow.

Quotes would give the cleanest buy/sell classification but cost ~2.6 min per
symbol-day, which is not tractable for a real sample. Trades alone cost ~30s a
day, so classification uses the TICK RULE: a print above the previous price is
buyer-initiated, below is seller-initiated, equal carries the last sign. On a
name as liquid as SPY that agrees with the quote rule roughly 85-90% of the
time -- good enough to detect an effect, and its errors are symmetric so they
dilute a signal rather than manufacture one.

Raw prints are never stored. Each day is aggregated on the fly into 1-minute
buckets, which keeps the whole study a few megabytes.
"""
import os
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env')
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.data.enums import DataFeed

import sys
SYMS = [sys.argv[1]] if len(sys.argv) > 1 else ['SPY', 'TSLA', 'NVDA']
NDAYS = 45
BIG = 1000          # shares; "block" threshold
OUT = Path('research/smc/data/flow')
OUT.mkdir(parents=True, exist_ok=True)
client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

spy = pd.read_parquet('research/smc/data/SPY_5m_ext.parquet').between_time('09:30', '15:59')
all_days = sorted({d for d in spy.index.date})
days = all_days[-NDAYS:]
print(f'{len(days)} sessions: {days[0]} -> {days[-1]}', flush=True)

done = {p.stem for p in OUT.glob('*.parquet')}
for sym in SYMS:
    for i, day in enumerate(days):
        tag = f'{sym}_{day}'
        if tag in done:
            continue
        # RTH in UTC (ET+4 during DST); pull a wide window and filter after
        s = dt.datetime.combine(day, dt.time(13, 25))
        e = dt.datetime.combine(day, dt.time(20, 5))
        try:
            df = client.get_stock_trades(StockTradesRequest(
                symbol_or_symbols=sym, start=s, end=e, feed=DataFeed.SIP)).df
        except Exception as ex:
            print(f'  {tag}: ERR {type(ex).__name__} {str(ex)[:80]}', flush=True)
            continue
        if len(df) == 0:
            continue
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('America/New_York')
        df = df[(df.timestamp.dt.time >= dt.time(9, 30)) & (df.timestamp.dt.time < dt.time(16, 0))]
        if len(df) < 500:
            continue

        # vectorised tick rule: sign of the price change, zeros carried forward
        p = df.price.values
        raw = np.sign(np.diff(p, prepend=p[0])).astype(np.int8)
        s_ser = pd.Series(np.where(raw == 0, np.nan, raw)).ffill().bfill().fillna(1)
        df['sign'] = s_ser.values.astype(np.int8)
        sz = df['size'].values.astype(float)
        big = sz >= BIG
        pos = df['sign'].values > 0
        df['m'] = df.timestamp.dt.floor('1min')
        df['_buy'] = np.where(pos, sz, 0.0)
        df['_sell'] = np.where(~pos, sz, 0.0)
        df['_bbuy'] = np.where(big & pos, sz, 0.0)
        df['_bsell'] = np.where(big & ~pos, sz, 0.0)

        g = df.groupby('m', sort=True).agg(
            px=('price', 'last'), vol=('size', 'sum'), nt=('price', 'size'),
            buy=('_buy', 'sum'), sell=('_sell', 'sum'),
            big_buy=('_bbuy', 'sum'), big_sell=('_bsell', 'sum')).reset_index()
        g['sym'] = sym
        g['day'] = day
        g.to_parquet(OUT / f'{tag}.parquet')
        if i % 15 == 0:
            print(f'  {sym} [{i}/{len(days)}] {day} rows={len(g)}', flush=True)
        time.sleep(0.05)
    print(f'{sym} done', flush=True)

print('ALL DONE files=', len(list(OUT.glob('*.parquet'))), flush=True)
