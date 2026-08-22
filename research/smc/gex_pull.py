"""
Pull SPY 0DTE option bars so GEX can be computed from data we already pay for.

Alpaca gives no greeks, no IV and no open interest, so:
  * pull 30-min bars for every strike within +/-30 of the open
  * derive session IV from the ATM straddle (one sigma per day, applied across
    strikes -- far more stable than inverting Black-Scholes per contract when
    time to expiry is a few hours and gamma explodes near the money)
  * weight by VOLUME, not open interest. For 0DTE that is the right measure:
    those contracts are opened and closed the same session.

Saved raw so the GEX construction can be re-run without re-pulling.
"""
import os
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv('.env')
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

OUT = Path('research/smc/data/gex')
OUT.mkdir(parents=True, exist_ok=True)
client = OptionHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])

spy = pd.read_parquet('research/smc/data/SPY_5m_ext.parquet').between_time('09:30', '15:59')
spy['d'] = spy.index.date
daily = spy.groupby('d').agg(o=('open', 'first'), h=('high', 'max'),
                             l=('low', 'min'), c=('close', 'last'))
days = [d for d in daily.index if d >= dt.date(2024, 8, 1)]
print(f'{len(days)} candidate sessions from {days[0]} to {days[-1]}', flush=True)

done = {p.stem for p in OUT.glob('*.parquet')}
ok = miss = 0
for n, day in enumerate(days):
    tag = day.strftime('%Y-%m-%d')
    if tag in done:
        continue
    spot = float(daily.loc[day, 'o'])
    base = int(round(spot))
    ymd = day.strftime('%y%m%d')
    syms = [f"SPY{ymd}{cp}{k*1000:08d}" for k in range(base - 30, base + 31) for cp in ("C", "P")]
    try:
        r = client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=syms, timeframe=TimeFrame(30, TimeFrameUnit.Minute),
            start=dt.datetime.combine(day, dt.time(0, 0)),
            end=dt.datetime.combine(day, dt.time(23, 59))))
        d = r.df
        if len(d) == 0:
            miss += 1
            continue
        d = d.reset_index()
        d['spot_open'] = spot
        d.to_parquet(OUT / f'{tag}.parquet')
        ok += 1
    except Exception as e:
        miss += 1
        if miss < 6:
            print(f'  {tag}: {type(e).__name__} {str(e)[:90]}', flush=True)
    if n % 25 == 0:
        print(f'  [{n}/{len(days)}] ok={ok} miss={miss}', flush=True)
    time.sleep(0.05)

print(f'DONE ok={ok} miss={miss} files={len(list(OUT.glob("*.parquet")))}', flush=True)
