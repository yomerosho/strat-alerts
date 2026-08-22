"""
Build a daily GEX series from the raw 0DTE bars, then test the three claims
the gamma-exposure thesis actually rests on.

CONSTRUCTION
  For each session, using only the 09:30-10:30 bars (so the signal is knowable
  at 10:30 and everything measured against it is strictly in the future):
    sigma   from the ATM straddle price, inverted for a 0DTE horizon
    gamma   Black-Scholes gamma per strike at that sigma
    GEX     sum( gamma * volume * 100 * S^2 * 0.01 ), calls +ve, puts -ve
            (the standard dealer-long-calls / short-puts convention)
    flip    the strike where the cumulative strike profile crosses zero
    magnet  the strike carrying the largest absolute gamma

TESTS
  A  does GEX predict the REST OF DAY realised volatility and range?
     This is the core dealer-hedging mechanism: long gamma suppresses moves,
     short gamma amplifies them.
  B  does the magnet strike pull the close toward it, versus a random strike
     the same distance away? This is the "price magnet" claim.
  C  does GEX regime condition the intraday VWAP scalp, which lost -1.17bp
     gross overall? If positive gamma really is mean-reverting, the scalp
     should be less bad, or good, on those days.
"""
import glob
import numpy as np
import pandas as pd
from math import log, sqrt, exp, pi

RAW = sorted(glob.glob('research/smc/data/gex/*.parquet'))
print(f'{len(RAW)} session files')

spy = pd.read_parquet('research/smc/data/SPY_5m_ext.parquet').between_time('09:30', '15:59')
spy['d'] = spy.index.date


def bs_gamma(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + 0.5 * sig * sig * T) / (sig * sqrt(T))
    return exp(-0.5 * d1 * d1) / (S * sig * sqrt(T) * sqrt(2 * pi))


rows = []
for f in RAW:
    day = pd.Timestamp(f.split('\\')[-1].split('/')[-1].replace('.parquet', '')).date()
    d = pd.read_parquet(f)
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True).dt.tz_convert('America/New_York')
    sess = spy[spy.d == day]
    if len(sess) < 60:
        continue
    spot_open = float(d.spot_open.iloc[0])

    # signal window only: 09:30-10:30
    w = d[(d.timestamp.dt.time >= pd.Timestamp('09:30').time()) &
          (d.timestamp.dt.time < pd.Timestamp('10:30').time())]
    if len(w) < 10:
        continue
    w = w.copy()
    w['cp'] = w.symbol.str[9]
    w['strike'] = w.symbol.str[-8:].astype(float) / 1000.0

    px1030 = sess.between_time('10:25', '10:30').close
    if len(px1030) == 0:
        continue
    S = float(px1030.iloc[-1])

    # session sigma from the ATM straddle at the end of the window
    last = w.sort_values('timestamp').groupby(['strike', 'cp']).close.last().unstack()
    if last.empty or 'C' not in last or 'P' not in last:
        continue
    atmK = min(last.index, key=lambda k: abs(k - S))
    strad = last.loc[atmK].get('C', np.nan) + last.loc[atmK].get('P', np.nan)
    if not np.isfinite(strad) or strad <= 0:
        continue
    T = 5.5 / 6.5 / 252.0          # ~5.5 trading hours left, in years
    sig = strad / (0.7979 * S * sqrt(T))
    if not (0.01 < sig < 5):
        continue

    vol = w.groupby(['strike', 'cp']).volume.sum().unstack().fillna(0.0)
    gex_by_k = {}
    for K in vol.index:
        g = bs_gamma(S, K, T, sig)
        cv = vol.loc[K].get('C', 0.0)
        pv = vol.loc[K].get('P', 0.0)
        gex_by_k[K] = g * (cv - pv) * 100.0 * S * S * 0.01
    ser = pd.Series(gex_by_k).sort_index()
    total_gex = ser.sum()
    magnet = float(ser.abs().idxmax())
    cum = ser.cumsum()
    flip = np.nan
    sgn = np.sign(cum.values)
    for i in range(1, len(sgn)):
        if sgn[i - 1] != 0 and sgn[i] != 0 and sgn[i - 1] != sgn[i]:
            x0, x1 = cum.index[i - 1], cum.index[i]
            y0, y1 = cum.values[i - 1], cum.values[i]
            flip = x0 + (x1 - x0) * (-y0) / (y1 - y0)
            break

    rest = sess.between_time('10:30', '15:59')
    if len(rest) < 30:
        continue
    r5 = rest.close.pct_change().dropna()
    rows.append(dict(
        day=day, spot_open=spot_open, S1030=S, sigma=sig,
        gex=total_gex, gex_norm=total_gex / 1e9, magnet=magnet, flip=flip,
        rest_rv=float(r5.std() * np.sqrt(78) * 100),          # rest-of-day realised vol, %
        rest_range=float((rest.high.max() - rest.low.min()) / S * 100),
        rest_ret=float(rest.close.iloc[-1] / S - 1) * 100,
        close=float(rest.close.iloc[-1]),
        dist_to_magnet=float((magnet - S) / S * 100),
    ))

G = pd.DataFrame(rows).sort_values('day').reset_index(drop=True)
G['close_to_magnet'] = (G.close - G.magnet).abs() / G.S1030 * 100
G['open_to_magnet'] = (G.magnet - G.S1030).abs() / G.S1030 * 100
G.to_csv('research/smc/gex_daily.csv', index=False)
print(f'built {len(G)} sessions  {G.day.min()} -> {G.day.max()}')
print(G[['gex_norm', 'sigma', 'rest_rv', 'rest_range']].describe().round(3).to_string())
