"""
FVG-3R against the same standard the RSI(2) strategy passed.

Sample: sweep -> displacement -> limit at the 50% of the fair value gap, LONG,
daily close above its 200SMA, 3R target, stop beyond the sweep extreme.
Deduplicated to ONE trade per ticker-day (raw signals overlap ~2.8x, which would
make any CI far too narrow).

The two tests that matter:
  1. DATE-CLUSTERED bootstrap. Trades on the same calendar date share a market
     move, so resampling individual trades overstates precision. Resample whole
     dates instead.
  2. MATCHED RANDOM CONTROL. For every real trade, a coin-flip entry on the SAME
     ticker, the SAME day, with the SAME R distance and the SAME 3R target and
     hold. Identical instrument, direction, regime and geometry -- only the
     timing differs. If the FVG timing carries information, it beats this.
     If it does not, the +0.17R is long drift in a bull market.
"""
import os
import numpy as np
import pandas as pd

HOLD5 = 78 * 3
TGT = 3.0
rng = np.random.default_rng(20260822)

T = pd.read_csv('research/smc/smc_strategy2.csv')
F = T[(T.filled == 1) & (T.zone == 'fvg_mid') & (T.dir == 1) & (T.tu == True)].copy()
F['day'] = pd.to_datetime(F.day).dt.date
F['ts'] = pd.to_datetime(F.ts, utc=True).dt.tz_convert('America/New_York')
F = F.sort_values('ts').groupby(['sym', 'day']).head(1).reset_index(drop=True)
F['y'] = F.ts.dt.year
print(f'deduped sample: n={len(F)}  tickers={F.sym.nunique()}  '
      f'{F.day.min()} -> {F.day.max()}')


def clustered_ci(df, col, key='day', iters=8000):
    """Resample whole dates, not individual trades."""
    groups = [g[col].values.astype(float) for _, g in df.groupby(key)]
    k = len(groups)
    means = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, k, k)
        vals = np.concatenate([groups[j] for j in pick])
        means[i] = vals.mean()
    return np.percentile(means, 2.5), np.percentile(means, 97.5), (means > 0).mean()


x = F.r3.values.astype(float)
lo_n, hi_n, _ = (np.percentile(rng.choice(x, (8000, len(x)), replace=True).mean(1), [2.5, 97.5]).tolist() + [0])[:3]
lo_c, hi_c, p_c = clustered_ci(F, 'r3')
print('\n' + '=' * 96)
print('1  Does the confidence interval survive clustering by date?')
print('=' * 96)
print(f"  expR(3R) = {x.mean():+.3f}   win {F.w3.mean():.0%}  (break-even 25%)")
print(f"  naive bootstrap CI      [{lo_n:+.3f}, {hi_n:+.3f}]")
print(f"  date-clustered CI       [{lo_c:+.3f}, {hi_c:+.3f}]   P(>0) = {p_c:.3f}")

print('\n' + '=' * 96)
print('2  Stability')
print('=' * 96)
py = F.groupby('y').agg(n=('r3', 'size'), win=('w3', 'mean'), expR=('r3', 'mean')).round(3)
print(py.to_string())
pt = F.groupby('sym').agg(n=('r3', 'size'), win=('w3', 'mean'), expR=('r3', 'mean')).round(3)
print()
print(pt.sort_values('expR').to_string())
print(f"\n  tickers positive: {(pt.expR > 0).sum()}/{len(pt)}   years positive: {(py.expR > 0).sum()}/{len(py)}")
tr = F[F.y < 2025].r3
ho = F[F.y >= 2025].r3
print(f"  train 2022-24 {tr.mean():+.3f} (n={len(tr)})   holdout 2025-26 {ho.mean():+.3f} (n={len(ho)})")

# ---------------------------------------------------------------- control
print('\n' + '=' * 96)
print('3  MATCHED CONTROL: same ticker, same day, same R, same target -- random entry time')
print('=' * 96)
ctrl_rows = []
for sym, grp in F.groupby('sym'):
    path = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(path):
        continue
    d5 = pd.read_parquet(path).between_time('09:30', '15:59')
    m15 = d5.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                    'close': 'last', 'volume': 'sum'}).dropna()
    m15 = m15[m15.volume > 0]
    tr_ = np.maximum(m15.high - m15.low,
                     np.maximum((m15.high - m15.close.shift()).abs(),
                                (m15.low - m15.close.shift()).abs()))
    atr15 = tr_.rolling(28).mean().shift(1)
    day15 = pd.Series(m15.index.date, index=m15.index)
    f5, f5h, f5l = d5.index, d5.high.values, d5.low.values
    c5 = d5.close.values

    for _, r in grp.iterrows():
        bars = m15[day15.values == r.day]
        if len(bars) < 6:
            continue
        pick = rng.integers(1, len(bars) - 1)
        t0 = bars.index[pick]
        entry = float(bars.close.iloc[pick])
        a = atr15.get(t0, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        R = float(r.R_atr) * a          # same risk distance, in the same ATR units
        if R <= 0:
            continue
        stop, tgt = entry - R, entry + TGT * R
        s0 = f5.searchsorted(t0 + pd.Timedelta(minutes=15))
        s1 = min(s0 + HOLD5, len(f5))
        out = np.nan
        for i in range(s0, s1):
            if f5l[i] <= stop:
                out = -1.0
                break
            if f5h[i] >= tgt:
                out = TGT
                break
        if np.isnan(out):
            out = (c5[s1 - 1] - entry) / R if s1 > s0 else 0.0
        ctrl_rows.append(dict(sym=sym, day=r.day, y=r.y, r3=out, w3=1 if out > 0 else 0))

C = pd.DataFrame(ctrl_rows)
cx = C.r3.values.astype(float)
clo, chi, _ = clustered_ci(C, 'r3')
print(f"  strategy  n={len(F):>5}  win {F.w3.mean():.0%}  expR {x.mean():+.3f}   CI [{lo_c:+.3f}, {hi_c:+.3f}]")
print(f"  random    n={len(C):>5}  win {C.w3.mean():.0%}  expR {cx.mean():+.3f}   CI [{clo:+.3f}, {chi:+.3f}]")

# paired-by-date difference, clustered
merged = (F.groupby('day').r3.mean().rename('s')
          .to_frame().join(C.groupby('day').r3.mean().rename('c'), how='inner'))
d = (merged.s - merged.c).values
db = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(8000)])
print(f"\n  difference (paired by date, {len(d)} dates): {d.mean():+.3f}R"
      f"   95% CI [{np.percentile(db,2.5):+.3f}, {np.percentile(db,97.5):+.3f}]"
      f"   P(strategy better) = {(db>0).mean():.3f}")

print('\n' + '=' * 96)
print('VERDICT')
print('=' * 96)
checks = [
    ('expectancy > +0.15R',        x.mean() > 0.15,                     f'{x.mean():+.3f}'),
    ('date-clustered CI above 0',  lo_c > 0,                            f'[{lo_c:+.3f}, {hi_c:+.3f}]'),
    ('n >= 100',                   len(F) >= 100,                       f'{len(F)}'),
    ('majority of tickers positive', (pt.expR > 0).sum() > len(pt) / 2, f'{(pt.expR>0).sum()}/{len(pt)}'),
    ('holdout positive',           ho.mean() > 0,                       f'{ho.mean():+.3f}'),
    ('beats matched random entry', np.percentile(db, 2.5) > 0,          f'{d.mean():+.3f}R, P={(db>0).mean():.2f}'),
]
for name, ok, val in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {val}")
print(f"\n  overall: {'PASSES the standard' if all(o for _, o, _ in checks) else 'FAILS the standard'}")
