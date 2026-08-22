"""
Order flow, judged as a trade rather than as a correlation.

The top-vs-bottom "spread" from the first pass is a comparison, not a position.
Here each signal becomes an actual trade: buy when order-flow imbalance is in
the top quintile, hold a fixed horizon, pay the round trip. Net is what matters.

Costs: SPY 1.5bp round trip, single names 4bp. These are the same figures used
in the scalping test, so results are comparable.

Bootstrap is clustered by DAY throughout -- minutes inside a session are far
too correlated for per-observation resampling.
"""
import glob
import numpy as np
import pandas as pd

COST = {'SPY': 1.5, 'TSLA': 4.0, 'NVDA': 4.0}
rng = np.random.default_rng(4242)

files = sorted(glob.glob('research/smc/data/flow/*.parquet'))
F = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
F['m'] = pd.to_datetime(F.m, utc=True).dt.tz_convert('America/New_York')
F = F.sort_values(['sym', 'm']).reset_index(drop=True)
cover = F.groupby('sym').day.nunique()
print('sessions per symbol:', cover.to_dict())
FULL = [s for s in cover.index if cover[s] >= 40]
print('symbols with a full sample:', FULL)

rows = []
for (sym, day), g in F.groupby(['sym', 'day'], sort=False):
    g = g.sort_values('m').reset_index(drop=True)
    if len(g) < 120:
        continue
    buy, sell = g.buy.values.astype(float), g.sell.values.astype(float)
    bb, bs = g.big_buy.values.astype(float), g.big_sell.values.astype(float)
    px = g.px.values.astype(float)
    n = len(g)

    def ofi(b, s, k):
        cb = pd.Series(b).rolling(k).sum().values
        cs = pd.Series(s).rolling(k).sum().values
        t = cb + cs
        return np.where(t > 0, (cb - cs) / np.maximum(t, 1), np.nan)

    o15, b15 = ofi(buy, sell, 15), ofi(bb, bs, 15)
    for h in (15, 30):
        fwd = np.full(n, np.nan)
        fwd[:-h] = px[h:] / px[:-h] - 1
        for i in range(20, n - 35):
            if not np.isfinite(fwd[i]):
                continue
            rows.append(dict(sym=sym, day=day, h=h, hour=g.m[i].hour,
                             ofi=o15[i], bofi=b15[i], fwd_bp=fwd[i] * 1e4))
T = pd.DataFrame(rows).dropna(subset=['ofi'])
T['day'] = pd.to_datetime(T.day).dt.date
print(f'observations: {len(T):,}')


def cci(x_by_day, iters=5000):
    groups = list(x_by_day)
    k = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


print('\n' + '=' * 100)
print('TRADE THE SIGNAL: buy the top-quintile flow, hold the horizon, pay the round trip')
print('=' * 100)
print(f"{'symbol':<8}{'feat':<7}{'h':>4}{'n':>8}{'gross bp':>10}{'cost':>7}{'net bp':>9}"
      f"{'  net 95% CI':<22}{'P>0':>6}")
for sym in FULL:
    for feat in ('ofi', 'bofi'):
        for h in (15, 30):
            s = T[(T.sym == sym) & (T.h == h)].dropna(subset=[feat])
            if len(s) < 500:
                continue
            thr = s[feat].quantile(0.8)
            g = s[s[feat] >= thr].copy()
            c = COST[sym]
            g['net'] = g.fwd_bp - c
            lo, hi, p = cci([x.net.values for _, x in g.groupby('day')])
            print(f"{sym:<8}{feat:<7}{h:>4}{len(g):>8}{g.fwd_bp.mean():>10.2f}{c:>7.1f}"
                  f"{g.net.mean():>9.2f}   [{lo:+.2f}, {hi:+.2f}]{p:>7.2f}")

print('\n' + '=' * 100)
print('Is it stable? first half vs second half of the sample (net bp, 15-min horizon)')
print('=' * 100)
for sym in FULL:
    s = T[(T.sym == sym) & (T.h == 15)].dropna(subset=['bofi'])
    thr = s.bofi.quantile(0.8)
    g = s[s.bofi >= thr].copy()
    g['net'] = g.fwd_bp - COST[sym]
    days = sorted(g.day.unique())
    mid = days[len(days) // 2]
    a = g[g.day < mid].net
    b = g[g.day >= mid].net
    print(f"  {sym:<6} block-flow  first half {a.mean():+6.2f} bp (n={len(a):>5})   "
          f"second half {b.mean():+6.2f} bp (n={len(b):>5})")

print('\n' + '=' * 100)
print('By hour of day (block flow, 15-min horizon, net bp)')
print('=' * 100)
for sym in FULL:
    s = T[(T.sym == sym) & (T.h == 15)].dropna(subset=['bofi'])
    thr = s.bofi.quantile(0.8)
    g = s[s.bofi >= thr].copy()
    g['net'] = g.fwd_bp - COST[sym]
    hh = g.groupby('hour').net.agg(['size', 'mean']).round(2)
    print(f"  {sym}: " + '  '.join(f"{int(i)}h {r['mean']:+.1f}({int(r['size'])})" for i, r in hh.iterrows()))
