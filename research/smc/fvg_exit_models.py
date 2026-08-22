"""
Same validated FVG entries, five different exit models.

Entry side is frozen: liquidity sweep -> displacement -> limit at 50% of the
fair value gap, LONG only, daily close above its 200SMA, deduplicated to the
first signal per ticker-day. That is the set that scored +0.162R at a fixed 3R
target and beat a matched random-entry control.

Only the EXIT changes, so any difference is the exit rule alone:

  A  fixed 3R                      the benchmark
  B  flat after 10 minutes         "in and out in 10 mins"
  C  half at 1R + BE, runner 3R    scale-out, fixed runner target
  D  half at 1R + BE, runner       runner trails the 5-min structure  <- their rule
  E  same as D but the 1R must     tests whether the speed matters
     arrive within 10 minutes

Structure trail: after the runner is live, the stop ratchets up to the most
recent CONFIRMED 5-minute swing low (fractal, confirmed 2 bars later, so no
lookahead). It never moves down.
"""
import os
import numpy as np
import pandas as pd

LVL = 'fvg_mid'
MAXWAIT_DISP, MAXWAIT_FILL, L = 6, 16, 2
HOLD5 = 78 * 3
SLIP = 0.0002
UNIVERSE = ['IWM', 'SPY', 'QQQ', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
            'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']

# ---- daily trend map, causal
dly = pd.read_parquet('research/smc/data/daily_2015.parquet')
dly['date'] = pd.to_datetime(dly['timestamp']).dt.date
dly = dly.sort_values(['symbol', 'timestamp'])
dly['sma200'] = dly.groupby('symbol').close.transform(lambda s: s.rolling(200).mean())
dly['up'] = (dly.close > dly.sma200).groupby(dly.symbol).shift(1)
TREND = {(r.symbol, r.date): bool(r.up) if r.up == r.up else False for r in dly.itertuples()}


def pivots_low(l, n):
    out = np.full(len(l), np.nan)
    for i in range(n, len(l) - n):
        if l[i] == l[i - n:i + n + 1].min() and (l[i] < l[i - n:i]).all():
            out[i] = l[i]
    return out


rows = []
for sym in UNIVERSE:
    path = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(path):
        continue
    d5 = pd.read_parquet(path).between_time('09:30', '15:59')
    m15 = d5.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                    'close': 'last', 'volume': 'sum'}).dropna()
    m15 = m15[m15.volume > 0]
    o, h, l, c = m15.open.values, m15.high.values, m15.low.values, m15.close.values
    n = len(m15)
    idx15 = m15.index
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(28).mean().shift(1).values

    f5 = d5.index
    f5h, f5l, f5c = d5.high.values, d5.low.values, d5.close.values
    piv5 = pivots_low(f5l, L)

    known = []
    seen_days = set()
    for i in range(n):
        j = i - L
        if j >= 0:
            pl = l[j] if (j >= L and l[j] == l[j - L:j + L + 1].min() and (l[j] < l[j - L:j]).all()) else np.nan
            if pl == pl:
                known.append((j, pl))
        if np.isnan(atr[i]) or atr[i] <= 0 or i + MAXWAIT_DISP + MAXWAIT_FILL >= n:
            continue
        recent = [p for p in known[-10:] if i - p[0] <= 60]
        if not recent or not any(l[i] < lv and c[i] > lv for _, lv in recent):
            continue
        sweep_lo = l[i]

        disp = None
        for k in range(i + 1, min(i + 1 + MAXWAIT_DISP, n)):
            if c[k] <= h[i]:
                continue
            for g in range(max(i + 2, 2), k + 1):
                if l[g] > h[g - 2]:
                    disp = (k, h[g - 2], l[g])
                    break
            if disp:
                break
        if not disp:
            continue
        k, z_far, z_near = disp
        if z_near <= z_far:
            continue
        day = idx15[k].date()
        if not TREND.get((sym, day), False):
            continue
        if (sym, day) in seen_days:
            continue

        zprice = (z_far + z_near) / 2.0
        stop0 = sweep_lo - 0.05 * atr[i]
        if zprice <= stop0 or (zprice - stop0) > 3 * atr[i]:
            continue
        fill = None
        for w in range(k + 1, min(k + 1 + MAXWAIT_FILL, n)):
            if l[w] <= zprice:
                fill = w
                break
            if c[w] < stop0:
                break
        if fill is None:
            continue
        seen_days.add((sym, day))

        entry = zprice * (1 + SLIP)
        R = entry - stop0
        if R <= 0:
            continue
        s0 = f5.searchsorted(idx15[fill] + pd.Timedelta(minutes=15))
        s1 = min(s0 + HOLD5, len(f5))
        if s0 >= s1:
            continue

        def walk(mode):
            """returns total R for the position"""
            stop = stop0
            half_done = False
            trail = None
            for x in range(s0, s1):
                mins = (x - s0 + 1) * 5
                # stop / trail check first (conservative)
                if f5l[x] <= stop:
                    return -1.0 if not half_done else 0.5 * 1.0 + 0.5 * ((stop - entry) / R)
                if mode == 'B' and mins >= 10:
                    return (f5c[x] - entry) / R
                if not half_done and f5h[x] >= entry + R:
                    if mode == 'E' and mins > 10:
                        return (f5c[x] - entry) / R      # too slow: flat it
                    if mode in ('C', 'D', 'E'):
                        half_done = True
                        stop = entry                      # break-even on the runner
                    elif mode == 'A':
                        pass
                if mode in ('C',) and half_done and f5h[x] >= entry + 3 * R:
                    return 0.5 * 1.0 + 0.5 * 3.0
                if mode == 'A' and f5h[x] >= entry + 3 * R:
                    return 3.0
                if mode in ('D', 'E') and half_done:
                    p = piv5[x - L] if x - L >= 0 else np.nan
                    if p == p and p > stop:
                        stop = p                          # ratchet up, never down
            last = f5c[s1 - 1]
            if half_done:
                return 0.5 * 1.0 + 0.5 * ((last - entry) / R)
            return (last - entry) / R

        rows.append(dict(sym=sym, day=day, y=idx15[k].year, Rpct=R / entry * 100,
                         A=walk('A'), B=walk('B'), C=walk('C'), D=walk('D'), E=walk('E')))

T = pd.DataFrame(rows)
T.to_csv('research/smc/fvg_exits.csv', index=False)

rng = np.random.default_rng(5150)
def cci(df, col, iters=6000):
    groups = [g[col].values.astype(float) for _, g in df.groupby('day')]
    k = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

NAMES = {'A': 'fixed 3R (benchmark)',
         'B': 'flat after 10 minutes',
         'C': 'half @1R + BE, runner 3R',
         'D': 'half @1R + BE, runner trails 5m structure',
         'E': 'same as D, but 1R must hit within 10 min'}

print('=' * 104)
print(f'Same {len(T)} validated FVG entries, five exit models')
print('=' * 104)
print(f"{'exit model':<46}{'win':>7}{'expR':>9}{'  95% CI':<24}{'medR':>8}{'best':>8}{'worst':>8}")
for k in ['A', 'B', 'C', 'D', 'E']:
    x = T[k].values.astype(float)
    lo, hi = cci(T, k)
    print(f"{NAMES[k]:<46}{(x>0).mean():>7.0%}{x.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]      "
          f"{np.median(x):>+8.2f}{x.max():>8.1f}{x.min():>8.1f}")

print('\npaired difference vs the fixed-3R benchmark (clustered by date):')
for k in ['B', 'C', 'D', 'E']:
    d = T.groupby('day').apply(lambda g: (g[k] - g.A).mean(), include_groups=False).values
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(6000)])
    print(f"  {NAMES[k]:<46}{d.mean():>+8.3f}R  95% CI [{np.percentile(bs,2.5):+.3f}, "
          f"{np.percentile(bs,97.5):+.3f}]  P(better) = {(bs>0).mean():.2f}")

print('\nby year (their model, D):')
print(T.groupby('y').agg(n=('D', 'size'), win=('D', lambda x: (x > 0).mean()),
                         expR=('D', 'mean')).round(3).to_string())
