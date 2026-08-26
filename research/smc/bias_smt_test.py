"""
The BIA$ Model (DayloTrading) -- testing the one part that has never been tested.

The blueprint is four steps:
  1  Daily bias + Draw on Liquidity      -> already tested
  2  Opposite-side liquidity sweep       -> already tested
  3  SMT divergence between correlated   -> NEVER TESTED  <-- this script
  4  Entry on inverted FVG, stop past    -> already tested
     the manipulation low, target the
     opposing liquidity pool

Steps 1, 2 and 4 plus the exit rule are exactly `research/smc/ict_external.py`,
which returned expR -0.180, PF 0.46, n=746, negative on every ticker, every
year and both sides. The blueprint's exit ("target the opposing liquidity
pool") is the specific thing that broke it: TP2 averaged 6.3R away and only
3.8% of trades ever reached it.

So the only open question is whether Step 3 rescues the other three.

SMT here is the retail equivalent of the blueprint's ES vs NQ: SPY vs QQQ,
plus IWM as a third leg. At the moment of the sweep, does the correlated
partner FAIL to make the same extreme?
    bullish SMT : this symbol makes a lower low, the partner does not
    bearish SMT : this symbol makes a higher high, the partner does not
Measured on 5-minute bars over a lookback window ending at the signal bar, so
it is knowable at entry.
"""
import numpy as np
import pandas as pd

LOOKBACK = 12          # 5-min bars used to define "the recent extreme" (1 hour)
PARTNER = {'SPY': ['QQQ', 'IWM'], 'QQQ': ['SPY', 'IWM'], 'IWM': ['SPY', 'QQQ']}

T = pd.read_csv('research/smc/ict_ext_all.csv')
T['ts'] = pd.to_datetime(T.ts, utc=True).dt.tz_convert('America/New_York')
T['day'] = pd.to_datetime(T.day).dt.date
print(f'{len(T)} ICT trades | expR {T.net_r.mean():+.3f} | tickers {sorted(T.sym.unique())}')

BARS = {}
for s in ['SPY', 'QQQ', 'IWM']:
    d = pd.read_parquet(f'research/smc/data/{s}_5m_ext.parquet').between_time('09:30', '15:59')
    BARS[s] = d

rows = []
for _, r in T.iterrows():
    sym, ts, side = r.sym, r.ts, r.side
    me = BARS[sym]
    i = me.index.searchsorted(ts)
    if i < LOOKBACK or i >= len(me):
        rows.append(dict(smt=np.nan, smt_n=0))
        continue
    w = slice(i - LOOKBACK, i + 1)
    # did I make the extreme, and did the partner fail to?
    if side == 'L':
        my_ext = me.low.values[w].min()
        my_is_ext = me.low.values[i] <= my_ext + 1e-9
    else:
        my_ext = me.high.values[w].max()
        my_is_ext = me.high.values[i] >= my_ext - 1e-9

    divs = 0
    checked = 0
    for p in PARTNER[sym]:
        pb = BARS[p]
        j = pb.index.searchsorted(ts)
        if j < LOOKBACK or j >= len(pb):
            continue
        pw = slice(j - LOOKBACK, j + 1)
        if side == 'L':
            p_ext = pb.low.values[pw].min()
            p_is_ext = pb.low.values[j] <= p_ext + 1e-9
        else:
            p_ext = pb.high.values[pw].max()
            p_is_ext = pb.high.values[j] >= p_ext - 1e-9
        checked += 1
        # divergence = I made the extreme, partner did NOT
        if my_is_ext and not p_is_ext:
            divs += 1
    rows.append(dict(smt=(divs > 0) if checked else np.nan, smt_n=divs))

S = pd.DataFrame(rows)
T = pd.concat([T.reset_index(drop=True), S], axis=1)
T.to_csv('research/smc/ict_smt.csv', index=False)

rng = np.random.default_rng(1234)
def cci(df, col='net_r', iters=8000):
    g = [x[col].values.astype(float) for _, x in df.groupby('day')]
    k = len(g)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([g[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

V = T.dropna(subset=['smt'])
print(f'\nSMT computable on {len(V)} of {len(T)} trades')
print('=' * 96)
print('DOES SMT DIVERGENCE RESCUE THE SETUP?')
print('=' * 96)
print(f"{'bucket':<34}{'n':>6}{'win':>7}{'expR':>9}{'  95% CI':<22}{'PF':>6}")
def line(lbl, g):
    if len(g) < 20:
        print(f"{lbl:<34}{len(g):>6}   too few")
        return
    x = g.net_r.values.astype(float)
    w, l = x[x > 0], x[x <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    lo, hi = cci(g)
    print(f"{lbl:<34}{len(g):>6}{(x>0).mean():>7.0%}{x.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{pf:>6.2f}")

line('ALL trades (the ICT baseline)', V)
line('  WITH SMT divergence', V[V.smt == True])
line('  without SMT', V[V.smt == False])
line('  SMT on BOTH partners', V[V.smt_n >= 2])
print()
line('  WITH SMT, longs', V[(V.smt == True) & (V.side == 'L')])
line('  WITH SMT, shorts', V[(V.smt == True) & (V.side == 'S')])

a = V[V.smt == True].net_r.values.astype(float)
b = V[V.smt == False].net_r.values.astype(float)
if len(a) > 20 and len(b) > 20:
    d = (rng.choice(a, (8000, len(a))).mean(1) - rng.choice(b, (8000, len(b))).mean(1))
    print(f"\nSMT minus no-SMT: {a.mean()-b.mean():+.3f}R  95% CI [{np.percentile(d,2.5):+.3f}, "
          f"{np.percentile(d,97.5):+.3f}]  P(SMT better) = {(d>0).mean():.2f}")
    print(f"Does SMT clear zero on its own? mean {a.mean():+.3f}, "
          f"{'YES' if cci(V[V.smt == True])[0] > 0 else 'NO'}")

print('\nby ticker (WITH SMT):')
g = V[V.smt == True]
print(g.groupby('sym').agg(n=('net_r', 'size'), win=('net_r', lambda x: (x > 0).mean()),
                           expR=('net_r', 'mean')).round(3).to_string())
print('\nby year (WITH SMT):')
g2 = g.assign(y=pd.to_datetime(g.day).dt.year)
print(g2.groupby('y').agg(n=('net_r', 'size'), expR=('net_r', 'mean')).round(3).to_string())
print(f"\nSMT fires on {V.smt.mean():.0%} of setups")
