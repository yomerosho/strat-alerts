"""If you literally set a 1H CHoCH alert and took every confirmation, what happens?"""
import numpy as np, pandas as pd
T = pd.read_csv('research/smc/events.csv')
T['ts'] = pd.to_datetime(T.ts, utc=True).dt.tz_convert('America/New_York')
T['y'] = T.ts.dt.year
B = pd.read_csv('research/smc/baseline.csv')
YEARS = 4.55

def boot(x, n=10000, seed=11):
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()

def dd(x):
    eq = np.cumsum(x); return float((np.maximum.accumulate(eq) - eq).max())

print("="*118)
print("Taking EVERY confirmed 1H CHoCH  (entry at break close, stop at protected swing)")
print("="*118)
print(f"{'what you would alert on':<42}{'n':>5}{'/yr':>6}{'hit1R':>7}{'hit2R':>7}"
      f"{'exp2R':>8}{'95% CI':>18}{'P>0':>6}{'totR':>8}{'maxDD':>8}")

rows = [
 ("every CHoCH, both directions",  T[T.kind=='CHoCH']),
 ("long CHoCH only",               T[(T.kind=='CHoCH')&(T.dir==1)]),
 ("long CHoCH + 1W bullish",       T[(T.kind=='CHoCH')&(T.dir==1)&(T.bw==1)]),
 ("long CHoCH + full stack",       T[(T.kind=='CHoCH')&(T.dir==1)&(T.stacked==1)]),
 ("long CHoCH + stack + swept",    T[(T.kind=='CHoCH')&(T.dir==1)&(T.stacked==1)&(T.swept==1)]),
 ("short CHoCH only",              T[(T.kind=='CHoCH')&(T.dir==-1)]),
 ("[BOS long, for contrast]",      T[(T.kind=='BOS')&(T.dir==1)&(T.stacked==1)]),
]
for lbl, g in rows:
    if len(g) < 10: continue
    x = g.r_2r.values; lo, hi, p = boot(x)
    print(f"{lbl:<42}{len(g):>5}{len(g)/YEARS:>6.0f}{g.w1.mean()*100:6.0f}%"
          f"{g.w2.mean()*100:6.0f}%{x.mean():>8.3f}"
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>18}{p:>6.2f}{x.sum():>8.1f}{dd(x):>8.1f}")
b = B[B.dir==1]
print(f"{'BENCHMARK long any hour, stack bull':<42}{len(b):>5}{'':>6}{b.w1.mean()*100:6.0f}%"
      f"{b.w2.mean()*100:6.0f}%{b.r_2r.mean():>8.3f}{'':>18}{'':>6}")

print("\n" + "="*118)
print("Year by year, stacked-bull long CHoCH  (the best-looking CHoCH cell)")
print("="*118)
g = T[(T.kind=='CHoCH')&(T.dir==1)&(T.stacked==1)]
print(g.groupby('y').agg(n=('r_2r','size'), hit1R=('w1','mean'), hit2R=('w2','mean'),
      exp2R=('r_2r','mean'), totR=('r_2r','sum'), expSO=('r_so','mean')).round(2).to_string())

print("\n" + "="*118)
print("Same cell, is it better than the benchmark? (difference in means, bootstrapped)")
print("="*118)
rng = np.random.default_rng(5)
a, c = g.r_2r.values, b.r_2r.values
d = rng.choice(a,(10000,len(a)),replace=True).mean(1) - rng.choice(c,(10000,len(c)),replace=True).mean(1)
print(f"CHoCH exp2R {a.mean():+.3f} vs benchmark {c.mean():+.3f}   diff {a.mean()-c.mean():+.3f}"
      f"   95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]   P(better) = {(d>0).mean():.2f}")
a1, c1 = g.w1.values.astype(float), b.w1.values.astype(float)
d1 = rng.choice(a1,(10000,len(a1)),replace=True).mean(1) - rng.choice(c1,(10000,len(c1)),replace=True).mean(1)
print(f"hit-1R      {a1.mean():.1%} vs benchmark {c1.mean():.1%}   diff {a1.mean()-c1.mean():+.1%}"
      f"   95% CI [{np.percentile(d1,2.5):+.1%}, {np.percentile(d1,97.5):+.1%}]   P(better) = {(d1>0).mean():.2f}")

print("\n" + "="*118)
print("The best cell -- long CHoCH + 1W bullish (no 1D/4H filter) -- year by year")
print("="*118)
g2 = T[(T.kind=='CHoCH')&(T.dir==1)&(T.bw==1)]
print(g2.groupby('y').agg(n=('r_2r','size'), hit1R=('w1','mean'), hit2R=('w2','mean'),
      exp2R=('r_2r','mean'), totR=('r_2r','sum'), expSO=('r_so','mean')).round(2).to_string())
x = g2.r_2r.values
print(f"\ntotal {x.sum():+.1f}R over 4.55y  |  maxDD {dd(x):.1f}R  |  return/DD = {x.sum()/dd(x):.2f}")
print(f"drop the single best year: ", end="")
for y in sorted(g2.y.unique()):
    z = g2[g2.y != y].r_2r.values
    print(f"ex-{y} {z.mean():+.3f}  ", end="")
print()
