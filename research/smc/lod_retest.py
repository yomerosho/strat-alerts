"""
"Price got rejected from the session high -- is it coming back to test the LOD?"

Turn that into a base rate. For every QQQ session 2022-2026, at a given time of
day, measure P(trade at or below the session low before the close), split by how
far above the low price already is and by whether a rejection just happened.
The rejection flag has to beat the plain distance bucket, or it isn't telling
you anything the price ladder wasn't already telling you.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('research/.lab_cache/QQQ_5Min_2022-01-01.parquet')
df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
df = df.set_index('ts').sort_index()
df['d'] = df.index.date

CHECKS = ['11:00', '11:30', '12:00', '12:30', '13:00']
REJ_WINDOW = 9          # HOD printed within the last 45 min
REJ_PULLBACK = 0.20     # and price has given back >=20% of the day's range

# daily structure bias (bearish = lower highs and lower lows on the daily)
daily = df.groupby('d').agg(o=('open', 'first'), h=('high', 'max'),
                            l=('low', 'min'), c=('close', 'last'))
daily['bear'] = ((daily.h < daily.h.shift(1)) & (daily.l < daily.l.shift(1))).shift(1)
daily['pc'] = daily.c.shift(1)

rows = []
for d, g in df.groupby('d'):
    g = g.between_time('09:30', '16:00')
    if len(g) < 60:
        continue
    hi = g.high.values
    lo = g.low.values
    cl = g.close.values
    ts = g.index
    for chk in CHECKS:
        pos = ts.indexer_between_time(chk, chk)
        if len(pos) == 0:
            continue
        t = int(pos[0])
        if t < 6 or t >= len(g) - 6:
            continue
        LOD = lo[:t + 1].min()
        HOD = hi[:t + 1].max()
        rng = HOD - LOD
        if rng <= 0:
            continue
        C = cl[t]
        hod_i = int(np.argmax(hi[:t + 1]))
        rejected = (t - hod_i <= REJ_WINDOW) and ((HOD - C) / rng >= REJ_PULLBACK)
        rest_lo = lo[t + 1:].min()
        rest_hi = hi[t + 1:].max()
        rows.append(dict(d=d, chk=chk, dist=(C - LOD) / LOD * 100,
                         dist_r=(C - LOD) / rng, rejected=rejected,
                         hit_lod=rest_lo <= LOD, hit_hod=rest_hi >= HOD,
                         bear=bool(daily.bear.get(d, False)),
                         above_pc=C > daily.pc.get(d, np.nan)))
T = pd.DataFrame(rows)

BUCKETS = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.2), (1.2, 99)]


def tag(x):
    for a, b in BUCKETS:
        if a <= x < b:
            return f"{a:.1f}-{b:.1f}%" if b < 99 else f">{a:.1f}%"
    return None


T['bucket'] = T.dist.map(tag)

print("=" * 96)
print("QQQ 2022-2026 · P(price trades back to the session low before the close)")
print("=" * 96)
print(f"{'how far above the LOD':<16}{'n':>6}{'retest LOD':>12}{'new HOD':>10}"
      f"{'neither':>9}   {'if just rejected':>17}{'n':>6}")
for a, b in BUCKETS:
    lbl = f"{a:.1f}-{b:.1f}%" if b < 99 else f">{a:.1f}%"
    g = T[T.bucket == lbl]
    if len(g) < 30:
        continue
    r = g[g.rejected]
    neither = (~g.hit_lod & ~g.hit_hod).mean()
    rj = f"{r.hit_lod.mean():.0%}" if len(r) >= 20 else "--"
    print(f"{lbl:<16}{len(g):>6}{g.hit_lod.mean():>11.0%}{g.hit_hod.mean():>10.0%}"
          f"{neither:>9.0%}   {rj:>17}{len(r):>6}")

print("\n" + "=" * 96)
print("Does the rejection add anything over just knowing the distance?")
print("=" * 96)
for lbl in [f"{a:.1f}-{b:.1f}%" if b < 99 else f">{a:.1f}%" for a, b in BUCKETS]:
    g = T[T.bucket == lbl]
    r, nr = g[g.rejected], g[~g.rejected]
    if len(r) < 20 or len(nr) < 20:
        continue
    rng_ = np.random.default_rng(4)
    a_, c_ = r.hit_lod.values.astype(float), nr.hit_lod.values.astype(float)
    dd = (rng_.choice(a_, (8000, len(a_)), replace=True).mean(1)
          - rng_.choice(c_, (8000, len(c_)), replace=True).mean(1))
    print(f"{lbl:<12} rejected {a_.mean():.0%} (n={len(r)})  vs  not rejected "
          f"{c_.mean():.0%} (n={len(nr)})   diff {a_.mean()-c_.mean():+.0%}  "
          f"95% CI [{np.percentile(dd,2.5):+.0%}, {np.percentile(dd,97.5):+.0%}]")

print("\n" + "=" * 96)
print("Today's cell: ~12:20, price 0.65% above the LOD, HOD 30 min ago, gave back ~29%")
print("=" * 96)
now = T[(T.chk.isin(['12:00', '12:30'])) & (T.dist.between(0.45, 0.85))]
print(f"all sessions           n={len(now):>4}   retest LOD {now.hit_lod.mean():.0%}   "
      f"new HOD {now.hit_hod.mean():.0%}   neither {(~now.hit_lod & ~now.hit_hod).mean():.0%}")
nr = now[now.rejected]
print(f"...and just rejected     n={len(nr):>4}   retest LOD {nr.hit_lod.mean():.0%}   "
      f"new HOD {nr.hit_hod.mean():.0%}   neither {(~nr.hit_lod & ~nr.hit_hod).mean():.0%}")
nb = now[now.rejected & now.bear]
if len(nb) >= 15:
    print(f"...+ daily structure bearish  n={len(nb):>4}   retest LOD {nb.hit_lod.mean():.0%}   "
          f"new HOD {nb.hit_hod.mean():.0%}")
na = now[now.rejected & now.above_pc]
if len(na) >= 15:
    print(f"...+ above prior close        n={len(na):>4}   retest LOD {na.hit_lod.mean():.0%}   "
          f"new HOD {na.hit_hod.mean():.0%}")
