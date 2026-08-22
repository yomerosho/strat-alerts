import numpy as np, pandas as pd
Q = pd.read_csv('research/smc/qualified.csv'); Q['ts']=pd.to_datetime(Q.ts,utc=True).dt.tz_convert('America/New_York')
F = pd.read_csv('research/smc/events.csv'); F['ts']=pd.to_datetime(F.ts,utc=True).dt.tz_convert('America/New_York')

print("="*92); print("1  EQL (equal-lows) sweep longs -- is it stable, or did one year make it?"); print("="*92)
for style in ('break','retrace50'):
    g = Q[(Q.entry_style==style)&(Q.dir==1)&(Q.filled==1)&(Q.tag.fillna('').str.contains('EQL'))]
    yr = g.assign(y=g.ts.dt.year).groupby('y').agg(n=('r_so','size'),hit1R=('w1','mean'),
                  hit2R=('w2','mean'),exp2R=('r_2r','mean'),expSO=('r_so','mean')).round(3)
    print(f"\n-- entry={style}  total n={len(g)}  hit1R={g.w1.mean():.0%}  exp2R={g.r_2r.mean():+.3f}  expSO={g.r_so.mean():+.3f}")
    print(yr.to_string())

print("\n"+"="*92); print("2  EQL-only (no PDL/PWL confluence) vs EQL+PDL, break entry, longs"); print("="*92)
g = Q[(Q.entry_style=='break')&(Q.dir==1)&(Q.filled==1)]
for lbl,sel in (("EQL alone", g.tag=='EQL'), ("EQL+PDL", g.tag=='EQL+PDL'),
                ("any EQL", g.tag.fillna('').str.contains('EQL')), ("no sweep", g.named==0)):
    s = g[sel]
    if len(s)>5: print(f"{lbl:<12} n={len(s):>3}  hit1R={s.w1.mean():.0%}  hit2R={s.w2.mean():.0%}  exp2R={s.r_2r.mean():+.3f}  expSO={s.r_so.mean():+.3f}")

print("\n"+"="*92); print("3  Bootstrap: is EQL-long exp2R distinguishable from zero? (10k resamples)"); print("="*92)
rng = np.random.default_rng(7)
for lbl,sel in (("EQL any (break)", (Q.entry_style=='break')&(Q.dir==1)&(Q.filled==1)&(Q.tag.fillna('').str.contains('EQL'))),
                ("EQL alone (break)", (Q.entry_style=='break')&(Q.dir==1)&(Q.filled==1)&(Q.tag=='EQL'))):
    x = Q[sel].r_2r.values
    if len(x)<10: continue
    bs = rng.choice(x, size=(10000,len(x)), replace=True).mean(axis=1)
    print(f"{lbl:<20} n={len(x):>3}  mean={x.mean():+.3f}  95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  P(>0)={ (bs>0).mean():.2f}")

print("\n"+"="*92); print("4  The strongest raw signal: bearish 1H BOS -> forward returns, by year"); print("="*92)
