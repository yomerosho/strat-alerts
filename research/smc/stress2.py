import numpy as np, pandas as pd
exec(open('research/smc/smc_lab.py').read().split('# ---------- build the event table')[0])
HOR={'+1h':1,'+4h':4,'+1d':7,'+3d':20}
rows=[]
for _,e in ev_1h.iterrows():
    i=int(e['i']); d=int(e['dir'])
    if i+20>=n1: continue
    r=dict(ts=e['ts'],dir=d,kind=e['kind'])
    for k,h in HOR.items(): r[k]=(cc[i+h]/cc[i]-1)*1e4
    rows.append(r)
F=pd.DataFrame(rows); F['y']=F.ts.dt.year
print("="*96); print("Bearish 1H BOS -> RAW forward return (bps, unsigned: negative = SPY fell). By year."); print("="*96)
g=F[(F.dir==-1)&(F.kind=='BOS')]
print(g.groupby('y').agg(n=('+1d','size'),**{k:(k,'mean') for k in HOR}).round(1).to_string())
print(f"\nALL: n={len(g)}  " + "  ".join(f"{k}={g[k].mean():+6.1f} (down {g[k].lt(0).mean():.0%})" for k in HOR))
rng=np.random.default_rng(1); x=g['+3d'].values
bs=rng.choice(x,size=(10000,len(x)),replace=True).mean(axis=1)
print(f"+3d bootstrap 95% CI [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}]  P(<0)={(bs<0).mean():.2f}")
print("\nBullish 1H BOS for contrast:")
g2=F[(F.dir==1)&(F.kind=='BOS')]
print(g2.groupby('y').agg(n=('+1d','size'),**{k:(k,'mean') for k in HOR}).round(1).to_string())
print(f"ALL: n={len(g2)}  " + "  ".join(f"{k}={g2[k].mean():+6.1f} (up {g2[k].gt(0).mean():.0%})" for k in HOR))
