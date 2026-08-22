"""The synthesis: treat the 1H break itself as the liquidity event and FADE it,
filtered by HTF bias. Stop beyond the break bar's extreme."""
import numpy as np, pandas as pd
exec(open('research/smc/smc_lab.py').read().split('# ---------- build the event table')[0])

rows=[]
for _,e in ev_1h.iterrows():
    i=int(e['i']); d=int(e['dir'])
    if i+1>=n1: continue
    ts=e['ts']; fd=-d                     # fade direction
    b4,bd,bw = bias_at(ts,h4_close,tr_4h), bias_at(ts,d1_close,tr_1d), bias_at(ts,w1_close,tr_1w)
    # stop just beyond the extreme of the break bar (the sweep candle itself)
    if fd==1: stop=float(ll[i])-BUF; entry=cc[i]+SLIP
    else:     stop=float(hh[i])+BUF; entry=cc[i]-SLIP
    s=simulate(h1.index[i]+pd.Timedelta(minutes=60), entry, stop, fd)
    if not s: continue
    rows.append(dict(ts=ts,broke=d,fade=fd,kind=e['kind'],b4=b4,bd=bd,bw=bw,
        aligned=int(b4==fd and bd==fd and bw==fd), w_ok=int(bw==fd), d_ok=int(bd==fd),
        Rpct=s['R']/entry*100,mfe=s['mfe'],w1=s['w1'],w2=s['w2'],w3=s['w3'],
        stopped=s['stopped'],r_1r=s['r_1r'],r_2r=s['r_2r'],r_so=s['r_so']))
D=pd.DataFrame(rows); D.to_csv('research/smc/fade.csv',index=False)
D['y']=D.ts.dt.year

def line(g,lbl,minn=15):
    if len(g)<minn: return
    print(f"{lbl:<52}{len(g):>5}{g.w1.mean()*100:6.0f}%{g.w2.mean()*100:6.0f}%"
          f"{g.stopped.mean()*100:6.0f}%{g.r_1r.mean():>8.3f}{g.r_2r.mean():>8.3f}"
          f"{g.r_so.mean():>8.3f}{g.Rpct.median():>7.2f}")

print("="*104)
print("FADE the 1H break. Stop beyond the break bar. Entry at its close.")
print("="*104)
print(f"{'setup':<52}{'n':>5}{'hit1R':>7}{'hit2R':>7}{'stop%':>7}{'exp1R':>8}{'exp2R':>8}{'expSO':>8}{'R%px':>7}")
L=D[D.fade==1]; S=D[D.fade==-1]
line(L,"LONG the bearish break -- any HTF")
line(L[L.w_ok==1],"LONG the bearish break -- 1W bullish")
line(L[(L.w_ok==1)&(L.d_ok==1)],"LONG the bearish break -- 1W+1D bullish")
line(L[L.aligned==1],"LONG the bearish break -- 4H+1D+1W bullish")
line(L[(L.w_ok==1)&(L.kind=='BOS')],"LONG a bearish BOS   -- 1W bullish")
line(L[(L.w_ok==1)&(L.kind=='CHoCH')],"LONG a bearish CHoCH -- 1W bullish")
print()
line(S,"SHORT the bullish break -- any HTF")
line(S[S.w_ok==1],"SHORT the bullish break -- 1W bearish")
line(S[(S.w_ok==1)&(S.kind=='BOS')],"SHORT a bullish BOS  -- 1W bearish")

best=L[(L.w_ok==1)&(L.kind=='BOS')]
print("\n"+"="*104); print("Year by year: LONG a bearish 1H BOS while the weekly is bullish"); print("="*104)
print(best.groupby('y').agg(n=('r_so','size'),hit1R=('w1','mean'),hit2R=('w2','mean'),
      exp2R=('r_2r','mean'),expSO=('r_so','mean')).round(3).to_string())
rng=np.random.default_rng(3)
for col in ('r_1r','r_2r','r_so'):
    x=best[col].values; bs=rng.choice(x,size=(10000,len(x)),replace=True).mean(axis=1)
    print(f"{col}: mean={x.mean():+.3f}  95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  P(>0)={(bs>0).mean():.2f}")
print(f"\nMFE p25/50/75/90: {best.mfe.quantile([.25,.5,.75,.9]).round(2).tolist()}   median stop = {best.Rpct.median():.2f}% of price")
