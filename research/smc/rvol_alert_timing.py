"""With the PROFILE-based RVOL (not the biased linear one), how early does a
reading of >=1.5x actually forecast a >=1.5x day? That decides when an alert
should fire -- or whether it should exist."""
import os, numpy as np, pandas as pd
U=['IWM','SPY','QQQ','AAPL','AMZN','GOOGL','META','MSFT','NVDA','PLTR','TSLA','NFLX','INTC','QCOM','ORCL']
CHECK=['09:45','10:00','10:30','11:00','11:30','12:00','13:00','14:00']
rows=[]
for sym in U:
    p=f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(p): continue
    d5=pd.read_parquet(p); d5['d']=d5.index.date
    rth=d5.between_time('09:30','15:59')
    days=sorted(rth.d.unique())
    # cumulative volume matrix: rows=days, cols=bar index within session
    cums={}
    for day,g in rth.groupby('d'):
        cums[day]=g.volume.cumsum().values
    ncol=max(len(v) for v in cums.values())
    prof=np.full(ncol,np.nan)   # EWMA of cumulative volume at each slot
    for day in days:
        c=cums[day]
        if len(c)<70: continue
        prior=prof[:len(c)].copy()
        final_prior=prior[len(c)-1]
        # reading at each check time, using only prior sessions
        if np.isfinite(final_prior) and final_prior>0:
            fin=c[-1]/final_prior
            g=rth[rth.d==day]
            for t in CHECK:
                idx=g.index.indexer_between_time(t,t)
                if len(idx)==0: continue
                i=int(idx[0])
                if i>=len(prior) or not np.isfinite(prior[i]) or prior[i]<=0: continue
                rows.append(dict(sym=sym,day=day,t=t,read=c[i]/prior[i],final=fin))
        # update profile after reading
        m=np.isnan(prof[:len(c)])
        prof[:len(c)]=np.where(m,c,prof[:len(c)]*0.9+c*0.1)
R=pd.DataFrame(rows)
print(f"{len(R):,} observations, {R.day.nunique()} sessions\n")
print("PROFILE-BASED RVOL: is 1.0x actually neutral at every hour?")
print(R.groupby('t').read.agg(['median','mean']).round(3).to_string())
print("\nFORECAST VALUE OF AN EARLY READING")
print(f"{'time':<9}{'corr w/final':>13}{'n read>=1.5':>13}{'P(final>=1.5)':>15}{'P(final>=1.2)':>15}{'base rate':>11}")
base=(R.groupby('day').final.first()>=1.5).mean()
for t in CHECK:
    s=R[R.t==t]; hi=s[s.read>=1.5]
    if len(hi)<30: continue
    print(f"{t:<9}{s.read.corr(s.final):>13.2f}{len(hi):>13}{(hi.final>=1.5).mean():>15.0%}"
          f"{(hi.final>=1.2).mean():>15.0%}{base:>11.0%}")
print("\nHow many alerts would fire, per ticker per year, at each threshold/time?")
for t in ['09:45','10:30','11:30']:
    s=R[R.t==t]
    for thr in (1.5,2.0):
        n=(s.read>=thr).sum(); d=s.day.nunique()
        print(f"  {t} thr {thr}: {n/max(d,1)*252/15:.0f} per ticker per year")
