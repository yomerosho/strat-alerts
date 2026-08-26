"""Two questions about the RVOL row:
   1. Does the linear pro-rata bias the reading by time of day? (volume is U-shaped)
   2. When can you trust an intraday pace reading as a forecast of the final day?"""
import os, numpy as np, pandas as pd
U=['IWM','SPY','QQQ','AAPL','AMZN','GOOGL','META','MSFT','NVDA','PLTR','TSLA','NFLX','INTC','QCOM','ORCL']
CHECK=['10:00','10:30','11:00','12:00','13:00','14:00','15:00']
rows=[]
for sym in U:
    p=f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(p): continue
    d5=pd.read_parquet(p); d5['d']=d5.index.date
    rth=d5.between_time('09:30','15:59')
    dayv=rth.groupby('d').volume.sum()
    avg=dayv.rolling(20).mean().shift(1)
    for day,g in rth.groupby('d'):
        a=avg.get(day,np.nan)
        if not np.isfinite(a) or a<=0 or len(g)<70: continue
        final=g.volume.sum()/a
        cum=g.volume.cumsum()
        for c in CHECK:
            idx=g.index.indexer_between_time(c,c)
            if len(idx)==0: continue
            i=int(idx[0])
            mins=(pd.Timestamp(g.index[i]).hour*60+pd.Timestamp(g.index[i]).minute)-570+5
            frac=max(min(mins/390.0,1.0),0.02)
            rows.append(dict(sym=sym,day=day,t=c,pace=cum.iloc[i]/(a*frac),final=final,
                             share=cum.iloc[i]/g.volume.sum()))
R=pd.DataFrame(rows)
print(f"{len(R):,} observations across {R.day.nunique()} sessions\n")
print("1. IS THE LINEAR PRO-RATA BIASED BY TIME OF DAY?")
print("   (if unbiased, mean pace should be ~1.0 at every check and share should track elapsed)")
g=R.groupby('t').agg(mean_pace=('pace','mean'),median_pace=('pace','median'),
                     actual_share=('share','mean')).round(3)
g['linear_assumes']=[ (int(t[:2])*60+int(t[3:])-570+5)/390 for t in g.index]
g['linear_assumes']=g['linear_assumes'].round(3)
g['bias']=(g.median_pace-1).round(3)
print(g.to_string())
print("\n2. HOW WELL DOES AN INTRADAY PACE READING FORECAST THE FINAL DAY?")
print(f"{'time':<8}{'corr w/ final':>14}{'P(final>=1.5 | pace>=1.5)':>28}{'n pace>=1.5':>13}")
for t in CHECK:
    s=R[R.t==t]
    hi=s[s.pace>=1.5]
    print(f"{t:<8}{s.pace.corr(s.final):>14.2f}{(hi.final>=1.5).mean() if len(hi)>20 else float('nan'):>28.0%}{len(hi):>13}")
