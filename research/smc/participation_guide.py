"""What do PM VOL and RVOL actually tell you? Descriptive, not predictive."""
import os, numpy as np, pandas as pd
U = ['IWM','SPY','QQQ','AAPL','AMZN','GOOGL','META','MSFT','NVDA','PLTR','TSLA','NFLX','INTC','QCOM','ORCL']
rows=[]
for sym in U:
    p=f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(p): continue
    d5=pd.read_parquet(p); d5['d']=d5.index.date
    pm=d5.between_time('04:00','09:29').groupby('d').volume.sum()
    rth=d5.between_time('09:30','15:59')
    day=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum'))
    day=day[day.v>0]
    day['pmv']=pm.reindex(day.index).fillna(0)
    day['pm_avg']=day.pmv.rolling(20).mean().shift(1)
    day['v_avg']=day.v.rolling(20).mean().shift(1)
    day['pm_x']=day.pmv/day.pm_avg
    day['rvol']=day.v/day.v_avg
    day['range_pct']=(day.h-day.l)/day.o*100
    day['absret']=(day.c/day.o-1).abs()*100
    day['sym']=sym
    rows.append(day.dropna(subset=['pm_x','rvol']))
T=pd.concat(rows).reset_index()
print(f"{len(T):,} symbol-days, {T.sym.nunique()} tickers\n")
print("HOW OFTEN IS EACH ROW GREEN?")
print(f"  PM VOL >= 1.75x : {(T.pm_x>=1.75).mean():.0%} of days   (about 1 in {1/(T.pm_x>=1.75).mean():.0f})")
print(f"  RVOL   >= 1.5x  : {(T.rvol>=1.5).mean():.0%} of days   (about 1 in {1/(T.rvol>=1.5).mean():.0f})")
print(f"  both green      : {((T.pm_x>=1.75)&(T.rvol>=1.5)).mean():.0%}")
print(f"  PM VOL <  0.7x  : {(T.pm_x<0.7).mean():.0%}   RVOL < 0.7x: {(T.rvol<0.7).mean():.0%}")

print("\nWHAT DOES A GREEN PM VOL ASSOCIATE WITH? (same-day, descriptive)")
b=pd.cut(T.pm_x,[0,0.7,1.0,1.75,1e9],labels=['<0.7 dead','0.7-1.0','1.0-1.75','>=1.75 GREEN'])
print(T.groupby(b,observed=True).agg(n=('range_pct','size'),day_range_pct=('range_pct','mean'),
      abs_move_pct=('absret','mean'),rvol_that_day=('rvol','mean')).round(2).to_string())

print("\nWHAT DOES RVOL ASSOCIATE WITH?")
b2=pd.cut(T.rvol,[0,0.7,1.0,1.5,1e9],labels=['<0.7 quiet','0.7-1.0','1.0-1.5','>=1.5 GREEN'])
print(T.groupby(b2,observed=True).agg(n=('range_pct','size'),day_range_pct=('range_pct','mean'),
      abs_move_pct=('absret','mean')).round(2).to_string())

print("\nTHE FOUR COMBINATIONS (mean day range %, mean |open-to-close| %)")
T['pmg']=T.pm_x>=1.75; T['rvg']=T.rvol>=1.5
for pg in (True,False):
    for rg in (True,False):
        g=T[(T.pmg==pg)&(T.rvg==rg)]
        if len(g)<30: continue
        print(f"  PM {'GREEN' if pg else 'quiet':<5} + RVOL {'GREEN' if rg else 'quiet':<5}  n={len(g):>5}  "
              f"range {g.range_pct.mean():.2f}%  move {g.absret.mean():.2f}%")
print(f"\ncorr(PM VOL, same-day range) = {T.pm_x.corr(T.range_pct):.2f}   corr(RVOL, range) = {T.rvol.corr(T.range_pct):.2f}")
