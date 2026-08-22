import pandas as pd, numpy as np
T = pd.read_csv('research/smc/breakouts.csv')

TODAY = dict(gap=1.41, pm_vol_x=0.77, pm_range=(350.93-347.02)/11.00, pm_above=0,
             open_above=0, atr_contract=1.12, base_tight=8.8, vs_sma20=6.0,
             vs_sma50=-5.7, dist_to_lvl=(351.62/345.28-1)*100)

print("="*100); print("TSLA 21 Aug 2026 scored against the 1,120-event distribution"); print("="*100)
print(f"{'feature':<16}{'today':>9}{'pctile':>9}{'tercile':>10}{'win rate of that tercile':>28}")
for k, v in TODAY.items():
    if k in ('pm_above','open_above'):
        g = T[T[k]==v]; print(f"{k:<16}{v:>9}{'--':>9}{'flag':>10}{g.win.mean():>27.0%}"); continue
    s = T[k].dropna()
    pct = (s < v).mean()
    q1, q2 = s.quantile([1/3, 2/3])
    ter = 'low' if v < q1 else ('mid' if v < q2 else 'high')
    g = T[T[k].notna()].assign(b=pd.qcut(T[k].dropna(),3,labels=['low','mid','high'],duplicates='drop'))
    w = g[g.b==ter].win.mean()
    print(f"{k:<16}{v:>9.2f}{pct:>9.0%}{ter:>10}{w:>27.0%}")

print("\n"+"="*100); print("Does stacking the features that matter give a usable gate?"); print("="*100)
T['pmv_hi'] = T.pm_vol_x > T.pm_vol_x.quantile(2/3)
T['gap_hi'] = T.gap > T.gap.quantile(2/3)
T['or_hi']  = T.or15_vol_x > T.or15_vol_x.quantile(2/3)
combos = [
 ("no filter (every breakout)",            T.win.notna()),
 ("PRE: gap high",                          T.gap_hi),
 ("PRE: premkt volume high",                T.pmv_hi),
 ("PRE: gap high AND premkt vol high",      T.gap_hi & T.pmv_hi),
 ("PRE: premkt already above the level",    T.pm_above==1),
 ("PRE: all three pre-market flags",        T.gap_hi & T.pmv_hi & (T.pm_above==1)),
 ("OPEN: opens above the level",            T.open_above==1),
 ("OPEN: first-15m volume high",            T.or_hi),
 ("OPEN: opens above + 15m vol high",       (T.open_above==1) & T.or_hi),
 ("OPEN: opens above + 15m vol + closes strong", (T.open_above==1) & T.or_hi & (T.or15_pos>0.6)),
 ("BEST PRE + BEST OPEN",                   T.gap_hi & T.pmv_hi & (T.open_above==1) & T.or_hi),
]
print(f"{'gate':<46}{'n':>6}{'/yr/name':>10}{'win':>7}{'medMFE':>9}{'medMAE':>9}")
for lbl, m in combos:
    g = T[m]
    if len(g) < 15: continue
    print(f"{lbl:<46}{len(g):>6}{len(g)/4.6/5:>10.0f}{g.win.mean():>7.0%}"
          f"{g.mfe.median():>9.2f}{g.mae.median():>9.2f}")

print("\n"+"="*100); print("Bootstrap: is the open-based gate really better than no filter?"); print("="*100)
rng = np.random.default_rng(9)
base = T.win.values.astype(float)
for lbl, m in [("opens above + 15m vol high", (T.open_above==1) & T.or_hi),
               ("all three pre-market flags", T.gap_hi & T.pmv_hi & (T.pm_above==1))]:
    a = T[m].win.values.astype(float)
    d = rng.choice(a,(10000,len(a)),replace=True).mean(1) - rng.choice(base,(10000,len(base)),replace=True).mean(1)
    print(f"{lbl:<34} {a.mean():.1%} vs {base.mean():.1%}  diff {a.mean()-base.mean():+.1%}"
          f"  95% CI [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]  P(better)={(d>0).mean():.2f}")

print("\n"+"="*100); print("Per-name check on the open-based gate (is it one stock?)"); print("="*100)
g = T[(T.open_above==1) & T.or_hi]
print(g.groupby('sym').agg(n=('win','size'), win=('win','mean'), mfe=('mfe','median')).round(3).to_string())
print("\nby year:")
g2 = g.assign(y=pd.to_datetime(g.day).dt.year)
print(g2.groupby('y').agg(n=('win','size'), win=('win','mean')).round(3).to_string())
