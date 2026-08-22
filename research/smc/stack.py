import pandas as pd, numpy as np
T = pd.read_csv('research/smc/gates.csv')
BASE = T.win.mean(); YRS = 4.6; N = T.sym.nunique()
T['g_gap']  = T.gap > T.gap.quantile(2/3)
T['g_pmv']  = T.pm_vol_x > 1.75
T['g_open'] = T.open_above == 1
T['g_or']   = T.or15_vol_x > 3.6
T['g_pos']  = T.or15_pos > 0.6
T['g_pma']  = T.pm_above == 1
T['score']  = T[['g_gap','g_pmv','g_open','g_or']].sum(axis=1)

print("="*100); print("Hard AND: every gate added costs you signals"); print("="*100)
print(f"{'stack':<48}{'n':>6}{'sig/name/yr':>13}{'win':>7}{'medMFE':>9}")
for lbl, m in [
  ("no gate",                                   pd.Series(True, index=T.index)),
  ("gap",                                       T.g_gap),
  ("gap + premkt vol",                          T.g_gap & T.g_pmv),
  ("gap + premkt vol + opens above",            T.g_gap & T.g_pmv & T.g_open),
  ("gap + premkt vol + opens above + 15m vol",  T.g_gap & T.g_pmv & T.g_open & T.g_or),
  ("...+ 15m closes strong",                    T.g_gap & T.g_pmv & T.g_open & T.g_or & T.g_pos),
  ("opens above + 15m vol  (open-only)",        T.g_open & T.g_or),
  ("opens above + 15m vol + closes strong",     T.g_open & T.g_or & T.g_pos),
]:
    g = T[m]
    print(f"{lbl:<48}{len(g):>6}{len(g)/YRS/N:>13.1f}{g.win.mean():>7.0%}{g.mfe.median() if 'mfe' in g else np.nan:>9}"
          if 'mfe' in T.columns else
          f"{lbl:<48}{len(g):>6}{len(g)/YRS/N:>13.1f}{g.win.mean():>7.0%}")

print("\n"+"="*100); print("Score instead of AND: how many of the 4 gates fired"); print("="*100)
print(f"{'score':<48}{'n':>6}{'sig/name/yr':>13}{'win':>7}")
for s in range(5):
    g = T[T.score==s]
    if len(g)<20: continue
    print(f"{'  '+str(s)+' of 4':<48}{len(g):>6}{len(g)/YRS/N:>13.1f}{g.win.mean():>7.0%}")
for s in (2,3):
    g = T[T.score>=s]
    print(f"{'  >= '+str(s)+' of 4':<48}{len(g):>6}{len(g)/YRS/N:>13.1f}{g.win.mean():>7.0%}")

rng = np.random.default_rng(77); base=T.win.values.astype(float)
print("\n"+"="*100); print("Bootstrap vs base rate"); print("="*100)
for lbl, m in [(">=2 of 4", T.score>=2), (">=3 of 4", T.score>=3),
               ("opens above + 15m vol", T.g_open & T.g_or),
               ("all 4 hard AND", T.score==4)]:
    a = T[m].win.values.astype(float)
    if len(a)<30: continue
    d = rng.choice(a,(8000,len(a)),replace=True).mean(1)-rng.choice(base,(8000,len(base)),replace=True).mean(1)
    print(f"{lbl:<26} n={len(a):>4}  win {a.mean():.0%}  lift {a.mean()-BASE:+.1%}  "
          f"95% CI [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]")

print("\n"+"="*100); print(">=2 of 4 -- per name and per year"); print("="*100)
g = T[T.score>=2]
print(g.groupby('sym').agg(n=('win','size'),win=('win','mean')).round(3).to_string())
print(g.assign(y=pd.to_datetime(g.day).dt.year).groupby('y').agg(n=('win','size'),win=('win','mean')).round(3).to_string())
