"""Judge the SMC setup against the stated bar:
   expectancy > +0.15R, 95% CI clear of zero, stable across tickers and years,
   n >= 100 -- and the win rate must beat break-even for the R:R used."""
import numpy as np
import pandas as pd

T = pd.read_csv('research/smc/smc_strategy.csv')
F = T[T.filled == 1].copy()
rng = np.random.default_rng(4242)

def boot(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return (np.nan, np.nan, np.nan)
    b = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return (np.percentile(b, 2.5), np.percentile(b, 97.5), (b > 0).mean())

def line(lbl, g, tm, minn=25):
    col_r, col_w = 'r%g' % tm, 'w%g' % tm
    g = g.dropna(subset=[col_r])
    if len(g) < minn:
        return
    x = g[col_r].values.astype(float)
    lo, hi, p = boot(x)
    be = 1.0 / (1.0 + tm)
    tr_ = g[g.y < 2025][col_r]
    ho = g[g.y >= 2025][col_r]
    print(f"{lbl:<34}{len(g):>6}{g[col_w].mean():>7.0%}{be:>8.0%}"
          f"{x.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{p:>7.2f}"
          f"{(tr_.mean() if len(tr_)>20 else np.nan):>+9.3f}{(ho.mean() if len(ho)>20 else np.nan):>+9.3f}")

HDR = (f"{'variant':<34}{'n':>6}{'win':>7}{'b/e':>8}{'expR':>9}"
       f"{'   95% CI':<20}{'P>0':>5}{'train':>9}{'hold':>9}")

print('=' * 112)
print(f"Signals {len(T)} | filled {len(F)} | fill rate {T.filled.mean():.0%} | "
      f"tickers {T.sym.nunique()} | 2022-2026")
print('=' * 112)
print("Fill rate and R size by zone")
print(T.groupby('zone').agg(signals=('filled', 'size'), fill=('filled', 'mean'),
                            medR_atr=('R_atr', 'median'), medR_pct=('Rpct', 'median')).round(2).to_string())

for tm in (1.0, 2.0, 3.0):
    print('\n' + '=' * 112)
    print(f'TARGET {tm:g}R   (break-even win rate {1/(1+tm):.0%})')
    print('=' * 112)
    print(HDR)
    for z in ['fvg_mid', 'fvg_far', 'ob_mid', 'ob_edge']:
        line(f'  {z}  all', F[F.zone == z], tm)
    print('  ' + '-' * 108)
    for z in ['fvg_mid', 'ob_mid']:
        line(f'  {z}  long only', F[(F.zone == z) & (F.dir == 1)], tm)
        line(f'  {z}  short only', F[(F.zone == z) & (F.dir == -1)], tm)
        line(f'  {z}  long + daily up', F[(F.zone == z) & (F.dir == 1) & (F.trend_up == True)], tm)
        line(f'  {z}  short + daily down', F[(F.zone == z) & (F.dir == -1) & (F.trend_up == False)], tm)
        line(f'  {z}  with daily trend', F[(F.zone == z) & (((F.dir == 1) & (F.trend_up == True)) | ((F.dir == -1) & (F.trend_up == False)))], tm)
        print('  ' + '-' * 108)

print('\n' + '=' * 112)
print('Best cell by ticker and year (fvg_mid, 2R, all)')
print('=' * 112)
g = F[F.zone == 'fvg_mid'].dropna(subset=['r2'])
print(g.groupby('sym').agg(n=('r2', 'size'), win=('w2', 'mean'), expR=('r2', 'mean')).round(2).sort_values('n', ascending=False).to_string())
print()
print(g.groupby('y').agg(n=('r2', 'size'), win=('w2', 'mean'), expR=('r2', 'mean')).round(2).to_string())

print('\n' + '=' * 112)
print('VERDICT vs the bar: expR > +0.15, CI clear of zero, n >= 100, stable')
print('=' * 112)
best = []
for z in F.zone.unique():
    for tm in (1.0, 2.0, 3.0):
        col = 'r%g' % tm
        for lbl, sub in (('all', F[F.zone == z]),
                         ('with trend', F[(F.zone == z) & (((F.dir == 1) & (F.trend_up == True)) | ((F.dir == -1) & (F.trend_up == False)))])):
            s = sub.dropna(subset=[col])
            if len(s) < 100:
                continue
            x = s[col].values.astype(float)
            lo, hi, p = boot(x)
            best.append((z, tm, lbl, len(s), x.mean(), lo, hi))
best.sort(key=lambda r: -r[4])
for z, tm, lbl, nn, m, lo, hi in best[:8]:
    ok = 'PASS' if (m > 0.15 and lo > 0) else 'fail'
    print(f"  {z:<10} {tm:g}R {lbl:<11} n={nn:<5} expR={m:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  -> {ok}")
if not best:
    print('  no variant reached n=100')
