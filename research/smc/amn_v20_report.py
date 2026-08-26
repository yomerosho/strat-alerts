import numpy as np
import pandas as pd

T = pd.read_csv('research/smc/amn_v20.csv')
rng = np.random.default_rng(11)


def cci(g, col, iters=3000):
    grp = [x[col].values.astype(float) for _, x in g.groupby('day')]
    k = len(grp)
    if k < 5:
        return np.nan, np.nan
    idx = rng.integers(0, k, (iters, k))
    m = np.array([np.concatenate([grp[j] for j in idx[i]]).mean() for i in range(iters)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(lbl, g, col, be=None):
    if len(g) < 30:
        print(f'{lbl:<34}{len(g):>7}   too few'); return
    x = g[col].values.astype(float)
    w, l = x[x > 0], x[x <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    lo, hi = cci(g, col)
    flag = '  <<' if lo > 0 and x.mean() > 0.15 else ''
    print(f'{lbl:<34}{len(g):>7}{(x>0).mean():>7.0%}{x.mean():>+9.3f}'
          f'   [{lo:+.3f}, {hi:+.3f}]{pf:>7.2f}{flag}')


print('=' * 96)
print('AMN point 6 (sweep + reclaim) -- CURRENT v20 logic')
print(f'{len(T)} entries, 15 tickers, 2022-01 -> 2026-08, 5-min, 2bp each way')
print('=' * 96)

for m, nm in [(1.0, '1R'), (2.0, '2R'), (3.0, '3R')]:
    print(f'\n--- {nm} target (break-even win {1/(1+m):.0%}) ' + '-' * 46)
    print(f"{'':<34}{'n':>7}{'win':>7}{'expR':>9}{'   95% CI':<22}{'PF':>7}")
    col = 'r%g' % m
    line('  all point-6 entries', T, col)
    line('  + FTFC 3  (5m,15m,1h)', T[T.ftfc3 == 1], col)
    line('  + FTFC 5  (+4h,1D)', T[T.ftfc5 == 1], col)
    line('  + point 4 in 1H zone', T[T.inzone], col)
    line('  + FTFC 3 AND 1H zone', T[(T.ftfc3 == 1) & (T.inzone)], col)
    line('  AGAINST FTFC 3', T[T.ftfc3 == 0], col)

print('\n' + '=' * 96)
print('DOES DIRECTIONAL AGREEMENT HELP AT ALL?  expR by how many of the 5 TFs agree')
print('=' * 96)
print(f"{'TFs agreeing':<34}{'n':>7}{'win':>7}{'expR(2R)':>10}{'   95% CI':<22}")
for k in range(6):
    g = T[T.agree == k]
    if len(g) < 30:
        print(f'  {k}/5{"":<29}{len(g):>7}   too few'); continue
    x = g.r2.values.astype(float); lo, hi = cci(g, 'r2')
    print(f'  {k}/5{"":<29}{len(g):>7}{(x>0).mean():>7.0%}{x.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]')

print('\n' + '=' * 96)
print('VS MATCHED CONTROL (same ticker/day/direction/stop distance, random entry bar)')
print('=' * 96)
print(f"{'':<34}{'n':>7}{'signal':>9}{'control':>9}{'diff':>9}{'  P(better)':>12}")
for lbl, g in [('all entries', T), ('FTFC 3', T[T.ftfc3 == 1]),
               ('FTFC 5', T[T.ftfc5 == 1]), ('FTFC 3 + zone', T[(T.ftfc3 == 1) & (T.inzone)])]:
    for col in ['r2', 'rSO']:
        gg = g.dropna(subset=['c_' + col])
        if len(gg) < 30:
            continue
        a = gg[col].values.astype(float); b = gg['c_' + col].values.astype(float)
        dd = pd.Series(a - b, index=gg.day.values)
        grp = [v.values for _, v in dd.groupby(level=0)]
        kk = len(grp); idx = rng.integers(0, kk, (3000, kk))
        boot = np.array([np.concatenate([grp[j] for j in idx[i]]).mean() for i in range(3000)])
        print(f'  {lbl} [{col}]{"":<{max(0,20-len(lbl))}}{len(gg):>7}{a.mean():>+9.3f}'
              f'{b.mean():>+9.3f}{(a-b).mean():>+9.3f}{(boot>0).mean():>12.2f}')

print('\n' + '=' * 96)
print('SIGNAL RATE (what actually reaches your phone, 16 tickers)')
print('=' * 96)
sessions = T.day.nunique()
for lbl, g in [('no gate', T), ('FTFC 3', T[T.ftfc3 == 1]),
               ('FTFC 5', T[T.ftfc5 == 1]), ('FTFC 3 + zone', T[(T.ftfc3 == 1) & (T.inzone)])]:
    per_day = len(g) / sessions / 15 * 16
    print(f'  {lbl:<28}{len(g):>7} signals   ~{per_day:.1f} alerts/day across 16 tickers')

print('\nstability of the best gate (FTFC 3, 2R) by year:')
g = T[T.ftfc3 == 1]
print(g.groupby('year').r2.agg(['size', 'mean']).round(3).to_string())
print('\nby ticker:')
pt = g.groupby('sym').r2.agg(['size', 'mean']).round(3)
print(f'  positive on {int((pt["mean"]>0).sum())}/{len(pt)}: ' +
      '  '.join(f'{s}:{v:+.2f}' for s, v in pt['mean'].items()))
