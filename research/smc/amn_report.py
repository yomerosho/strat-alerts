"""Verdict on both AMN alert conditions, held to the project's usual bar."""
import numpy as np
import pandas as pd

T = pd.read_csv('research/smc/amn_signals.csv')
rng = np.random.default_rng(7)


def cci(g, col, iters=4000):
    """Bootstrap CI clustered by DATE -- signals on one day are not independent."""
    grp = [x[col].values.astype(float) for _, x in g.groupby('day')]
    k = len(grp)
    if k < 5:
        return np.nan, np.nan
    idx = rng.integers(0, k, (iters, k))
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([grp[j] for j in idx[i]]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(lbl, g, col):
    if len(g) < 30:
        print(f'{lbl:<30}{len(g):>7}    too few')
        return
    x = g[col].values.astype(float)
    w, l = x[x > 0], x[x <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    lo, hi = cci(g, col)
    print(f'{lbl:<30}{len(g):>7}{(x > 0).mean():>7.0%}{x.mean():>+9.3f}'
          f'   [{lo:+.3f}, {hi:+.3f}]{pf:>7.2f}')


print('=' * 92)
print('AMN 6-POINT SEQUENCE -- both alert conditions')
print('=' * 92)
print('  A = "1-5 sequence qualified"      B = "point 6 sweep and reclaim"')
print(f'  {len(T)} signals, 15 tickers, 2022-01 -> 2026-08, 5-min bars, 2bp each way\n')

for m, name in [(1.0, '1R target'), (2.0, '2R target'), (3.0, '3R target')]:
    be = 1 / (1 + m)
    print(f'--- {name}   (break-even win rate {be:.0%}) ' + '-' * 40)
    print(f"{'':<30}{'n':>7}{'win':>7}{'expR':>9}{'   95% CI':<22}{'PF':>7}")
    for k in ['A', 'B']:
        g = T[T.kind == k]
        line(f'  {k}  all', g, 'r%g' % m)
        line(f'  {k}  long', g[g.dir == 1], 'r%g' % m)
        line(f'  {k}  short', g[g.dir == -1], 'r%g' % m)
    print()

print('--- committed scale-out: half at 1R, stop to BE, runner to 2R ' + '-' * 20)
print(f"{'':<30}{'n':>7}{'win':>7}{'expR':>9}{'   95% CI':<22}{'PF':>7}")
for k in ['A', 'B']:
    line(f'  {k}  all', T[T.kind == k], 'rSO')

print('\n' + '=' * 92)
print('VS MATCHED CONTROL  (same ticker, same day, same direction, same stop distance,')
print('                     random entry bar -- isolates the setup from the drift)')
print('=' * 92)
print(f"{'':<34}{'n':>7}{'signal':>9}{'control':>9}{'diff':>9}{'  95% CI of diff':<24}{'P(>0)':>7}")
for k in ['A', 'B']:
    for col in ['r1', 'r2', 'r3', 'rSO']:
        g = T[(T.kind == k)].dropna(subset=['c_' + col])
        if len(g) < 30:
            continue
        a = g[col].values.astype(float)
        b = g['c_' + col].values.astype(float)
        d = a - b
        gg = g.assign(_d=d)
        lo, hi = cci(gg, '_d')
        grp = [x['_d'].values for _, x in gg.groupby('day')]
        kk = len(grp)
        idx = rng.integers(0, kk, (4000, kk))
        boot = np.array([np.concatenate([grp[j] for j in idx[i]]).mean() for i in range(4000)])
        print(f'  {k}  {col:<28}{len(g):>7}{a.mean():>+9.3f}{b.mean():>+9.3f}'
              f'{d.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{(boot > 0).mean():>7.2f}')

print('\n' + '=' * 92)
print('DOES THE 1H ZONE FLAG ON POINT 4 ADD ANYTHING?  (alert A only -- it is')
print('the only one that records it)')
print('=' * 92)
A = T[(T.kind == 'A')].copy()
A['inzone'] = A.inzone.astype(str)
print(f"{'':<30}{'n':>7}{'win':>7}{'expR':>9}{'   95% CI':<22}{'PF':>7}")
for v, lbl in [('True', '  point 4 IN 1H zone'), ('False', '  point 4 not in zone')]:
    line(lbl, A[A.inzone == v], 'r2')

print('\n' + '=' * 92)
print('STABILITY  (2R target, expectancy in R)')
print('=' * 92)
for k in ['A', 'B']:
    g = T[T.kind == k]
    pt = g.groupby('sym').r2.agg(['size', 'mean']).round(3)
    print(f'\n{k} by ticker -- positive on {int((pt["mean"] > 0).sum())}/{len(pt)}')
    print('   ' + '  '.join(f'{s}:{v:+.2f}' for s, v in pt['mean'].items()))
    py = g.groupby('year').r2.agg(['size', 'mean']).round(3)
    print(f'{k} by year:  ' + '  '.join(f'{int(y)}:{v:+.3f}(n={int(n)})'
                                        for y, (n, v) in py.iterrows()))

print('\n' + '=' * 92)
print('VERDICT vs the standard used on every other strategy here:')
print('  expR > +0.15R,  date-clustered CI clear of zero,  n >= 100,')
print('  stable across tickers and years,  AND beats the matched control')
print('=' * 92)
for k in ['A', 'B']:
    g = T[T.kind == k]
    best = None
    for col in ['r1', 'r2', 'r3', 'rSO']:
        lo, hi = cci(g, col)
        gg = g.dropna(subset=['c_' + col])
        d = (gg[col].values - gg['c_' + col].values)
        grp = [x.values for _, x in pd.Series(d, index=gg.day.values).groupby(level=0)]
        kk = len(grp)
        idx = rng.integers(0, kk, (2000, kk))
        boot = np.array([np.concatenate([grp[j] for j in idx[i]]).mean() for i in range(2000)])
        ok = (g[col].mean() > 0.15) and (lo > 0) and (len(g) >= 100) and ((boot > 0).mean() > 0.95)
        print(f'  {k} {col:<5} expR {g[col].mean():>+7.3f}  CI_lo {lo:>+7.3f}  '
              f'beats control P={(boot > 0).mean():.2f}   -> {"PASS" if ok else "FAIL"}')
