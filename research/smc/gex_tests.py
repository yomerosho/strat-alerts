"""
Three tests of the gamma-exposure thesis. 516 SPY sessions, Aug 2024 - Aug 2026.
GEX is measured 09:30-10:30; everything it is tested against happens after 10:30.
"""
import numpy as np
import pandas as pd

G = pd.read_csv('research/smc/gex_daily.csv')
G['day'] = pd.to_datetime(G.day).dt.date
rng = np.random.default_rng(3141)


def ci(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def terciles(df, by, cols):
    q = df[by].quantile([1/3, 2/3]).values
    lab = pd.cut(df[by], [-np.inf, q[0], q[1], np.inf], labels=['low', 'mid', 'high'])
    return df.groupby(lab, observed=True)[cols].mean().round(3), df.groupby(lab, observed=True).size()


print('=' * 100)
print('TEST A  Does GEX predict rest-of-day realised volatility?')
print('=' * 100)
print('raw correlation gex vs rest-of-day realised vol : %.3f' % G.gex_norm.corr(G.rest_rv))
print('raw correlation gex vs rest-of-day range        : %.3f' % G.gex_norm.corr(G.rest_range))
print('  ...but implied vol drives both. sigma vs rest_rv: %.3f' % G.sigma.corr(G.rest_rv))

# control for implied vol: residualise realised vol on sigma, then sort by GEX
x = G.sigma.values
y = G.rest_rv.values
b1, b0 = np.polyfit(x, y, 1)
G['rv_resid'] = y - (b0 + b1 * x)
yr = G.rest_range.values
c1, c0 = np.polyfit(x, yr, 1)
G['range_resid'] = yr - (c0 + c1 * x)
print('\nafter removing the implied-vol effect:')
print('  corr(gex, realised-vol residual) : %.3f' % G.gex_norm.corr(G.rv_resid))
print('  corr(gex, range residual)        : %.3f' % G.gex_norm.corr(G.range_resid))

t, n = terciles(G, 'gex_norm', ['rest_rv', 'rest_range', 'rv_resid', 'sigma'])
print('\nby GEX tercile:')
print(t.to_string())
print('n per tercile:', n.values)
lo = G[G.gex_norm <= G.gex_norm.quantile(1/3)].rv_resid.values
hi = G[G.gex_norm >= G.gex_norm.quantile(2/3)].rv_resid.values
d = rng.choice(hi, (10000, len(hi))).mean(1) - rng.choice(lo, (10000, len(lo))).mean(1)
print(f"\n  high-GEX minus low-GEX, vol residual: {hi.mean()-lo.mean():+.3f}"
      f"  95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]"
      f"  P(high<low as theory says) = {(d<0).mean():.2f}")

print('\n' + '=' * 100)
print('TEST B  Does the max-gamma strike pull the close toward it?')
print('=' * 100)
# placebo: mirror the magnet across the 10:30 price -- identical distance, no gamma
G['placebo'] = G.S1030 - (G.magnet - G.S1030)
G['d_magnet'] = (G.close - G.magnet).abs()
G['d_placebo'] = (G.close - G.placebo).abs()
sel = G[G.open_to_magnet > 0.05]        # ignore days where the magnet is already at spot
print(f"  sessions where the magnet is a meaningful distance away: {len(sel)}")
print(f"  mean |close - magnet|  : {sel.d_magnet.mean():.3f}")
print(f"  mean |close - placebo| : {sel.d_placebo.mean():.3f}")
diff = (sel.d_placebo - sel.d_magnet).values      # positive = magnet is closer = thesis holds
lo_, hi_ = ci(diff)
print(f"  placebo minus magnet   : {diff.mean():+.3f}  95% CI [{lo_:+.3f}, {hi_:+.3f}]"
      f"  P(magnet closer) = {(rng.choice(diff,(10000,len(diff))).mean(1)>0).mean():.2f}")
print(f"  close finished nearer the magnet than the placebo on {(diff>0).mean():.0%} of sessions")

print('\n' + '=' * 100)
print('TEST C  Does GEX regime condition the intraday VWAP scalp?')
print('=' * 100)
try:
    S = pd.read_csv('research/smc/intraday_scalp.csv')
    S['day'] = pd.to_datetime(S.day).dt.date
    S = S[S.k == 2.0]
    M = S.merge(G[['day', 'gex_norm']], on='day', how='inner')
    print(f"  matched {len(M)} scalp trades to {M.day.nunique()} GEX sessions")
    q1, q2 = M.gex_norm.quantile([1/3, 2/3])
    for lbl, m in (('GEX low (short gamma)', M.gex_norm <= q1),
                   ('GEX mid', (M.gex_norm > q1) & (M.gex_norm < q2)),
                   ('GEX high (long gamma)', M.gex_norm >= q2)):
        g = M[m]
        lo_, hi_ = ci(g.net.values)
        print(f"  {lbl:<24} n={len(g):>6}  win {(g.net>0).mean():.0%}  "
              f"net {g.net.mean()*10000:+.2f} bp  95% CI [{lo_*10000:+.2f}, {hi_*10000:+.2f}]")
    a = M[M.gex_norm >= q2].net.values
    b = M[M.gex_norm <= q1].net.values
    dd = rng.choice(a, (10000, len(a))).mean(1) - rng.choice(b, (10000, len(b))).mean(1)
    print(f"\n  high-GEX minus low-GEX: {(a.mean()-b.mean())*10000:+.2f} bp"
          f"  95% CI [{np.percentile(dd,2.5)*10000:+.2f}, {np.percentile(dd,97.5)*10000:+.2f}]"
          f"  P(>0) = {(dd>0).mean():.2f}")
except FileNotFoundError:
    print('  intraday_scalp.csv not found')
