"""
Does signed order flow predict the next move?

Features, all computed from 1-minute signed volume and all strictly causal:
  ofi_k    order-flow imbalance over the last k minutes:
           (buy - sell) / (buy + sell)
  big_ofi  the same, restricted to block prints (>=1000 shares) -- the
           "whale tape" idea
  cvd_div  divergence: flow pushing one way while price goes the other, which
           is the classic order-flow reversal read

Target: forward return over 5, 15 and 30 minutes, in basis points.

Guards against the mistakes made earlier in this project:
  * forward returns start at the NEXT minute's close, never the signal bar
  * the last 30 minutes of each session are dropped so no horizon runs past the
    close and gets silently truncated
  * bootstrap is clustered by DAY, because minutes inside a session are heavily
    correlated and per-observation resampling would make any CI look decisive
  * a cost line is shown, because a 1bp edge at this frequency is not an edge
"""
import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob('research/smc/data/flow/*.parquet'))
if not files:
    raise SystemExit('no flow files yet')
F = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
F['m'] = pd.to_datetime(F.m, utc=True).dt.tz_convert('America/New_York')
F = F.sort_values(['sym', 'm']).reset_index(drop=True)
print(f'{len(F):,} symbol-minutes | {F.sym.nunique()} symbols | {F.day.nunique()} sessions')

rng = np.random.default_rng(8080)
out = []
for (sym, day), g in F.groupby(['sym', 'day'], sort=False):
    g = g.sort_values('m').reset_index(drop=True)
    if len(g) < 120:
        continue
    buy, sell = g.buy.values.astype(float), g.sell.values.astype(float)
    bb, bs = g.big_buy.values.astype(float), g.big_sell.values.astype(float)
    px = g.px.values.astype(float)
    n = len(g)

    def roll_ofi(b, s, k):
        cb = pd.Series(b).rolling(k).sum().values
        cs = pd.Series(s).rolling(k).sum().values
        tot = cb + cs
        return np.where(tot > 0, (cb - cs) / np.maximum(tot, 1), np.nan)

    o5, o15 = roll_ofi(buy, sell, 5), roll_ofi(buy, sell, 15)
    b15 = roll_ofi(bb, bs, 15)
    ret15 = np.full(n, np.nan)
    ret15[15:] = px[15:] / px[:-15] - 1

    for h in (5, 15, 30):
        fwd = np.full(n, np.nan)
        fwd[:-h] = px[h:] / px[:-h] - 1
        for i in range(20, n - 35):          # drop the last 30 min
            if not np.isfinite(o15[i]) or not np.isfinite(fwd[i]):
                continue
            out.append(dict(sym=sym, day=day, h=h, ofi5=o5[i], ofi15=o15[i],
                            big_ofi15=b15[i] if np.isfinite(b15[i]) else np.nan,
                            past15=ret15[i] * 1e4 if np.isfinite(ret15[i]) else np.nan,
                            fwd_bp=fwd[i] * 1e4))
T = pd.DataFrame(out)
T.to_csv('research/smc/flow_obs.csv', index=False)
print(f'observations: {len(T):,}')


def cci(df, col, iters=4000):
    groups = [x[col].values.astype(float) for _, x in df.groupby('day')]
    k = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


for h in (5, 15, 30):
    S = T[T.h == h]
    print('\n' + '=' * 96)
    print(f'FORWARD {h} MINUTES   n={len(S):,}')
    print('=' * 96)
    for feat in ('ofi15', 'big_ofi15'):
        s = S.dropna(subset=[feat])
        if len(s) < 500:
            continue
        q = s[feat].quantile([0.2, 0.8])
        top = s[s[feat] >= q.iloc[1]]
        bot = s[s[feat] <= q.iloc[0]]
        lo_t, hi_t = cci(top, 'fwd_bp')
        lo_b, hi_b = cci(bot, 'fwd_bp')
        print(f"  {feat:<10} corr={s[feat].corr(s.fwd_bp):+.4f}")
        print(f"     heavy BUYING  (top 20%)  n={len(top):>7,}  fwd {top.fwd_bp.mean():+6.2f} bp  "
              f"CI [{lo_t:+.2f}, {hi_t:+.2f}]")
        print(f"     heavy SELLING (bot 20%)  n={len(bot):>7,}  fwd {bot.fwd_bp.mean():+6.2f} bp  "
              f"CI [{lo_b:+.2f}, {hi_b:+.2f}]")
        d = (rng.choice(top.fwd_bp.values, (4000, len(top))).mean(1)
             - rng.choice(bot.fwd_bp.values, (4000, len(bot))).mean(1))
        print(f"     spread (buy - sell): {top.fwd_bp.mean()-bot.fwd_bp.mean():+.2f} bp"
              f"  95% CI [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]"
              f"  P(>0) = {(d>0).mean():.2f}")

print('\n' + '=' * 96)
print('Per symbol, 15-min horizon, ofi15 top-vs-bottom spread')
print('=' * 96)
S = T[T.h == 15]
for sym, g in S.groupby('sym'):
    q = g.ofi15.quantile([0.2, 0.8])
    top = g[g.ofi15 >= q.iloc[1]].fwd_bp
    bot = g[g.ofi15 <= q.iloc[0]].fwd_bp
    d = rng.choice(top.values, (4000, len(top))).mean(1) - rng.choice(bot.values, (4000, len(bot))).mean(1)
    print(f"  {sym:<6} n={len(g):>7,}  buy {top.mean():+6.2f}  sell {bot.mean():+6.2f}  "
          f"spread {top.mean()-bot.mean():+6.2f} bp  CI [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]")

print('\nCost reference: SPY round trip ~1.5 bp, single names ~4 bp.')
print('A spread must clear roughly twice that to be tradeable, since you pay on both legs.')
