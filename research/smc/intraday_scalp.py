"""
Intraday mean-reversion scalping -- the thing most retail "scalp bots" are.

Signal: session VWAP plus a rolling standard deviation of the price-VWAP gap.
When price stretches k sigma away from VWAP, fade it back toward VWAP.
Symmetric by construction: longs below, shorts above.

  entry   close of the 5-min bar that breaches k sigma
  target  VWAP
  stop    a further m sigma of extension
  time    flat by a max hold, and always flat by 15:55

Costs are the whole question at this horizon, so they are explicit and
per-symbol: half-spread each way plus slippage, in basis points. ETFs are
cheap, single names are not. Results are shown gross AND net so you can see
exactly how much of the edge the spread eats.
"""
import os
import numpy as np
import pandas as pd

UNIVERSE = ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
            'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']
# round-trip cost in basis points (half-spread x2 + slippage), by liquidity tier
COST_BP = {'SPY': 1.5, 'QQQ': 1.5, 'IWM': 2.0}
DEFAULT_COST_BP = 4.0

K_ENTRY = [1.5, 2.0, 2.5]
M_STOP = 1.5          # additional sigma beyond entry
MAXHOLD = 12          # 5-min bars = 1 hour
WARMUP = 6            # bars into the session before signals are allowed

rows = []
for sym in UNIVERSE:
    path = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(path):
        continue
    d = pd.read_parquet(path).between_time('09:30', '15:59').copy()
    d['day'] = d.index.date
    cost = COST_BP.get(sym, DEFAULT_COST_BP) / 10000.0

    for day, g in d.groupby('day'):
        if len(g) < 40:
            continue
        px = g.close.values
        hi, lo = g.high.values, g.low.values
        vol = g.volume.values.astype(float)
        tp = (g.high.values + g.low.values + g.close.values) / 3.0
        cum_pv = np.cumsum(tp * vol)
        cum_v = np.cumsum(vol)
        vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1), px)
        dev = px - vwap
        n = len(g)
        # causal rolling sigma of the deviation
        sig = pd.Series(dev).expanding(min_periods=WARMUP).std().shift(1).values

        for k in K_ENTRY:
            i = WARMUP
            while i < n - 2:
                s = sig[i]
                if not np.isfinite(s) or s <= 0:
                    i += 1
                    continue
                z = dev[i] / s
                direction = 0
                if z <= -k:
                    direction = 1
                elif z >= k:
                    direction = -1
                if direction == 0:
                    i += 1
                    continue
                entry = px[i]
                target = vwap[i]
                stop = entry - direction * M_STOP * s
                out = None
                for j in range(i + 1, min(i + 1 + MAXHOLD, n)):
                    if direction == 1:
                        if lo[j] <= stop:
                            out = (stop - entry) / entry
                            break
                        if hi[j] >= target:
                            out = (target - entry) / entry
                            break
                    else:
                        if hi[j] >= stop:
                            out = (entry - stop) / entry
                            break
                        if lo[j] <= target:
                            out = (entry - target) / entry
                            break
                if out is None:
                    j = min(i + MAXHOLD, n - 1)
                    out = ((px[j] - entry) if direction == 1 else (entry - px[j])) / entry
                rows.append(dict(sym=sym, day=day, k=k, dir=direction,
                                 gross=out, net=out - cost,
                                 held=j - i, year=pd.Timestamp(day).year))
                i = j + 1

T = pd.DataFrame(rows)
T.to_csv('research/smc/intraday_scalp.csv', index=False)

rng = np.random.default_rng(606)
def cci(df, col, iters=5000):
    groups = [g[col].values.astype(float) for _, g in df.groupby('day')]
    kk = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, kk, kk)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

print('=' * 108)
print(f'VWAP mean-reversion scalp -- {T.sym.nunique()} tickers, 2022-2026, 5-min bars, n={len(T)}')
print('=' * 108)
print(f"{'entry':<10}{'side':<8}{'n':>8}{'win':>7}{'gross bp':>10}{'net bp':>9}"
      f"{'  net 95% CI (bp)':<24}{'held':>6}")
for k in K_ENTRY:
    for lbl, m in (('both', T.k == k), ('long', (T.k == k) & (T.dir == 1)),
                   ('short', (T.k == k) & (T.dir == -1))):
        g = T[m]
        if len(g) < 200:
            continue
        lo, hi = cci(g, 'net')
        print(f"{str(k)+' sigma':<10}{lbl:<8}{len(g):>8}{(g.net>0).mean():>7.0%}"
              f"{g.gross.mean()*10000:>10.2f}{g.net.mean()*10000:>9.2f}"
              f"   [{lo*10000:+.2f}, {hi*10000:+.2f}]        {g.held.mean():>5.1f}")

best = T[T.k == 2.0]
print('\n' + '=' * 108)
print('2.0 sigma, both sides -- where does it stand after costs?')
print('=' * 108)
print(f"  gross {best.gross.mean()*10000:+.2f} bp   net {best.net.mean()*10000:+.2f} bp   "
      f"costs ate {(best.gross.mean()-best.net.mean())*10000:.2f} bp "
      f"= {(1 - best.net.mean()/best.gross.mean())*100:.0f}% of the gross edge"
      if best.gross.mean() != 0 else '')
print(f"  trades per day across the universe: {len(best)/best.day.nunique():.1f}")
print('\n  by ticker (net bp):')
pt = best.groupby('sym').agg(n=('net', 'size'), win=('net', lambda x: (x > 0).mean()),
                             net_bp=('net', lambda x: x.mean() * 10000)).round(2)
print(pt.sort_values('net_bp').to_string())
print('\n  by year (net bp):')
print(best.groupby('year').agg(n=('net', 'size'), win=('net', lambda x: (x > 0).mean()),
                               net_bp=('net', lambda x: x.mean() * 10000)).round(2).to_string())
