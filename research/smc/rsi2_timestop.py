"""
Does a hard time-stop help RSI(2) reversion?

The day-5 cliff in the holding-period table is partly SELECTION: trades still
open on day 5 are, by definition, the ones that have not bounced. Cutting there
also kills the 48% of day-5 trades that still work. So test it properly.

Two comparisons, because they answer different questions:

  A  MATCHED  -- same entry dates for every cap, so only the exit rule differs.
                 Isolates the effect of the time-stop itself.
  B  LIVE     -- full re-run per cap. A shorter cap frees capital sooner and
                 lets the next signal be taken, so trade counts differ. This is
                 what you would actually experience.

Bootstrap is clustered by entry date: trades opened the same day across tickers
share a market move.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('research/smc/data/daily_2015.parquet')
df['ts'] = pd.to_datetime(df['timestamp'])
RSI_TH = 10
rng = np.random.default_rng(31337)


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


PREP = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('ts').reset_index(drop=True)
    if len(g) < 300:
        continue
    g['rsi2'] = rsi(g.close, 2)
    g['sma5'] = g.close.rolling(5).mean()
    g['sma200'] = g.close.rolling(200).mean()
    PREP[sym] = dict(
        sig=((g.rsi2 < RSI_TH) & (g.close > g.sma200)).fillna(False).values,
        o=g.open.values, c=g.close.values, s5=g.sma5.values,
        ts=g.ts.values, n=len(g))


def simulate(cap, sequential=True, entry_bars=None):
    """sequential=True re-runs the scan (trade count varies with cap).
       entry_bars given -> replay those exact entries with this cap."""
    out = []
    for sym, P in PREP.items():
        sig, o, c, s5, ts, n = P['sig'], P['o'], P['c'], P['s5'], P['ts'], P['n']
        if entry_bars is not None:
            starts = [e for (s, e) in entry_bars if s == sym]
        else:
            starts = None
        i = 0
        idx = 0
        while True:
            if starts is not None:
                if idx >= len(starts):
                    break
                e = starts[idx]
                idx += 1
            else:
                while i < n - 2 and not sig[i]:
                    i += 1
                if i >= n - 2:
                    break
                e = i + 1
            if e >= n - 1:
                continue
            entry = o[e]
            if not np.isfinite(entry) or entry <= 0:
                i = e
                continue
            exit_bar = None
            for k in range(e, min(e + cap + 1, n - 1)):
                if c[k] > s5[k]:
                    exit_bar = k + 1
                    break
            capped = exit_bar is None
            if capped:
                exit_bar = min(e + cap, n - 1)
            out.append(dict(sym=sym, e=e, date=pd.Timestamp(ts[e]).date(),
                            ret=o[exit_bar] / entry - 1, held=exit_bar - e,
                            capped=int(capped)))
            i = exit_bar
    return pd.DataFrame(out)


def cci(df_, col='ret', iters=6000):
    groups = [g[col].values for _, g in df_.groupby('date')]
    k = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def dd(x):
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(eq) - eq).max())


base = simulate(10)
base_entries = list(zip(base.sym, base.e))

print('=' * 112)
print('A  MATCHED -- identical entry dates, only the time-stop changes   (n=%d each)' % len(base))
print('=' * 112)
print(f"{'cap':<8}{'win':>7}{'avgW':>8}{'avgL':>8}{'exp':>9}{'  95% CI (date-clustered)':<28}"
      f"{'PF':>6}{'worst':>8}{'capped':>8}{'totRet':>9}{'maxDD':>8}")
matched = {}
for cap in (3, 4, 5, 6, 7, 10):
    t = simulate(cap, entry_bars=base_entries)
    matched[cap] = t
    w, l = t[t.ret > 0].ret, t[t.ret <= 0].ret
    lo, hi = cci(t)
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    print(f"{cap:<8}{(t.ret>0).mean():>7.0%}{w.mean()*100:>7.2f}%{l.mean()*100:>7.2f}%"
          f"{t.ret.mean()*100:>8.2f}%   [{lo*100:+.2f}%, {hi*100:+.2f}%]      "
          f"{pf:>6.2f}{t.ret.min()*100:>7.1f}%{t.capped.mean():>8.0%}"
          f"{t.ret.sum()*100:>8.0f}%{dd(t.ret.values)*100:>7.0f}%")

b = matched[10]
print('\n  paired difference vs the 10-day cap (same trades, clustered by date):')
for cap in (3, 4, 5, 6, 7):
    t = matched[cap]
    j = b[['sym', 'e', 'ret']].merge(t[['sym', 'e', 'ret']], on=['sym', 'e'], suffixes=('_10', '_x'))
    j = j.merge(b[['sym', 'e', 'date']], on=['sym', 'e'])
    d = j.groupby('date').apply(lambda g: (g.ret_x - g.ret_10).mean(), include_groups=False).values
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(6000)])
    print(f"   cap {cap:<3} {d.mean()*100:+6.2f}%   95% CI [{np.percentile(bs,2.5)*100:+.2f}%, "
          f"{np.percentile(bs,97.5)*100:+.2f}%]   P(better than cap 10) = {(bs>0).mean():.2f}")

print('\n' + '=' * 112)
print('B  LIVE -- full re-run per cap; a shorter hold frees capital and changes the trade count')
print('=' * 112)
print(f"{'cap':<8}{'n':>7}{'win':>7}{'exp':>9}{'  95% CI':<26}{'PF':>6}{'totRet':>9}{'maxDD':>8}{'ret/DD':>8}")
for cap in (3, 4, 5, 6, 7, 10):
    t = simulate(cap)
    w, l = t[t.ret > 0].ret, t[t.ret <= 0].ret
    lo, hi = cci(t)
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    D = dd(t.ret.values)
    print(f"{cap:<8}{len(t):>7}{(t.ret>0).mean():>7.0%}{t.ret.mean()*100:>8.2f}%"
          f"   [{lo*100:+.2f}%, {hi*100:+.2f}%]     {pf:>6.2f}{t.ret.sum()*100:>8.0f}%"
          f"{D*100:>7.0f}%{(t.ret.sum()/D if D else np.nan):>8.1f}")
