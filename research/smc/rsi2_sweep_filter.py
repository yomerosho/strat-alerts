"""
Does the external-liquidity sweep add information ON TOP of RSI(2) reversion?

Baseline: the committed RSI(2) system -- daily bars, RSI(2)<10, close>SMA200,
enter next open, exit next open after close>SMA5, 10-day cap. 15 tickers,
2016-2026. (Same engine as rsi2_timestop.py.)

Filter under test: the signal day's relationship to HTF external liquidity.
On daily bars the external level is the prior day's low (PDL). Buckets:
  sweep      low < PDL and close > PDL   (ICT sweep-and-reclaim)
  deep       low < PDL and close <= PDL  (undercut, no reclaim)
  none       low >= PDL                  (no external liquidity taken)
Also tested: swept the LOWEST LOW OF THE PRIOR 5 DAYS (a stronger external
level), same three buckets.

This is a conditioning test on the SAME trades the baseline takes, so the
only question is whether the sweep tag predicts the trade's return.
Date-clustered bootstrap for every bucket. If a bucket separates, a live
sequential re-run with the filter follows.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('research/smc/data/daily_2015.parquet')
df['ts'] = pd.to_datetime(df['timestamp'])
RSI_TH = 10
CAP = 10
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
        o=g.open.values, h=g.high.values, l=g.low.values, c=g.close.values,
        s5=g.sma5.values, ts=g.ts.values, n=len(g))


def bucket(l_sig, c_sig, lev):
    if not np.isfinite(lev):
        return 'none'
    if l_sig < lev:
        return 'sweep' if c_sig > lev else 'deep'
    return 'none'


def simulate(allowed=None):
    """allowed: None = take all; else set of buckets (PDL def) to take.
       Sequential per ticker, so filtering frees capital like live trading."""
    out = []
    for sym, P in PREP.items():
        sig, o, h, l, c, s5, ts, n = (P['sig'], P['o'], P['h'], P['l'],
                                      P['c'], P['s5'], P['ts'], P['n'])
        i = 1
        while i < n - 2:
            if not sig[i]:
                i += 1
                continue
            pdl = l[i - 1]
            low5 = l[max(0, i - 5):i].min()
            b1 = bucket(l[i], c[i], pdl)
            b5 = bucket(l[i], c[i], low5)
            if allowed is not None and b1 not in allowed:
                i += 1
                continue
            e = i + 1
            entry = o[e]
            if not np.isfinite(entry) or entry <= 0:
                i += 1
                continue
            exit_bar = None
            for k in range(e, min(e + CAP + 1, n - 1)):
                if c[k] > s5[k]:
                    exit_bar = k + 1
                    break
            if exit_bar is None:
                exit_bar = min(e + CAP, n - 1)
            out.append(dict(sym=sym, date=pd.Timestamp(ts[i]).date(),
                            b_pdl=b1, b_low5=b5,
                            ret=o[exit_bar] / entry - 1,
                            held=exit_bar - e))
            i = exit_bar
    return pd.DataFrame(out)


def cci(t, iters=6000):
    groups = [g.ret.values for _, g in t.groupby('date')]
    k = len(groups)
    if k < 5:
        return np.nan, np.nan
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(t, label):
    if len(t) == 0:
        print(f'  {label:22s} n=0')
        return
    lo, hi = cci(t)
    w = (t.ret > 0).mean()
    print(f'  {label:22s} n={len(t):5d}  win={w:6.1%}  avg={t.ret.mean()*100:+.3f}%  '
          f'med={t.ret.median()*100:+.3f}%  CI=[{lo*100:+.3f}%, {hi*100:+.3f}%]  '
          f'hold={t.held.mean():.1f}d')


T = simulate()
print('=' * 100)
print(f'RSI(2)<{RSI_TH} + >SMA200, exit close>SMA5, cap {CAP}d   '
      f'n={len(T)}  2016-2026, 15 tickers')
print('=' * 100)
line(T, 'ALL (baseline)')
print('\nsignal day vs PRIOR DAY LOW (PDL):')
for b in ('sweep', 'deep', 'none'):
    line(T[T.b_pdl == b], f'  {b}')
print('\nsignal day vs 5-DAY LOW:')
for b in ('sweep', 'deep', 'none'):
    line(T[T.b_low5 == b], f'  {b}')

print('\nper-year avg ret (%), PDL buckets:')
y = pd.to_datetime(T.date.astype(str)).dt.year
piv = T.assign(y=y).pivot_table(values='ret', index='y', columns='b_pdl',
                                aggfunc=['mean', 'count'])
piv['mean'] = piv['mean'] * 100
print(piv.round(3).to_string())

print('\nLIVE re-runs (sequential capital, filter changes what gets taken):')
for allowed, lab in ((None, 'take everything'),
                     ({'sweep'}, 'sweep-and-reclaim only'),
                     ({'sweep', 'deep'}, 'any undercut'),
                     ({'deep'}, 'deep undercut only'),
                     ({'none'}, 'no-sweep only')):
    line(simulate(allowed), lab)
