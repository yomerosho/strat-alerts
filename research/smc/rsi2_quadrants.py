"""
Does RSI(2) reversion work short, or only long?

Four quadrants, same mechanics throughout (entry and exit at the NEXT OPEN,
10-day cap, no stop). The only things that change are direction and which side
of the 200-day SMA we require.

  1  LONG   RSI2 < 10, close ABOVE 200SMA   -- the deployed strategy
  2  SHORT  RSI2 > 90, close BELOW 200SMA   -- the exact mirror
  3  LONG   RSI2 < 10, close BELOW 200SMA   -- buying dips in a downtrend
  4  SHORT  RSI2 > 90, close ABOVE 200SMA   -- fading strength in an uptrend

Caveat stated up front: 2016-2026 was overwhelmingly bullish, which handicaps
anything short. Quadrant 2's sample is small for that reason -- there simply
were not many overbought days below the 200SMA.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('research/smc/data/daily_2015.parquet')
df['ts'] = pd.to_datetime(df['timestamp'])
rng = np.random.default_rng(777)


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run(direction, rsi_lo, rsi_hi, above_sma, cap=10):
    out = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('ts').reset_index(drop=True)
        if len(g) < 300:
            continue
        g['rsi2'] = rsi(g.close, 2)
        g['sma5'] = g.close.rolling(5).mean()
        g['sma200'] = g.close.rolling(200).mean()
        trend = (g.close > g.sma200) if above_sma else (g.close < g.sma200)
        sig = ((g.rsi2 > rsi_lo) & (g.rsi2 < rsi_hi) & trend).fillna(False).values
        o, c, s5, ts = g.open.values, g.close.values, g.sma5.values, g.ts.values
        n = len(g)
        i = 0
        while i < n - 2:
            if not sig[i]:
                i += 1
                continue
            e = i + 1
            entry = o[e]
            if not np.isfinite(entry) or entry <= 0:
                i += 1
                continue
            xb = None
            for k in range(e, min(e + cap + 1, n - 1)):
                done = (c[k] > s5[k]) if direction == 1 else (c[k] < s5[k])
                if done:
                    xb = k + 1
                    break
            if xb is None:
                xb = min(e + cap, n - 1)
            ret = (o[xb] / entry - 1) if direction == 1 else (entry / o[xb] - 1)
            out.append(dict(sym=sym, date=pd.Timestamp(ts[e]).date(), ret=ret,
                            held=xb - e, year=pd.Timestamp(ts[e]).year))
            i = xb
    return pd.DataFrame(out)


def cci(d, iters=6000):
    groups = [g.ret.values for _, g in d.groupby('date')]
    k = len(groups)
    m = np.empty(iters)
    for i in range(iters):
        m[i] = np.concatenate([groups[j] for j in rng.integers(0, k, k)]).mean()
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


QUAD = [
    ('1  LONG  oversold, ABOVE 200SMA   (deployed)',  1, -1, 10, True),
    ('2  SHORT overbought, BELOW 200SMA (mirror)',   -1, 90, 101, False),
    ('3  LONG  oversold, BELOW 200SMA',               1, -1, 10, False),
    ('4  SHORT overbought, ABOVE 200SMA',            -1, 90, 101, True),
]

print('=' * 112)
print('RSI(2) reversion by quadrant -- daily, 15 tickers, 2016-2026, next-open fills')
print('=' * 112)
print(f"{'quadrant':<44}{'n':>6}{'win':>7}{'avgW':>8}{'avgL':>8}{'exp':>9}"
      f"{'  95% CI (clustered)':<26}{'PF':>6}{'worst':>8}")
res = {}
for lbl, d, lo, hi, above in QUAD:
    t = run(d, lo, hi, above)
    res[lbl] = t
    if len(t) < 20:
        print(f"{lbl:<44}{len(t):>6}   too few signals")
        continue
    w, l = t[t.ret > 0].ret, t[t.ret <= 0].ret
    clo, chi, p = cci(t)
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.inf
    print(f"{lbl:<44}{len(t):>6}{(t.ret>0).mean():>7.0%}{w.mean()*100:>7.2f}%"
          f"{l.mean()*100:>7.2f}%{t.ret.mean()*100:>8.2f}%"
          f"   [{clo*100:+.2f}%, {chi*100:+.2f}%]      {pf:>6.2f}{t.ret.min()*100:>7.1f}%")

print('\n' + '=' * 112)
print('The short mirror, year by year (is it just the bull market?)')
print('=' * 112)
s = res['2  SHORT overbought, BELOW 200SMA (mirror)']
if len(s) > 20:
    print(s.groupby('year').agg(n=('ret', 'size'), win=('ret', lambda x: (x > 0).mean()),
                                mean=('ret', 'mean')).round(3).to_string())
    print(f"\n  2022 only (the one bear year): ", end='')
    b = s[s.year == 2022]
    print(f"n={len(b)}  win {(b.ret>0).mean():.0%}  mean {b.ret.mean()*100:+.2f}%" if len(b) else 'no trades')

print('\n' + '=' * 112)
print('Signal availability -- how often each quadrant even fires')
print('=' * 112)
for lbl, t in res.items():
    print(f"  {lbl:<44} {len(t):>5} trades = {len(t)/15/10.6:>4.1f} per ticker per year")
