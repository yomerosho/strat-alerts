"""
All four gates, extended-hours data, fillable entries only.

Every session of the 15 watchlist tickers is scored 0-4:
  g1 gap        premarket 09:25 vs prior close, top third of this symbol's own last 60 gaps
  g2 pm volume  04:00-09:29 volume >= 1.75x its own 20-day average
  g3 open       RTH open >= highest high of the last 10 RTH sessions
  g4 or15 vol   09:30-09:45 volume >= 3.6x (20-day avg daily volume * 15/390)

g1 and g2 are known at 09:30; g3 at 09:30; g4 at 09:45. So a 4/4 signal is
complete at 09:45 and the earliest honest entry is the 09:45 close.

Also tested: g1+g2 alone, which are complete BEFORE the bell, so they can be
traded at the open -- the one thing extended hours buys that RTH cannot.

Exits are identical everywhere: -1.0 ATR14(daily) stop, +1.5 ATR target,
5-session cap. Nothing here enters at a price that did not trade.
"""
import os
import numpy as np
import pandas as pd

LVL_N, STOP_ATR, TGT_ATR, HOLD = 10, 1.0, 1.5, 5
GAP_PCT, GAP_WIN, PM_MULT, OR_MULT = 66.7, 60, 1.75, 3.6
UNIVERSE = ['IWM', 'SPY', 'QQQ', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
            'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']

rows = []
for sym in UNIVERSE:
    path = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(path):
        continue
    d5 = pd.read_parquet(path)
    d5['d'] = d5.index.date
    rth = d5.between_time('09:30', '15:59')
    pm = d5.between_time('04:00', '09:29')
    daily = rth.groupby('d').agg(o=('open', 'first'), h=('high', 'max'),
                                 l=('low', 'min'), c=('close', 'last'),
                                 v=('volume', 'sum'))
    daily = daily[daily.v > 0]
    pmv = pm.groupby('d').volume.sum()
    pmc = pm.groupby('d').close.last()
    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(),
                    (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    daily['atr'] = tr.rolling(14).mean()
    days = list(daily.index)
    gaps = pd.Series({d: (pmc.get(d, np.nan) / daily.c.shift().get(d, np.nan) - 1) * 100
                      for d in days})
    r_h, r_l, r_c = rth.high.values, rth.low.values, rth.close.values
    r_idx = rth.index

    for k in range(60, len(days) - 1):
        day = days[k]
        prior = daily.iloc[:k]
        lvl = prior.h.iloc[-LVL_N:].max()
        atr = prior.atr.iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sess = rth[rth.d == day]
        if len(sess) < 30:
            continue
        o15 = sess.iloc[:3]
        if len(o15) < 3:
            continue

        gh = gaps.iloc[max(0, k - GAP_WIN):k].dropna()
        thr = np.percentile(gh, GAP_PCT) if len(gh) > 4 else np.nan
        gnow = gaps.iloc[k]
        pmv20 = pmv.reindex(days).iloc[max(0, k - 20):k].mean()
        v20 = prior.v.iloc[-20:].mean()
        orX = o15.volume.sum() / (v20 * 15.0 / 390.0) if v20 > 0 else np.nan

        g1 = int(np.isfinite(gnow) and np.isfinite(thr) and gnow >= thr)
        g2 = int(pmv20 > 0 and pmv.get(day, 0) / pmv20 >= PM_MULT)
        g3 = int(sess.open.iloc[0] >= lvl)
        g4 = int(np.isfinite(orX) and orX >= OR_MULT)

        end_day = days[min(k + HOLD, len(days) - 1)]
        end = min(r_idx.searchsorted(pd.Timestamp(str(end_day) + ' 23:59', tz=r_idx.tz),
                                     side='right'), len(r_idx))

        def run(entry, t0):
            stop, tgt = entry - STOP_ATR * atr, entry + TGT_ATR * atr
            start = r_idx.searchsorted(t0, side='right')
            for i in range(start, end):
                if r_l[i] <= stop:
                    return 0
                if r_h[i] >= tgt:
                    return 1
            return 1 if (end > start and r_c[end - 1] > entry) else 0

        rows.append(dict(
            sym=sym, day=day, g1=g1, g2=g2, g3=g3, g4=g4, score=g1 + g2 + g3 + g4,
            win_0945=run(float(o15.close.iloc[-1]), o15.index[-1]),
            win_open=run(float(sess.open.iloc[0]), sess.index[0] - pd.Timedelta(seconds=1)),
        ))

T = pd.DataFrame(rows)
T['y'] = pd.to_datetime(T.day).dt.year
T.to_csv('research/smc/four_gate.csv', index=False)

rng = np.random.default_rng(2024)
def ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return '(n<10)'
    b = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return f'[{np.percentile(b,2.5):.0%}, {np.percentile(b,97.5):.0%}]'

YRS, NT = 4.65, T.sym.nunique()
base = T.win_0945.values.astype(float)
print('=' * 98)
print(f'4-gate score, extended hours, fillable entry at 09:45. n={len(T)} sessions, {NT} tickers.')
print(f'Baseline (every session): {base.mean():.0%}  {ci(base)}')
print('=' * 98)
print(f"{'score':<10}{'n':>7}{'/tkr/yr':>9}{'win':>7}{'95% CI':>18}{'vs base':>10}{'diff CI':>18}")
for s in range(5):
    g = T[T.score == s]
    x = g.win_0945.values.astype(float)
    if len(x) < 5:
        continue
    d = rng.choice(x, (10000, len(x)), replace=True).mean(1) - rng.choice(base, (10000, len(base)), replace=True).mean(1)
    print(f"{str(s)+' of 4':<10}{len(x):>7}{len(x)/YRS/NT:>9.1f}{x.mean():>7.0%}{ci(x):>18}"
          f"{x.mean()-base.mean():>+10.1%}   [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]")

print('\n' + '=' * 98)
print('Gates complete BEFORE the bell (g1+g2) -- the only thing extended hours can trade early')
print('=' * 98)
print(f"{'selection':<44}{'n':>7}{'entry':>8}{'win':>7}{'95% CI':>18}")
for lbl, m, col, e in (
        ('g1+g2 both pass', (T.g1 == 1) & (T.g2 == 1), 'win_open', '09:30'),
        ('g1+g2 both pass', (T.g1 == 1) & (T.g2 == 1), 'win_0945', '09:45'),
        ('g1+g2 both pass AND opens above level', (T.g1 == 1) & (T.g2 == 1) & (T.g3 == 1), 'win_open', '09:30'),
        ('all four gates', T.score == 4, 'win_open', '09:30'),
        ('all four gates', T.score == 4, 'win_0945', '09:45')):
    g = T[m]
    x = g[col].values.astype(float)
    if len(x) < 10:
        continue
    print(f"{lbl:<44}{len(x):>7}{e:>8}{x.mean():>7.0%}{ci(x):>18}")

print('\n' + '=' * 98)
print('4/4 broken out by ticker and year (entry 09:45)')
print('=' * 98)
f4 = T[T.score == 4]
print(f4.groupby('sym').agg(n=('win_0945', 'size'), win=('win_0945', 'mean')).round(2).sort_values('n', ascending=False).to_string())
print()
print(f4.groupby('y').agg(n=('win_0945', 'size'), win=('win_0945', 'mean')).round(2).to_string())
