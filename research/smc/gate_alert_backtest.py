"""
Backtest the deployed alert exactly as it fires.

The alert is Strat v5 Companion's "Breakout gate 2/2": at the close of the
session's first 15-minute bar, both of these are true --
    gate 3  today's RTH open >= the highest high of the last 10 RTH sessions
    gate 4  09:30-09:45 volume >= 3.6x (20-day avg daily volume * 15/390)

Three entry models, because this matters more than anything else here:
    level   buy at the 10-day high            <- the original study's assumption
    open    buy at the session open
    0945    buy at the 09:45 close            <- what the alert can actually give you

Gate 3 means the stock OPENED above the level, so "buy at the level" is a price
that never traded. Any result from that model is fiction for this subset.

Extended-hours bars are loaded so the 4-gate version (adding the pre-market gap
and pre-market volume gates) can be measured on the same events.
"""
import glob
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
        print('MISSING', sym)
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

    r_h = rth.high.values
    r_l = rth.low.values
    r_c = rth.close.values
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

        v20 = prior.v.iloc[-20:].mean()
        orX = o15.volume.sum() / (v20 * 15.0 / 390.0) if v20 > 0 else np.nan
        g3 = sess.open.iloc[0] >= lvl
        g4 = np.isfinite(orX) and orX >= OR_MULT
        if not (g3 and g4):
            continue                                   # the alert did not fire

        # pre-market gates, for the 4-gate comparison
        gh = gaps.iloc[max(0, k - GAP_WIN):k].dropna()
        thr = np.percentile(gh, GAP_PCT) if len(gh) > 4 else np.nan
        gnow = gaps.iloc[k]
        pmv20 = pmv.reindex(days).iloc[max(0, k - 20):k].mean()
        g1 = bool(np.isfinite(gnow) and np.isfinite(thr) and gnow >= thr)
        g2 = bool(pmv20 > 0 and pmv.get(day, 0) / pmv20 >= PM_MULT)

        entries = {'level': lvl,
                   'open': float(sess.open.iloc[0]),
                   '0945': float(o15.close.iloc[-1])}
        t0 = o15.index[-1]
        start = r_idx.searchsorted(t0, side='right')
        end_day = days[min(k + HOLD, len(days) - 1)]
        end = r_idx.searchsorted(pd.Timestamp(str(end_day) + ' 23:59', tz=r_idx.tz), side='right')

        rec = dict(sym=sym, day=day, lvl=lvl, atr=atr, orX=orX,
                   g1=int(g1), g2=int(g2), gap=gnow,
                   pmX=(pmv.get(day, 0) / pmv20) if pmv20 > 0 else np.nan,
                   gap_pct_of_atr=(sess.open.iloc[0] - lvl) / atr)
        for name, entry in entries.items():
            stop = entry - STOP_ATR * atr
            tgt = entry + TGT_ATR * atr
            win = np.nan
            for i in range(start, min(end, len(r_idx))):
                if r_l[i] <= stop:
                    win = 0
                    break
                if r_h[i] >= tgt:
                    win = 1
                    break
            if np.isnan(win):
                last = r_c[min(end, len(r_idx)) - 1]
                win = 1 if last > entry else 0
            rec['win_' + name] = win
            rec['entry_' + name] = entry
        rows.append(rec)

T = pd.DataFrame(rows)
T['y'] = pd.to_datetime(T.day).dt.year
T.to_csv('research/smc/gate_alert.csv', index=False)

rng = np.random.default_rng(101)
def ci(x):
    if len(x) < 10:
        return '(n<10)'
    b = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return f'[{np.percentile(b,2.5):.0%}, {np.percentile(b,97.5):.0%}]'

YRS = 4.65
print('=' * 96)
print(f'"Breakout gate 2/2" fires: n={len(T)} across {T.sym.nunique()} tickers '
      f'= {len(T)/YRS/T.sym.nunique():.1f} per ticker per year')
print('=' * 96)
print(f"{'entry model':<38}{'win':>7}{'95% CI':>18}   note")
for name, note in (('level', 'price never traded when it gaps above'),
                   ('open',  'fillable, but worst price of the three'),
                   ('0945',  'what the alert actually gives you')):
    x = T['win_' + name].values.astype(float)
    print(f"{name:<38}{x.mean():>7.0%}{ci(x):>18}   {note}")

print('\n' + '=' * 96)
print('Does adding the two pre-market gates (extended hours) help? Entry = 09:45')
print('=' * 96)
print(f"{'gate set':<38}{'n':>6}{'/tkr/yr':>9}{'win':>7}{'95% CI':>18}")
for lbl, m in (('2/2 (gates 3+4, RTH only)', T.index == T.index),
               ('3 of 4 (+ either premkt gate)', (T.g1 + T.g2) >= 1),
               ('4 of 4 (both premkt gates too)', (T.g1 + T.g2) == 2),
               ('2/2 but BOTH premkt gates fail', (T.g1 + T.g2) == 0)):
    g = T[m]
    if len(g) < 10:
        continue
    x = g.win_0945.values.astype(float)
    print(f"{lbl:<38}{len(g):>6}{len(g)/YRS/T.sym.nunique():>9.1f}{x.mean():>7.0%}{ci(x):>18}")

print('\n' + '=' * 96)
print('Per ticker and per year (entry = 09:45)')
print('=' * 96)
pt = T.groupby('sym').agg(n=('win_0945', 'size'), win=('win_0945', 'mean')).round(2)
print(pt.sort_values('n', ascending=False).to_string())
print()
print(T.groupby('y').agg(n=('win_0945', 'size'), win=('win_0945', 'mean')).round(2).to_string())

print('\n' + '=' * 96)
print('How far above the level does it open? (why the "level" entry is fiction)')
print('=' * 96)
q = T.gap_pct_of_atr.quantile([.1, .25, .5, .75, .9]).round(2)
print('open minus level, in ATR:  p10/p25/p50/p75/p90 =', q.tolist())
print(f'opens more than 0.25 ATR above the level: {(T.gap_pct_of_atr > 0.25).mean():.0%} of fires')
