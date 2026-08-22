"""
Controls for the 09:45-entry result. Same 15 tickers, same stop/target geometry
(-1.0 ATR / +1.5 ATR, 5 sessions), only the selection rule changes -- so any
difference is the gate, not the exit model.

  A  any session at all                     pure baseline
  B  opens above the 10-day high (gate 3)   gate 3 alone
  C  gate 3 + gate 4                        the deployed alert
  D  crosses the level intraday but does NOT open above it, entry AT the level
     -- the one case where the level entry is genuinely fillable
"""
import os
import numpy as np
import pandas as pd

LVL_N, STOP_ATR, TGT_ATR, HOLD, OR_MULT = 10, 1.0, 1.5, 5, 3.6
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
    daily = rth.groupby('d').agg(o=('open', 'first'), h=('high', 'max'),
                                 l=('low', 'min'), c=('close', 'last'),
                                 v=('volume', 'sum'))
    daily = daily[daily.v > 0]
    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(),
                    (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    daily['atr'] = tr.rolling(14).mean()
    days = list(daily.index)
    r_h, r_l, r_c = rth.high.values, rth.low.values, rth.close.values
    r_idx = rth.index

    def run(entry, t0, k):
        stop, tgt = entry - STOP_ATR * atr, entry + TGT_ATR * atr
        start = r_idx.searchsorted(t0, side='right')
        end_day = days[min(k + HOLD, len(days) - 1)]
        end = min(r_idx.searchsorted(pd.Timestamp(str(end_day) + ' 23:59', tz=r_idx.tz), side='right'), len(r_idx))
        for i in range(start, end):
            if r_l[i] <= stop:
                return 0
            if r_h[i] >= tgt:
                return 1
        return 1 if (end > start and r_c[end - 1] > entry) else 0

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
        g3 = bool(sess.open.iloc[0] >= lvl)
        g4 = bool(np.isfinite(orX) and orX >= OR_MULT)
        t45 = o15.index[-1]

        rec = dict(sym=sym, day=day, g3=int(g3), g4=int(g4),
                   win_0945=run(float(o15.close.iloc[-1]), t45, k))

        # D: no gap above, level actually reachable intraday
        if not g3 and sess.high.max() >= lvl:
            cross = sess[sess.high >= lvl]
            rec['win_level_fillable'] = run(float(lvl), cross.index[0], k)
            rec['crossed'] = 1
        else:
            rec['crossed'] = 0
        rows.append(rec)

T = pd.DataFrame(rows)
T['y'] = pd.to_datetime(T.day).dt.year
T.to_csv('research/smc/gate_controls.csv', index=False)

rng = np.random.default_rng(7)
def ci(x):
    if len(x) < 10:
        return '(n<10)'
    b = rng.choice(np.asarray(x, dtype=float), (10000, len(x)), replace=True).mean(axis=1)
    return f'[{np.percentile(b,2.5):.0%}, {np.percentile(b,97.5):.0%}]'

print('=' * 100)
print(f'Controls -- 15 tickers, 2022-2026, n={len(T)} sessions. Exit geometry identical throughout.')
print('=' * 100)
print(f"{'selection':<52}{'n':>7}{'win':>7}{'95% CI':>18}")

sets = [
    ('A  any session, entry 09:45', T, 'win_0945'),
    ('B  opens above 10-day high (gate 3), entry 09:45', T[T.g3 == 1], 'win_0945'),
    ('C  gate 3 + gate 4 = THE ALERT, entry 09:45', T[(T.g3 == 1) & (T.g4 == 1)], 'win_0945'),
    ('   gate 3, gate 4 FAILS, entry 09:45', T[(T.g3 == 1) & (T.g4 == 0)], 'win_0945'),
    ('D  no gap, crosses level intraday, entry AT level', T[T.crossed == 1], 'win_level_fillable'),
]
for lbl, g, col in sets:
    x = g[col].dropna().values.astype(float)
    if len(x) < 5:
        continue
    print(f"{lbl:<52}{len(x):>7}{x.mean():>7.0%}{ci(x):>18}")

a = T.win_0945.values.astype(float)
c = T[(T.g3 == 1) & (T.g4 == 1)].win_0945.values.astype(float)
d = rng.choice(c, (10000, len(c)), replace=True).mean(1) - rng.choice(a, (10000, len(a)), replace=True).mean(1)
print(f"\nALERT minus baseline: {c.mean()-a.mean():+.1%}  95% CI [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]"
      f"  P(alert better) = {(d>0).mean():.2f}")

b3 = T[T.g3 == 1].win_0945.values.astype(float)
d2 = rng.choice(c, (10000, len(c)), replace=True).mean(1) - rng.choice(b3, (10000, len(b3)), replace=True).mean(1)
print(f"gate 4's own contribution: {c.mean()-b3.mean():+.1%}  95% CI [{np.percentile(d2,2.5):+.1%}, {np.percentile(d2,97.5):+.1%}]")
