"""How long does an RSI(2) reversion trade actually take?"""
import numpy as np, pandas as pd

df = pd.read_parquet('research/smc/data/daily_2015.parquet')
df['ts'] = pd.to_datetime(df['timestamp'])

def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

RSI_TH, HOLD_CAP = 10, 10
trades = []
for sym, g in df.groupby('symbol'):
    g = g.sort_values('ts').reset_index(drop=True)
    if len(g) < 300: continue
    g['rsi2'] = rsi(g.close, 2)
    g['sma5'] = g.close.rolling(5).mean()
    g['sma200'] = g.close.rolling(200).mean()
    sig = ((g.rsi2 < RSI_TH) & (g.close > g.sma200)).fillna(False).values
    ov, cv, s5 = g.open.values, g.close.values, g.sma5.values
    ts = g.ts.values
    n = len(g); i = 0
    while i < n - 2:
        if not sig[i]:
            i += 1; continue
        e = i + 1
        entry = ov[e]
        if not np.isfinite(entry) or entry <= 0:
            i += 1; continue
        exit_bar, why = None, 'time'
        for k in range(e, min(e + HOLD_CAP + 1, n - 1)):
            if cv[k] > s5[k]:
                exit_bar, why = k + 1, 'sma5'
                break
        if exit_bar is None:
            exit_bar = min(e + HOLD_CAP, n - 1)
        ret = ov[exit_bar] / entry - 1
        tdays = exit_bar - e                       # trading days held
        cdays = (pd.Timestamp(ts[exit_bar]) - pd.Timestamp(ts[e])).days
        trades.append(dict(sym=sym, entry_date=pd.Timestamp(ts[e]).date(), ret=ret,
                           tdays=tdays, cdays=cdays, why=why, win=int(ret > 0),
                           year=pd.Timestamp(ts[e]).year))
        i = exit_bar
T = pd.DataFrame(trades)

print('='*92); print(f'RSI(2) reversion  n={len(T)}  win {T.win.mean():.0%}  mean {T.ret.mean()*100:+.2f}%'); print('='*92)
print(f"Trading days held :  mean {T.tdays.mean():.1f}   median {T.tdays.median():.0f}   "
      f"p90 {T.tdays.quantile(.9):.0f}   max {T.tdays.max():.0f}")
print(f"Calendar days     :  mean {T.cdays.mean():.1f}   median {T.cdays.median():.0f}   p90 {T.cdays.quantile(.9):.0f}")

print('\nHow trades resolve, by trading day held:')
print(f"{'days':<7}{'n':>7}{'share':>8}{'cumul':>8}{'win':>7}{'avg ret':>10}")
tot = len(T); cum = 0
for d in sorted(T.tdays.unique()):
    g = T[T.tdays == d]; cum += len(g)
    print(f"{d:<7}{len(g):>7}{len(g)/tot:>8.0%}{cum/tot:>8.0%}{g.win.mean():>7.0%}{g.ret.mean()*100:>9.2f}%")

print('\nWinners vs losers:')
w, l = T[T.win == 1], T[T.win == 0]
print(f"  winners  n={len(w):<5} mean hold {w.tdays.mean():.1f} trading days ({w.cdays.mean():.1f} calendar)  avg {w.ret.mean()*100:+.2f}%")
print(f"  losers   n={len(l):<5} mean hold {l.tdays.mean():.1f} trading days ({l.cdays.mean():.1f} calendar)  avg {l.ret.mean()*100:+.2f}%")

print('\nExit reason:')
for why, g in T.groupby('why'):
    print(f"  {why:<6} n={len(g):<5} {len(g)/tot:>4.0%}  win {g.win.mean():>4.0%}  avg {g.ret.mean()*100:+6.2f}%  mean hold {g.tdays.mean():.1f}d")

print('\nDoes a slow trade mean a bad trade?')
for lo, hi, lbl in ((1,1,'1 day'), (2,2,'2 days'), (3,4,'3-4 days'), (5,7,'5-7 days'), (8,99,'8+ days')):
    g = T[(T.tdays >= lo) & (T.tdays <= hi)]
    if len(g) < 15: continue
    print(f"  {lbl:<10} n={len(g):<5} win {g.win.mean():>4.0%}  avg {g.ret.mean()*100:+6.2f}%")

print(f"\nTrade frequency: {len(T)/15/10.6:.1f} per ticker per year across 15 tickers "
      f"= about {len(T)/10.6:.0f} trades a year total, ~{len(T)/10.6/52:.1f} a week.")
