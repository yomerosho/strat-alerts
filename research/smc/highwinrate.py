"""
Is there a 70%-win-rate strategy?

Two things are demonstrated here, and they are different questions.

PART 1  Win rate is largely a CHOICE. With a fixed stop, shrinking the target
        raises the win rate mechanically. Break-even for a target of m R is
        1/(1+m): a 0.43R target needs 70% just to break even. So "70% win rate"
        on its own says nothing about whether a strategy makes money.

PART 2  A genuine high-win-rate family: RSI(2) mean reversion (Connors-style).
        Buy a pullback in an uptrend, exit on reversion. This really does win
        ~70% of the time. The test is whether it beats simply holding.

Daily bars 2016-2026, split-adjusted, 15 tickers. Entry and exit at the NEXT
open after the signal -- fillable, no same-bar-close cheating.
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('research/smc/data/daily_2015.parquet')
df['date'] = pd.to_datetime(df['timestamp']).dt.date
UNIVERSE = sorted(df.symbol.unique())


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run(rsi_th, exit_rule, stop_pct=None, trend_filter=True, hold_cap=10):
    trades = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        if len(g) < 300:
            continue
        c, o, h, l = g.close, g.open, g.high, g.low
        g['rsi2'] = rsi(c, 2)
        g['sma5'] = c.rolling(5).mean()
        g['sma200'] = c.rolling(200).mean()
        sig = (g.rsi2 < rsi_th)
        if trend_filter:
            sig &= (c > g.sma200)
        sig = sig.fillna(False).values
        cv, ov, hv, lv = c.values, o.values, h.values, l.values
        s5 = g.sma5.values
        n = len(g)
        i = 0
        while i < n - 2:
            if not sig[i]:
                i += 1
                continue
            e = i + 1                       # buy next open
            entry = ov[e]
            if not np.isfinite(entry) or entry <= 0:
                i += 1
                continue
            stop = entry * (1 - stop_pct) if stop_pct else None
            ret, held, why = None, 0, ''
            for k in range(e, min(e + hold_cap + 1, n - 1)):
                held = k - e + 1
                if stop and lv[k] <= stop:
                    ret = -stop_pct
                    why = 'stop'
                    break
                done = (cv[k] > s5[k]) if exit_rule == 'sma5' else (cv[k] > cv[k - 1])
                if done and k > e - 1:
                    ret = ov[k + 1] / entry - 1     # exit next open
                    why = exit_rule
                    break
            if ret is None:
                kk = min(e + hold_cap, n - 1)
                ret = ov[kk] / entry - 1
                why = 'time'
                held = kk - e
            trades.append(dict(sym=sym, date=g.timestamp[e].date(), ret=ret,
                               held=held, why=why, year=g.timestamp[e].year))
            i = e + max(held, 1)
    return pd.DataFrame(trades)


rng = np.random.default_rng(11)
def ci(x):
    x = np.asarray(x, float)
    if len(x) < 20:
        return '(n<20)'
    b = rng.choice(x, (10000, len(x)), replace=True).mean(axis=1)
    return f'[{np.percentile(b,2.5)*100:+.2f}%, {np.percentile(b,97.5)*100:+.2f}%]'


print('=' * 108)
print('PART 1  break-even win rate is set by your target, not by your skill')
print('=' * 108)
print(f"{'target (stop = 1R)':<24}{'break-even win rate needed':>30}")
for m in (0.25, 0.43, 0.5, 1.0, 1.5, 2.0, 3.0):
    print(f"{str(m) + 'R':<24}{1/(1+m):>29.0%}")
print('\nA 70% win rate is break-even at a 0.43R target. Below that it loses money.')

print('\n' + '=' * 108)
print('PART 2  RSI(2) mean reversion -- daily, 15 tickers, 2016-2026, entry+exit at next open')
print('=' * 108)
print(f"{'variant':<42}{'n':>6}{'win':>7}{'avgW':>8}{'avgL':>8}{'exp/trade':>11}{'  95% CI':<22}{'PF':>6}{'worst':>8}")

results = {}
for th in (5, 10, 15):
    for stop in (None, 0.05):
        t = run(th, 'sma5', stop_pct=stop, trend_filter=True)
        if len(t) < 30:
            continue
        w = t[t.ret > 0].ret
        lo = t[t.ret <= 0].ret
        pf = (w.sum() / abs(lo.sum())) if len(lo) and lo.sum() != 0 else np.inf
        lbl = f'RSI2<{th}, >200SMA' + (', 5% stop' if stop else ', no stop')
        results[lbl] = t
        print(f"{lbl:<42}{len(t):>6}{(t.ret>0).mean():>7.0%}{w.mean()*100:>7.2f}%"
              f"{lo.mean()*100:>7.2f}%{t.ret.mean()*100:>10.2f}%{ci(t.ret.values):<22}"
              f"{pf:>6.2f}{t.ret.min()*100:>7.1f}%")

t = run(10, 'sma5', stop_pct=None, trend_filter=False)
w = t[t.ret > 0].ret; lo = t[t.ret <= 0].ret
print(f"{'RSI2<10, NO trend filter':<42}{len(t):>6}{(t.ret>0).mean():>7.0%}{w.mean()*100:>7.2f}%"
      f"{lo.mean()*100:>7.2f}%{t.ret.mean()*100:>10.2f}%{ci(t.ret.values):<22}"
      f"{(w.sum()/abs(lo.sum())):>6.2f}{t.ret.min()*100:>7.1f}%")

BEST = 'RSI2<10, >200SMA, no stop'
b = results.get(BEST)
if b is not None:
    print('\n' + '=' * 108)
    print(f'CONTROL -- does "{BEST}" beat simply holding for the same number of days?')
    print('=' * 108)
    hold = int(round(b.held.mean()))
    ctrl = []
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        ov = g.open.values
        for i in range(200, len(g) - hold - 2):
            ctrl.append(ov[i + hold] / ov[i] - 1)
    ctrl = np.array(ctrl)
    print(f"strategy: n={len(b)}  win {(b.ret>0).mean():.0%}  mean {b.ret.mean()*100:+.2f}%  "
          f"median hold {b.held.median():.0f} days")
    print(f"hold anything, same {hold} days: n={len(ctrl)}  win {(ctrl>0).mean():.0%}  mean {ctrl.mean()*100:+.2f}%")
    d = (rng.choice(b.ret.values, (10000, len(b)), replace=True).mean(1)
         - rng.choice(ctrl, (10000, min(len(ctrl), 5000)), replace=True).mean(1))
    print(f"difference: {b.ret.mean()*100 - ctrl.mean()*100:+.2f}%  "
          f"95% CI [{np.percentile(d,2.5)*100:+.2f}%, {np.percentile(d,97.5)*100:+.2f}%]  "
          f"P(strategy better) = {(d>0).mean():.2f}")

    print('\nBy year:')
    print(b.groupby('year').agg(n=('ret', 'size'), win=('ret', lambda x: (x > 0).mean()),
                                mean=('ret', 'mean')).round(3).to_string())
    print('\nBy ticker:')
    print(b.groupby('sym').agg(n=('ret', 'size'), win=('ret', lambda x: (x > 0).mean()),
                               mean=('ret', 'mean')).round(3).sort_values('n', ascending=False).to_string())
