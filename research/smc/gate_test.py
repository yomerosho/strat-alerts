"""
Test every candidate gate on the 1,372-breakout table before any of it goes
into an indicator. A gate earns its place only if it moves the win rate and
survives a bootstrap against the base rate.
"""
import glob
import os
import numpy as np
import pandas as pd

LOOKBACK = 10
STOP_ATR, TGT_ATR, HOLD = 1.0, 1.5, 5

# ---- market context: SPY daily ----
spy5 = pd.read_parquet('research/.lab_cache/SPY_5Min_2022-01-01.parquet')
spy5 = spy5.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
spy5 = spy5.set_index('ts').sort_index()
spyd = spy5.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                'close': 'last', 'volume': 'sum'}).dropna()
spyd.index = spyd.index.date
spyd['sma20'] = spyd.close.rolling(20).mean()
spyd['regime'] = spyd.close > spyd.sma20
spyd['ret5'] = spyd.close.pct_change(5)


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def macd(s):
    e12 = s.ewm(span=12, adjust=False).mean()
    e26 = s.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


rows = []
for path in sorted(glob.glob('research/smc/data/*_5m_ext.parquet')):
    sym = os.path.basename(path).split('_')[0]
    d5 = pd.read_parquet(path)
    d5['d'] = d5.index.date
    rth = d5.between_time('09:30', '15:59')
    pm = d5.between_time('04:00', '09:29')
    daily = rth.groupby('d').agg(o=('open', 'first'), h=('high', 'max'),
                                 l=('low', 'min'), c=('close', 'last'),
                                 v=('volume', 'sum'))
    daily = daily[daily.v > 0]
    pmv, pmh = pm.groupby('d').volume.sum(), pm.groupby('d').high.max()
    pmc = pm.groupby('d').close.last()
    pmv20 = pmv.rolling(20).mean()

    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(),
                    (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    daily['atr14'] = tr.rolling(14).mean()
    daily['v20'] = daily.v.rolling(20).mean()
    daily['rsi14'] = rsi(daily.c)
    ml, ms, mh = macd(daily.c)
    daily['macd_h'] = mh
    daily['macd_up'] = mh > mh.shift(1)
    daily['macd_pos'] = ml > ms
    daily['ret5'] = daily.c.pct_change(5)
    daily['ret20'] = daily.c.pct_change(20)
    daily['hi52'] = daily.h.rolling(252, min_periods=60).max()

    # hourly RSI for the 1H view
    h1 = rth.resample('60min', offset='30min', label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    h1['rsi'] = rsi(h1.close)

    days = list(daily.index)
    for k in range(60, len(days) - 1):
        day = days[k]
        prior = daily.iloc[:k]
        lvl = prior.h.iloc[-LOOKBACK:].max()
        pc, atr = prior.c.iloc[-1], prior.atr14.iloc[-1]
        if not np.isfinite(atr) or atr <= 0 or lvl <= pc:
            continue
        sess = rth[rth.d == day]
        if len(sess) < 30:
            continue
        cross = sess[sess.high >= lvl]
        if len(cross) == 0:
            continue
        t0 = cross.index[0]
        entry, stop, tgt = lvl, lvl - STOP_ATR * atr, lvl + TGT_ATR * atr
        fwd = rth[(rth.index > t0) & (rth.d <= days[min(k + HOLD, len(days) - 1)])]
        win = np.nan
        for _, b in fwd.iterrows():
            if b.low <= stop:
                win = 0
                break
            if b.high >= tgt:
                win = 1
                break
        if np.isnan(win):
            win = 1 if (len(fwd) and fwd.close.iloc[-1] > entry) else 0

        o15 = sess.iloc[:3]
        vwap_num = (sess.close * sess.volume).cumsum()
        vwap = (vwap_num / sess.volume.cumsum())
        vwap15 = vwap.iloc[2] if len(vwap) > 2 else np.nan

        # 1H RSI at the last completed hour before the cross
        h1p = h1[h1.index < t0]
        rsi1h = h1p.rsi.iloc[-1] if len(h1p) else np.nan

        sp = spyd[spyd.index < day]
        rows.append(dict(
            sym=sym, day=day, win=win, atr=atr,
            # --- the three that already earned a place ---
            gap=(pmc.get(day, np.nan) / pc - 1) * 100,
            pm_vol_x=pmv.get(day, 0) / pmv20.get(day, np.nan) if pmv20.get(day, 0) else np.nan,
            pm_above=int(pmh.get(day, -np.inf) >= lvl),
            open_above=int(sess.open.iloc[0] >= lvl),
            or15_vol_x=o15.volume.sum() / (prior.v.iloc[-20:].mean() * 3 / 78),
            or15_pos=((o15.close.iloc[-1] - o15.low.min()) /
                      (o15.high.max() - o15.low.min())) if o15.high.max() > o15.low.min() else .5,
            # --- candidates under test ---
            rsi14=prior.rsi14.iloc[-1],
            rsi1h=rsi1h,
            macd_pos=int(bool(prior.macd_pos.iloc[-1])),
            macd_rising=int(bool(prior.macd_up.iloc[-1])),
            rs5=(prior.ret5.iloc[-1] - sp.ret5.iloc[-1]) * 100 if len(sp) else np.nan,
            mkt_regime=int(bool(sp.regime.iloc[-1])) if len(sp) else 0,
            pct_of_52w=prior.c.iloc[-1] / prior.hi52.iloc[-1] * 100,
            above_vwap15=int(o15.close.iloc[-1] > vwap15) if np.isfinite(vwap15) else 0,
            lvl_touches=int((prior.h.iloc[-LOOKBACK:] >= lvl * 0.995).sum()),
            dow=pd.Timestamp(day).dayofweek,
        ))

T = pd.DataFrame(rows).dropna(subset=['gap', 'pm_vol_x'])
T.to_csv('research/smc/gates.csv', index=False)
BASE = T.win.mean()
rng = np.random.default_rng(31)
base_arr = T.win.values.astype(float)


def verdict(mask, label, minn=40):
    g = T[mask]
    if len(g) < minn:
        print(f"{label:<44}{len(g):>6}   too few")
        return
    a = g.win.values.astype(float)
    d = (rng.choice(a, (8000, len(a)), replace=True).mean(1)
         - rng.choice(base_arr, (8000, len(base_arr)), replace=True).mean(1))
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    mark = "KEEP" if lo > 0 else ("drop" if hi < 0 else "  --")
    print(f"{label:<44}{len(g):>6}{a.mean():>8.0%}{a.mean()-BASE:>+9.1%}"
          f"   [{lo:+.1%}, {hi:+.1%}]  {mark}")


print("=" * 104)
print(f"n={len(T)}   base win rate {BASE:.1%}   (a gate must beat this with a CI clear of zero)")
print("=" * 104)
print(f"{'gate':<44}{'n':>6}{'win':>8}{'vs base':>9}   {'95% CI of the lift':<22}verdict")
print("-" * 104)

print("ALREADY EARNED THEIR PLACE")
verdict(T.gap > T.gap.quantile(2 / 3), "  gap in top third")
verdict(T.pm_vol_x > 1.75, "  premarket volume > 1.75x")
verdict(T.pm_above == 1, "  premarket already above the level")
verdict(T.open_above == 1, "  opens above the level")
verdict(T.or15_vol_x > 3.6, "  first-15m volume > 3.6x")
verdict(T.or15_pos > 0.6, "  15m closes in top 40% of range")

print("\nYOUR PROPOSED ADDITIONS")
verdict(T.rsi14 > 60, "  daily RSI > 60")
verdict(T.rsi14 > 70, "  daily RSI > 70")
verdict(T.rsi14.between(40, 60), "  daily RSI 40-60 (not extended)")
verdict(T.rsi1h > 60, "  1H RSI > 60")
verdict(T.macd_pos == 1, "  MACD line above signal")
verdict(T.macd_rising == 1, "  MACD histogram rising")
verdict((T.macd_pos == 1) & (T.macd_rising == 1), "  MACD positive AND rising")

print("\nOTHER CANDIDATES I'D TEST")
verdict(T.above_vwap15 == 1, "  above VWAP at 09:45")
verdict(T.mkt_regime == 1, "  SPY above its 20-day SMA")
verdict(T.rs5 > 0, "  outperforming SPY over 5 days")
verdict(T.rs5 > T.rs5.quantile(2 / 3), "  rel. strength vs SPY, top third")
verdict(T.pct_of_52w > 95, "  within 5% of the 52-week high")
verdict(T.pct_of_52w < 80, "  more than 20% below 52w high")
verdict(T.lvl_touches >= 3, "  level tested 3+ times in the base")
verdict(T.lvl_touches <= 1, "  level tested only once")
for dw, nm in [(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'), (4, 'Friday')]:
    verdict(T.dow == dw, f"  {nm}")
