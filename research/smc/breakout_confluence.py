"""
"What pre-market confluence would have justified taking the breakout?"

Event: price crosses above the prior 10-session high intraday -- the kind of
level you can mark the night before, and the kind of alert the user had at 352.

For every such event on 6 high-beta names since 2022, record everything that was
observable BEFORE 09:30, plus a few things observable in the first 15 minutes,
then see which of them actually separated the breakouts that ran from the ones
that failed.

Trade model: enter at the level, volatility stop at -1.0 ATR14(daily),
target +1.5 ATR. Win = target reached before stop within 5 sessions.
"""
import glob
import os
import numpy as np
import pandas as pd

LOOKBACK = 10          # sessions defining the breakout level
STOP_ATR = 1.0
TGT_ATR = 1.5
HOLD = 5               # sessions

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
    pmv = pm.groupby('d').volume.sum()
    pmh = pm.groupby('d').high.max()
    pml = pm.groupby('d').low.min()
    pmc = pm.groupby('d').close.last()

    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(),
                    (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    daily['atr14'] = tr.rolling(14).mean()
    daily['atr5'] = tr.rolling(5).mean()
    daily['sma20'] = daily.c.rolling(20).mean()
    daily['sma50'] = daily.c.rolling(50).mean()
    daily['v20'] = daily.v.rolling(20).mean()
    pmv20 = pmv.rolling(20).mean()

    days = list(daily.index)
    for k in range(60, len(days) - 1):
        day = days[k]
        prior = daily.iloc[:k]
        lvl = prior.h.iloc[-LOOKBACK:].max()
        pc = prior.c.iloc[-1]
        atr = prior.atr14.iloc[-1]
        if not np.isfinite(atr) or atr <= 0 or lvl <= pc:
            continue                              # already above the level = no setup

        sess = rth[rth.d == day]
        if len(sess) < 30:
            continue
        cross = sess[sess.high >= lvl]
        if len(cross) == 0:
            continue                              # never triggered
        t0 = cross.index[0]

        entry = lvl
        stop = entry - STOP_ATR * atr
        tgt = entry + TGT_ATR * atr
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
            last = fwd.close.iloc[-1] if len(fwd) else entry
            win = 1 if last > entry else 0
        mfe = (fwd.high.max() - entry) / atr if len(fwd) else 0
        mae = (fwd.low.min() - entry) / atr if len(fwd) else 0

        # ---- observable before 09:30 ----
        pmv_x = pmv.get(day, 0) / pmv20.get(day, np.nan) if pmv20.get(day, 0) else np.nan
        pm_last = pmc.get(day, np.nan)
        base_hi = prior.h.iloc[-LOOKBACK:].max()
        base_lo = prior.l.iloc[-LOOKBACK:].min()
        hi_pos = LOOKBACK - int(np.argmax(prior.h.iloc[-LOOKBACK:].values)) - 1

        # ---- observable by 09:45 ----
        o15 = sess.iloc[:3]
        or15_vol_x = o15.volume.sum() / (prior.v.iloc[-20:].mean() * 3 / 78)
        or15_pos = ((o15.close.iloc[-1] - o15.low.min()) /
                    (o15.high.max() - o15.low.min())) if o15.high.max() > o15.low.min() else .5

        rows.append(dict(
            sym=sym, day=day, lvl=lvl, atr=atr, win=win, mfe=mfe, mae=mae,
            trig_time=t0.strftime('%H:%M'),
            gap=(pm_last / pc - 1) * 100 if np.isfinite(pm_last) else np.nan,
            pm_vol_x=pmv_x,
            pm_above=int(pmh.get(day, -np.inf) >= lvl),
            pm_range=(pmh.get(day, np.nan) - pml.get(day, np.nan)) / atr,
            dist_to_lvl=(lvl / pc - 1) * 100,
            base_tight=(base_hi - base_lo) / pc * 100,
            atr_contract=prior.atr5.iloc[-1] / atr,
            vs_sma20=(pc / prior.sma20.iloc[-1] - 1) * 100,
            vs_sma50=(pc / prior.sma50.iloc[-1] - 1) * 100,
            sma20_slope=(prior.sma20.iloc[-1] / prior.sma20.iloc[-6] - 1) * 100,
            days_since_high=hi_pos,
            prior_rvol=prior.v.iloc[-1] / prior.v20.iloc[-1],
            up_days=int(sum(1 for j in range(1, 6)
                            if prior.c.iloc[-j] > prior.c.iloc[-j - 1]
                            and all(prior.c.iloc[-m] > prior.c.iloc[-m - 1] for m in range(1, j)))),
            or15_vol_x=or15_vol_x, or15_pos=or15_pos,
            open_above=int(sess.open.iloc[0] >= lvl),
        ))

T = pd.DataFrame(rows).dropna(subset=['gap', 'pm_vol_x'])
T.to_csv('research/smc/breakouts.csv', index=False)

print("=" * 104)
print(f"{LOOKBACK}-day-high breakouts, 2022-2026, {T.sym.nunique()} names   n={len(T)}   "
      f"base win rate {T.win.mean():.1%}   (target +{TGT_ATR} ATR before -{STOP_ATR} ATR)")
print("=" * 104)
print(T.groupby('sym').agg(n=('win', 'size'), win=('win', 'mean'),
                           medMFE=('mfe', 'median')).round(3).to_string())

PRE = ['gap', 'pm_vol_x', 'pm_range', 'dist_to_lvl', 'base_tight', 'atr_contract',
       'vs_sma20', 'vs_sma50', 'sma20_slope', 'days_since_high', 'prior_rvol', 'up_days']
POST = ['or15_vol_x', 'or15_pos']

def terciles(col, label, tag):
    g = T.dropna(subset=[col])
    if g[col].nunique() < 4:
        return
    try:
        g = g.assign(b=pd.qcut(g[col], 3, labels=['low', 'mid', 'high'], duplicates='drop'))
    except ValueError:
        return
    s = g.groupby('b', observed=True).agg(n=('win', 'size'), win=('win', 'mean'),
                                          mfe=('mfe', 'median'))
    if len(s) < 3:
        return
    lo, hi = s.win.iloc[0], s.win.iloc[-1]
    rng = g.groupby('b', observed=True)[col].median()
    print(f"{tag}{label:<16}"
          + "".join(f"{s.win.iloc[i]:>7.0%}" for i in range(3))
          + f"{hi-lo:>+9.0%}   "
          + "  ".join(f"{rng.iloc[i]:.2f}" for i in range(3)))

print("\n" + "=" * 104)
print("Win rate by tercile of each feature   (low / mid / high, then the spread)")
print("=" * 104)
print(f"{'  feature':<18}{'low':>7}{'mid':>7}{'high':>7}{'spread':>9}   median value per tercile")
print("-" * 104)
print("KNOWABLE BEFORE 09:30")
for c in PRE:
    terciles(c, c, "  ")
print("KNOWABLE BY 09:45")
for c in POST:
    terciles(c, c, "  ")

print("\n" + "=" * 104)
print("Binary flags")
print("=" * 104)
for c in ['pm_above', 'open_above']:
    for v in (0, 1):
        g = T[T[c] == v]
        if len(g) > 20:
            print(f"  {c} = {v}   n={len(g):>4}   win {g.win.mean():.1%}   medMFE {g.mfe.median():.2f} ATR")
