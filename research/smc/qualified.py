"""
The two qualifiers the earlier runs ignored:

  1. NAMED liquidity. A generic swing low is not a liquidity pool. Restrict the
     sweep to prior-day low, prior-week low, or equal-lows (2+ swing lows inside
     a 0.15% band) -- the pools that actually hold resting stops.
  2. DISPLACEMENT. The leg out of the sweep must be impulsive: the break bar
     ranges > 1.5x ATR(14), and/or the leg leaves a fair value gap.

Then re-test both entries (break close, and 0.5 retrace) on the qualified set.
"""
import numpy as np
import pandas as pd

exec(open('research/smc/smc_lab.py').read().split('# ---------- build the event table')[0])

EQ_BAND = 0.0015        # 0.15% -> "equal" lows
DISP_MULT = 1.5
VALID_BARS = 12

oo = h1['open'].values
dates = pd.Series(h1.index.date, index=range(n1))
weeks = pd.Series(h1.index.isocalendar().week.values * 100 + h1.index.year.values,
                  index=range(n1))

# ---- ATR(14) on 1H, causal
tr = np.maximum(hh - ll, np.maximum(np.abs(hh - np.roll(cc, 1)),
                                    np.abs(ll - np.roll(cc, 1))))
tr[0] = hh[0] - ll[0]
atr = pd.Series(tr).rolling(14).mean().shift(1).values

# ---- prior-day / prior-week extremes, causal
day_lo, day_hi = {}, {}
wk_lo, wk_hi = {}, {}
pdl = np.full(n1, np.nan); pdh = np.full(n1, np.nan)
pwl = np.full(n1, np.nan); pwh = np.full(n1, np.nan)
prev_d = prev_w = None
for i in range(n1):
    d, w = dates[i], weeks[i]
    if prev_d is not None and d != prev_d:
        day_lo[prev_d], day_hi[prev_d] = cur_dlo, cur_dhi
    if d != prev_d:
        cur_dlo, cur_dhi = ll[i], hh[i]
        prev_d_done = prev_d
        prev_d = d
    else:
        cur_dlo, cur_dhi = min(cur_dlo, ll[i]), max(cur_dhi, hh[i])
    if prev_w is not None and w != prev_w:
        wk_lo[prev_w], wk_hi[prev_w] = cur_wlo, cur_whi
    if w != prev_w:
        cur_wlo, cur_whi = ll[i], hh[i]
        prev_w = w
    else:
        cur_wlo, cur_whi = min(cur_wlo, ll[i]), max(cur_whi, hh[i])
    # last COMPLETED day/week
    done_days = [k for k in day_lo if k < d]
    done_wks = [k for k in wk_lo if k < w]
    if done_days:
        kd = max(done_days)
        pdl[i], pdh[i] = day_lo[kd], day_hi[kd]
    if done_wks:
        kw = max(done_wks)
        pwl[i], pwh[i] = wk_lo[kw], wk_hi[kw]

# ---- named-pool sweeps
sw_named_dn = np.zeros(n1, dtype=bool)   # bullish
sw_named_up = np.zeros(n1, dtype=bool)
sw_tag_dn = np.empty(n1, dtype=object)
sw_tag_up = np.empty(n1, dtype=object)
kpl, kph = [], []
for i in range(n1):
    j = i - L
    if j >= 0:
        if not np.isnan(pl1[j]):
            kpl.append(pl1[j])
        if not np.isnan(ph1[j]):
            kph.append(ph1[j])
    pools_dn, pools_up = [], []
    if not np.isnan(pdl[i]):
        pools_dn.append(('PDL', pdl[i]))
    if not np.isnan(pwl[i]):
        pools_dn.append(('PWL', pwl[i]))
    if not np.isnan(pdh[i]):
        pools_up.append(('PDH', pdh[i]))
    if not np.isnan(pwh[i]):
        pools_up.append(('PWH', pwh[i]))
    # equal lows / highs among the last 10 known swings
    for arr, tag, bucket in ((kpl[-10:], 'EQL', pools_dn), (kph[-10:], 'EQH', pools_up)):
        for a in range(len(arr)):
            grp = [x for x in arr if abs(x - arr[a]) / arr[a] <= EQ_BAND]
            if len(grp) >= 2:
                bucket.append((tag, min(grp) if tag == 'EQL' else max(grp)))
    for tag, lvl in pools_dn:
        if ll[i] < lvl and cc[i] > lvl:
            sw_named_dn[i] = True
            sw_tag_dn[i] = tag if sw_tag_dn[i] is None else sw_tag_dn[i] + '/' + tag
    for tag, lvl in pools_up:
        if hh[i] > lvl and cc[i] < lvl:
            sw_named_up[i] = True
            sw_tag_up[i] = tag if sw_tag_up[i] is None else sw_tag_up[i] + '/' + tag


def has_fvg(a, b, direction):
    """Bullish FVG in bars [a,b]: low[k] > high[k-2]."""
    for k in range(max(a + 2, 2), b + 1):
        if direction == 1 and ll[k] > hh[k - 2]:
            return True
        if direction == -1 and hh[k] < ll[k - 2]:
            return True
    return False


rows = []
for _, e in ev_1h.iterrows():
    i = int(e['i'])
    d = int(e['dir'])
    if i + 1 >= n1 or np.isnan(atr[i]):
        continue
    ts = e['ts']
    b4, bd, bw = (bias_at(ts, h4_close, tr_4h), bias_at(ts, d1_close, tr_1d),
                  bias_at(ts, w1_close, tr_1w))
    win = slice(max(0, i - SWEEP_LOOKBACK + 1), i + 1)
    named = bool(sw_named_dn[win].any()) if d == 1 else bool(sw_named_up[win].any())
    generic = bool(sweep_dn[win].any()) if d == 1 else bool(sweep_up[win].any())
    tags = [t for t in (sw_tag_dn if d == 1 else sw_tag_up)[win] if t]
    prot_i = int(e['prot_i']) if e['prot_i'] == e['prot_i'] and e['prot_i'] >= 0 else i
    lo_leg, hi_leg = float(ll[prot_i:i + 1].min()), float(hh[prot_i:i + 1].max())
    disp_bar = (hh[i] - ll[i]) > DISP_MULT * atr[i]
    disp_fvg = has_fvg(prot_i, i, d)
    if hi_leg <= lo_leg:
        continue

    base = dict(ts=ts, dir=d, kind=e['kind'], b4=b4, bd=bd, bw=bw,
                stacked=int(b4 == d and bd == d and bw == d),
                named=int(named), generic=int(generic),
                tag='+'.join(sorted(set('/'.join(tags).split('/')))) if tags else '',
                disp=int(disp_bar or disp_fvg), disp_bar=int(disp_bar),
                disp_fvg=int(disp_fvg))

    # entry 1: break close
    entry = cc[i] + SLIP * d
    stop = (lo_leg - BUF) if d == 1 else (hi_leg + BUF)
    s = simulate(h1.index[i] + pd.Timedelta(minutes=60), entry, stop, d)
    if s:
        rows.append(dict(base, entry_style='break', filled=1,
                         Rpct=s['R'] / entry * 100, mfe=s['mfe'], w1=s['w1'],
                         w2=s['w2'], w3=s['w3'], stopped=s['stopped'],
                         r_1r=s['r_1r'], r_2r=s['r_2r'], r_so=s['r_so']))

    # entry 2: 0.5 retrace of the leg
    lvl = hi_leg - 0.5 * (hi_leg - lo_leg) if d == 1 else lo_leg + 0.5 * (hi_leg - lo_leg)
    fill_i = None
    for k in range(i + 1, min(i + 1 + VALID_BARS, n1)):
        if (d == 1 and ll[k] <= lvl) or (d == -1 and hh[k] >= lvl):
            fill_i = k
            break
        if (d == 1 and cc[k] < lo_leg) or (d == -1 and cc[k] > hi_leg):
            break
    if fill_i is None:
        rows.append(dict(base, entry_style='retrace50', filled=0))
    else:
        entry = lvl + SLIP * d
        s = simulate(h1.index[fill_i] + pd.Timedelta(minutes=60), entry, stop, d)
        if s:
            rows.append(dict(base, entry_style='retrace50', filled=1,
                             Rpct=s['R'] / entry * 100, mfe=s['mfe'], w1=s['w1'],
                             w2=s['w2'], w3=s['w3'], stopped=s['stopped'],
                             r_1r=s['r_1r'], r_2r=s['r_2r'], r_so=s['r_so']))

Q = pd.DataFrame(rows)
Q.to_csv('research/smc/qualified.csv', index=False)

hdr = (f"{'setup':<48}{'fill%':>7}{'n':>5}{'hit1R':>7}{'hit2R':>7}"
       f"{'exp1R':>8}{'exp2R':>8}{'expSO':>8}{'R%px':>7}")


def line(df, label, minn=8):
    f = df[df.filled == 1]
    if len(f) < minn:
        return
    print(f"{label:<48}{df.filled.mean()*100:6.0f}%{len(f):>5}"
          f"{f.w1.mean()*100:6.0f}%{f.w2.mean()*100:6.0f}%{f.r_1r.mean():>8.3f}"
          f"{f.r_2r.mean():>8.3f}{f.r_so.mean():>8.3f}{f.Rpct.median():>7.2f}")


print("=" * 105)
print("Named-pool sweeps found:  bullish", int(sw_named_dn.sum()),
      " bearish", int(sw_named_up.sum()), " of", n1, "1H bars")
print("Events tagged: named-sweep", int((Q[Q.entry_style == 'break'].named).sum()),
      " displacement", int((Q[Q.entry_style == 'break'].disp).sum()),
      " both", int(((Q[Q.entry_style == 'break'].named == 1) &
                    (Q[Q.entry_style == 'break'].disp == 1)).sum()))

for style in ('break', 'retrace50'):
    for d, dtag, bwv in ((1, 'LONG', 1), (-1, 'SHORT', -1)):
        g = Q[(Q.entry_style == style) & (Q.dir == d)]
        print(f"\n{'='*105}\n{dtag} / entry = {style}\n{'='*105}")
        print(hdr)
        line(g, "everything")
        line(g[g.disp == 1], "+ displacement")
        line(g[g.named == 1], "+ named-pool sweep")
        line(g[(g.named == 1) & (g.disp == 1)], "+ named sweep + displacement")
        line(g[(g.named == 1) & (g.disp == 1) & (g.bw == bwv)],
             "+ named + displacement + 1W aligned")
        line(g[(g.named == 1) & (g.disp == 1) & (g.stacked == 1)],
             "+ named + displacement + full stack")
        for kind in ('CHoCH', 'BOS'):
            k = g[g.kind == kind]
            line(k[(k.named == 1) & (k.disp == 1)], f"  {kind}: named + displacement")
            line(k[(k.named == 1) & (k.disp == 1) & (k.bw == bwv)],
                 f"  {kind}: named + displacement + 1W aligned")

print(f"\n{'='*105}\nWhich pool got swept? (break entry, both directions, named only)\n{'='*105}")
g = Q[(Q.entry_style == 'break') & (Q.named == 1) & (Q.filled == 1)]
print(g.groupby('tag').agg(n=('r_so', 'size'), hit1R=('w1', 'mean'),
                           hit2R=('w2', 'mean'), exp2R=('r_2r', 'mean'),
                           expSO=('r_so', 'mean')).round(3).to_string())
