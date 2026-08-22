"""
Refinement grid for the ICT external-liquidity strategy.

The spec-exact run failed (expR -0.180, PF 0.46). Diagnosis: TP2 at the
opposing external level averages 6.3R away (never reached), TP1 is often a
fraction of 1R (tiny partials vs full -1R losses). Refinements attack the
exit geometry, side selection, and level selection; detection is unchanged.

Variants share the same detection engine; each config changes only:
  sides       'L', 'S', or 'LS'
  levels      subset of {PD, ON}
  exit        'spec'   = TP1 @ choch (scale, BE) + TP2 @ external
              'fixed'  = single target at tgt*R, no scale-out
              'scale1' = half off at +1R, stop to BE, runner to external TP2
              'scale1e'= half off at +1R, stop to BE, runner to EOD
  tgt         R multiple for 'fixed'
  min_tp1     require choch TP1 >= this many R (0 disables; spec used >0)
  zone        'top' = FVG top (spec), 'mid' = FVG midpoint (deeper, smaller risk)
  rrmin       min R:R to external TP2 required (spec 2.0; 0 disables)
"""
import numpy as np
import pandas as pd

SYMS = ['QQQ', 'SPY', 'IWM']
K = 3
CHOCH_WIN = 18
FILL_WIN = 24
BUF_ATR = 0.05
SLIP = 0.0002
rng = np.random.default_rng(20260822)


def pivots(h, l, k):
    ph = np.full(len(h), np.nan)
    pl = np.full(len(l), np.nan)
    for i in range(k, len(h) - k):
        if h[i] == h[i - k:i + k + 1].max() and (h[i] > h[i - k:i]).all():
            ph[i] = h[i]
        if l[i] == l[i - k:i + k + 1].min() and (l[i] < l[i - k:i]).all():
            pl[i] = l[i]
    return ph, pl


DATA = {}
def load(sym):
    if sym in DATA:
        return DATA[sym]
    d = pd.read_parquet(f'research/smc/data/{sym}_5m_ext.parquet')
    rth = d.between_time('09:30', '15:55')
    dates = sorted(set(rth.index.date))
    rth_by_date = {dt: g for dt, g in rth.groupby(rth.index.date)}
    ah = d.between_time('16:00', '23:59')
    pm = d.between_time('04:00', '09:29')
    ah_h = ah.high.groupby(ah.index.date).max(); ah_l = ah.low.groupby(ah.index.date).min()
    pm_h = pm.high.groupby(pm.index.date).max(); pm_l = pm.low.groupby(pm.index.date).min()
    ext = {}
    for j in range(1, len(dates)):
        dt, prev = dates[j], dates[j - 1]
        pg = rth_by_date[prev]
        onh = np.nanmax([ah_h.get(prev, np.nan), pm_h.get(dt, np.nan)])
        onl = np.nanmin([ah_l.get(prev, np.nan), pm_l.get(dt, np.nan)])
        if np.isnan(onh) or np.isnan(onl):
            continue
        ext[dt] = dict(PDH=pg.high.max(), PDL=pg.low.min(), ONH=onh, ONL=onl)
    DATA[sym] = (dates, rth_by_date, ext)
    return DATA[sym]


def run_symbol(sym, sides='LS', levels=('PD', 'ON'), exit='spec', tgt=2.0,
               min_tp1=0.0, zone='top', rrmin=2.0):
    dates, rth_by_date, ext = load(sym)
    side_list = [s for s, ch in ((1, 'L'), (-1, 'S')) if ch in sides]
    trades = []
    for dt in dates[1:]:
        if dt not in ext or dt not in rth_by_date:
            continue
        g = rth_by_date[dt]
        o, h, l, c = g.open.values, g.high.values, g.low.values, g.close.values
        ts = g.index
        n = len(g)
        if n < 30:
            continue
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                          np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr).rolling(14, min_periods=5).mean().shift(1).values
        ph, pl = pivots(h, l, K)
        E = ext[dt]

        busy_until = -1
        for i in range(1, n):
            if i <= busy_until or np.isnan(atr[i]) or atr[i] <= 0:
                continue
            for side in side_list:
                cand = []
                if side == 1:
                    if 'PD' in levels: cand.append(('PDL', E['PDL'], E['PDH']))
                    if 'ON' in levels: cand.append(('ONL', E['ONL'], E['ONH']))
                    hit = next(((nm, v, v2) for nm, v, v2 in cand
                                if l[i] < v and c[i] > v), None)
                else:
                    if 'PD' in levels: cand.append(('PDH', E['PDH'], E['PDL']))
                    if 'ON' in levels: cand.append(('ONH', E['ONH'], E['ONL']))
                    hit = next(((nm, v, v2) for nm, v, v2 in cand
                                if h[i] > v and c[i] < v), None)
                if hit is None:
                    continue
                lev_name, lev, opp = hit
                swept_ext = l[i] if side == 1 else h[i]

                choch = np.nan
                for j in range(i - K, 0, -1):
                    v = ph[j] if side == 1 else pl[j]
                    if not np.isnan(v):
                        choch = v
                        break
                if np.isnan(choch) or side * (choch - c[i]) <= 0:
                    continue

                b = None
                dead = False
                for j in range(i + 1, min(i + 1 + CHOCH_WIN, n)):
                    if side * (l[j] if side == 1 else h[j]) < side * swept_ext or \
                       (side == -1 and h[j] > swept_ext):
                        if (side == 1 and l[j] < swept_ext) or (side == -1 and h[j] > swept_ext):
                            dead = True
                            break
                    if side * (c[j] - choch) > 0:
                        b = j
                        break
                if dead or b is None:
                    continue

                entry = np.nan
                for j in range(b, i + 1, -1):
                    if j >= 2 and j - 2 >= i:
                        if side == 1 and l[j] > h[j - 2]:
                            entry = h[j - 2] if zone == 'top' else 0.5 * (h[j - 2] + l[j])
                            break
                        if side == -1 and h[j] < l[j - 2]:
                            entry = l[j - 2] if zone == 'top' else 0.5 * (l[j - 2] + h[j])
                            break
                if np.isnan(entry):
                    for j in range(b, i - 1, -1):
                        if (side == 1 and c[j] < o[j]) or (side == -1 and c[j] > o[j]):
                            entry = h[j] if side == 1 else l[j]
                            if zone == 'mid':
                                entry = 0.5 * ((h[j] if side == 1 else l[j]) +
                                               (l[j] if side == 1 else h[j]))
                            break
                if np.isnan(entry):
                    continue

                stop = swept_ext - side * BUF_ATR * atr[i]
                risk = side * (entry - stop)
                if risk <= 0:
                    continue
                if rrmin > 0 and side * (opp - entry) < rrmin * risk:
                    continue
                if exit == 'spec':
                    tp1 = choch
                    if side * (tp1 - entry) <= 0 or side * (tp1 - entry) < min_tp1 * risk:
                        continue
                    tp2 = opp
                elif exit == 'fixed':
                    tp1 = None
                    tp2 = entry + side * tgt * risk
                else:  # scale1 / scale1e
                    tp1 = entry + side * 1.0 * risk
                    tp2 = opp if exit == 'scale1' else None

                fill = None
                dead = False
                for j in range(b + 1, min(b + 1 + FILL_WIN, n)):
                    if (side == 1 and l[j] <= entry) or (side == -1 and h[j] >= entry):
                        fill = j
                        break
                    if (side == 1 and l[j] < stop) or (side == -1 and h[j] > stop):
                        dead = True
                        break
                if dead or fill is None:
                    continue

                e_px = entry + side * SLIP * entry
                half_gone = False
                cur_stop = stop
                r1 = r2 = None
                out_j = None
                for j in range(fill, n):
                    if side == 1:
                        if l[j] <= cur_stop:
                            x = (cur_stop - SLIP * cur_stop - e_px) / risk
                            r1, r2 = (x, x) if not half_gone else (r1, x)
                            out_j = j
                            break
                        if tp1 is not None and not half_gone and h[j] >= tp1:
                            r1 = (tp1 - SLIP * tp1 - e_px) / risk
                            half_gone = True
                            cur_stop = e_px
                        if tp2 is not None and (half_gone or tp1 is None) and h[j] >= tp2:
                            r2 = (tp2 - SLIP * tp2 - e_px) / risk
                            if tp1 is None:
                                r1 = r2
                            out_j = j
                            break
                    else:
                        if h[j] >= cur_stop:
                            x = (e_px - cur_stop - SLIP * cur_stop) / risk
                            r1, r2 = (x, x) if not half_gone else (r1, x)
                            out_j = j
                            break
                        if tp1 is not None and not half_gone and l[j] <= tp1:
                            r1 = (e_px - tp1 - SLIP * tp1) / risk
                            half_gone = True
                            cur_stop = e_px
                        if tp2 is not None and (half_gone or tp1 is None) and l[j] <= tp2:
                            r2 = (e_px - tp2 - SLIP * tp2) / risk
                            if tp1 is None:
                                r1 = r2
                            out_j = j
                            break
                if out_j is None:
                    out_j = n - 1
                    x = side * (c[-1] - e_px) / risk - SLIP * c[-1] / risk
                    r1, r2 = (x, x) if not half_gone else (r1, x)
                net = 0.5 * r1 + 0.5 * r2
                trades.append(dict(sym=sym, day=dt, ts=ts[fill],
                                   side='L' if side == 1 else 'S', lev=lev_name,
                                   sweep=round(swept_ext, 2), choch=round(choch, 2),
                                   entry=round(entry, 2), stop=round(stop, 2),
                                   tp2=round(tp2, 2) if tp2 is not None else np.nan,
                                   rr=round(side * (opp - entry) / risk, 2),
                                   hit_tp1=half_gone, net_r=round(net, 3)))
                busy_until = out_j
                break
    return pd.DataFrame(trades)


def stats(T):
    if len(T) == 0:
        return dict(n=0)
    r = T.net_r.values
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    y = pd.to_datetime(T.day.astype(str)).dt.year
    ho = T[y >= 2025]
    return dict(n=len(T), win=(r > 0.05).mean(), expR=r.mean(), totR=r.sum(),
                pf=gp / gl if gl > 0 else np.inf,
                ho_n=len(ho), ho_expR=ho.net_r.mean() if len(ho) else np.nan)


CONFIGS = [
    ('V0 spec (baseline)',      dict()),
    ('V1 longs only',           dict(sides='L')),
    ('V2 L fixed 2R',           dict(sides='L', exit='fixed', tgt=2.0, rrmin=0)),
    ('V3 L fixed 1R',           dict(sides='L', exit='fixed', tgt=1.0, rrmin=0)),
    ('V4 L half@1R+BE, ext TP2',dict(sides='L', exit='scale1', rrmin=2.0)),
    ('V5 L half@1R+BE, EOD run',dict(sides='L', exit='scale1e', rrmin=0)),
    ('V6 V4 ON levels only',    dict(sides='L', exit='scale1', rrmin=2.0, levels=('ON',))),
    ('V7 V4 PD levels only',    dict(sides='L', exit='scale1', rrmin=2.0, levels=('PD',))),
    ('V8 V4 entry FVG mid',     dict(sides='L', exit='scale1', rrmin=2.0, zone='mid')),
    ('V9 spec, TP1>=1R req',    dict(min_tp1=1.0)),
    ('V10 S fixed 2R',          dict(sides='S', exit='fixed', tgt=2.0, rrmin=0)),
]

if __name__ == '__main__':
    print(f"{'variant':28s} {'n':>5} {'win':>6} {'expR':>7} {'totR':>8} "
          f"{'PF':>5} {'HO n':>5} {'HO expR':>8}")
    print('-' * 78)
    best = {}
    for name, cfg in CONFIGS:
        T = pd.concat([run_symbol(s, **cfg) for s in SYMS], ignore_index=True)
        st = stats(T)
        best[name] = T
        print(f"{name:28s} {st['n']:5d} {st['win']:6.1%} {st['expR']:+7.3f} "
              f"{st['totR']:+8.1f} {st['pf']:5.2f} {st['ho_n']:5d} {st['ho_expR']:+8.3f}")
    import pickle
    with open('research/smc/ict_refine_trades.pkl', 'wb') as f:
        pickle.dump(best, f)
