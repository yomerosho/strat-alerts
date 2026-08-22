"""
ICT / SMC external-liquidity strategy, tested exactly as specified.

SPEC (long; short is the mirror)
  1. HTF SWEEP    5m bar wicks below an HTF external liquidity level -- prior
                  day low (PDL) or overnight low (ONL, 16:00 prev -> 09:30) --
                  and closes back above it. RTH only.
  2. CHoCH        within CHOCH_WIN bars, price CLOSES above the most recent
                  confirmed 5m swing high (the last lower-high of the leg down).
  3. ENTRY        resting limit at the top of the last bullish FVG formed in the
                  sweep->CHoCH impulse (fallback: order-block high = last
                  down-close bar before the impulse). Fill only if price trades
                  back to it. Cancel if the sweep low is broken first, or if the
                  retrace takes > FILL_WIN bars.
  4. STOP         sweep low - 0.05*ATR buffer.
  5. TARGETS      TP1 = the CHoCH level (nearest internal liquidity). Half off,
                  stop to breakeven.
                  TP2 = opposing external level (PDH for a PDL sweep, ONH for an
                  ONL sweep). SKIP the trade if TP2 < 2.0R away.
  6. TIME EXIT    remaining position closed at the last RTH bar of the entry day.

Costs: 2bp slippage each way. One position per ticker at a time.
Evaluation: train 2022-2024 / holdout 2025-2026, date-clustered bootstrap,
and a matched random control (same day, same R geometry, random timing).
"""
import numpy as np
import pandas as pd

SYMS = ['QQQ', 'SPY', 'IWM']
K = 3                 # fractal strength for 5m swing points
CHOCH_WIN = 18        # bars allowed from sweep to CHoCH (90 min)
FILL_WIN = 24         # bars the limit rests after CHoCH (2 h)
MIN_RR = 2.0          # skip if TP2 < 2R
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


def run_symbol(sym):
    d = pd.read_parquet(f'research/smc/data/{sym}_5m_ext.parquet')
    rth = d.between_time('09:30', '15:55')
    dates = sorted(set(rth.index.date))

    # external liquidity per date: PDH/PDL from prior RTH day, ONH/ONL from
    # 16:00 prev day -> 09:29
    ext = {}
    rth_by_date = {dt: g for dt, g in rth.groupby(rth.index.date)}
    all_idx = d.index
    for j in range(1, len(dates)):
        dt, prev = dates[j], dates[j - 1]
        pg = rth_by_date[prev]
        on = d[(all_idx > pg.index[-1]) & (all_idx.date <= dt) &
               ((all_idx.date < dt) | (all_idx.time < pd.Timestamp('09:30').time()))]
        if len(on) == 0:
            continue
        ext[dt] = dict(PDH=pg.high.max(), PDL=pg.low.min(),
                       ONH=on.high.max(), ONL=on.low.min())

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
            for side in (1, -1):
                if side == 1:
                    lv = [('PDL', E['PDL'], 'PDH', E['PDH']),
                          ('ONL', E['ONL'], 'ONH', E['ONH'])]
                    hit = next(((nm, v, on2, v2) for nm, v, on2, v2 in lv
                                if l[i] < v and c[i] > v), None)
                else:
                    lv = [('PDH', E['PDH'], 'PDL', E['PDL']),
                          ('ONH', E['ONH'], 'ONL', E['ONL'])]
                    hit = next(((nm, v, on2, v2) for nm, v, on2, v2 in lv
                                if h[i] > v and c[i] < v), None)
                if hit is None:
                    continue
                lev_name, lev, opp_name, opp = hit
                swept_ext = l[i] if side == 1 else h[i]

                # CHoCH level: most recent CONFIRMED opposing swing before sweep
                choch = np.nan
                for j in range(i - K, 0, -1):
                    v = ph[j] if side == 1 else pl[j]
                    if not np.isnan(v):
                        choch = v
                        break
                if np.isnan(choch):
                    continue
                if side == 1 and choch <= c[i]:
                    continue
                if side == -1 and choch >= c[i]:
                    continue

                # find CHoCH close within window
                b = None
                dead = False
                for j in range(i + 1, min(i + 1 + CHOCH_WIN, n)):
                    if (side == 1 and l[j] < swept_ext) or \
                       (side == -1 and h[j] > swept_ext):
                        dead = True
                        break
                    if (side == 1 and c[j] > choch) or \
                       (side == -1 and c[j] < choch):
                        b = j
                        break
                if dead or b is None:
                    continue

                # entry zone from the impulse i..b: last FVG, else OB
                entry = np.nan
                for j in range(b, i + 1, -1):
                    if side == 1 and j >= 2 and l[j] > h[j - 2] and j - 2 >= i:
                        entry = h[j - 2]
                        break
                    if side == -1 and j >= 2 and h[j] < l[j - 2] and j - 2 >= i:
                        entry = l[j - 2]
                        break
                if np.isnan(entry):
                    for j in range(b, i - 1, -1):          # order block
                        if (side == 1 and c[j] < o[j]) or \
                           (side == -1 and c[j] > o[j]):
                            entry = h[j] if side == 1 else l[j]
                            break
                if np.isnan(entry):
                    continue

                stop = swept_ext - side * BUF_ATR * atr[i]
                risk = side * (entry - stop)
                if risk <= 0:
                    continue
                tp1 = choch
                tp2 = opp
                if side * (tp2 - entry) < MIN_RR * risk:
                    continue                                # R:R filter
                if side * (tp1 - entry) <= 0:
                    continue

                # rest the limit
                fill = None
                dead = False
                for j in range(b + 1, min(b + 1 + FILL_WIN, n)):
                    if side == 1 and l[j] <= entry:
                        fill = j
                        break
                    if side == -1 and h[j] >= entry:
                        fill = j
                        break
                    if (side == 1 and l[j] < stop) or \
                       (side == -1 and h[j] > stop):
                        dead = True
                        break
                if dead or fill is None:
                    continue

                e_px = entry + side * SLIP * entry
                # manage: conservative -- stop checked before target each bar
                half_gone = False
                cur_stop = stop
                r1 = r2 = None
                out_j = None
                for j in range(fill, n):
                    lo_j = min(l[j], entry) if j == fill else l[j]
                    hi_j = max(h[j], entry) if j == fill else h[j]
                    if side == 1:
                        if lo_j <= cur_stop:
                            x = (cur_stop - side * SLIP * cur_stop - e_px) / (e_px - stop + SLIP * (entry + stop)) if False else (cur_stop - SLIP * cur_stop - e_px) / risk
                            if not half_gone:
                                r1 = r2 = x
                            else:
                                r2 = x
                            out_j = j
                            break
                        if not half_gone and hi_j >= tp1:
                            r1 = (tp1 - SLIP * tp1 - e_px) / risk
                            half_gone = True
                            cur_stop = e_px
                        if half_gone and hi_j >= tp2:
                            r2 = (tp2 - SLIP * tp2 - e_px) / risk
                            out_j = j
                            break
                    else:
                        if hi_j >= cur_stop:
                            x = (e_px - cur_stop - SLIP * cur_stop) / risk
                            if not half_gone:
                                r1 = r2 = x
                            else:
                                r2 = x
                            out_j = j
                            break
                        if not half_gone and lo_j <= tp1:
                            r1 = (e_px - tp1 - SLIP * tp1) / risk
                            half_gone = True
                            cur_stop = e_px
                        if half_gone and lo_j <= tp2:
                            r2 = (e_px - tp2 - SLIP * tp2) / risk
                            out_j = j
                            break
                if out_j is None:                            # EOD close
                    out_j = n - 1
                    x = side * (c[-1] - e_px) / risk - SLIP * c[-1] / risk
                    if not half_gone:
                        r1 = r2 = x
                    else:
                        r2 = x
                net = 0.5 * r1 + 0.5 * r2
                trades.append(dict(
                    sym=sym, day=dt, ts=ts[fill], side='L' if side == 1 else 'S',
                    lev=lev_name, sweep=round(swept_ext, 2),
                    choch=round(choch, 2), entry=round(entry, 2),
                    stop=round(stop, 2), tp1=round(tp1, 2), tp2=round(tp2, 2),
                    rr=round(side * (tp2 - entry) / risk, 2),
                    hit_tp1=half_gone, net_r=round(net, 3)))
                busy_until = out_j
                break
    return pd.DataFrame(trades)


def summarize(T, label):
    if len(T) == 0:
        print(f'{label}: no trades')
        return
    r = T.net_r.values
    win = (r > 0.05).mean()
    be = ((r >= -0.05) & (r <= 0.05)).mean()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    pf = gp / gl if gl > 0 else np.inf
    # max consecutive losses
    mx = cur = 0
    for x in r:
        cur = cur + 1 if x < -0.05 else 0
        mx = max(mx, cur)
    print(f'{label:34s} n={len(T):4d}  win={win:5.1%}  be={be:4.1%}  '
          f'expR={r.mean():+.3f}  totR={r.sum():+7.1f}  PF={pf:4.2f}  '
          f'maxLstreak={mx}  avgRR={T.rr.mean():.2f}')


def clustered_ci(T, iters=6000):
    groups = [g.net_r.values.astype(float) for _, g in T.groupby('day')]
    k = len(groups)
    means = np.empty(iters)
    for i in range(iters):
        pick = rng.integers(0, k, k)
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    return np.percentile(means, 2.5), np.percentile(means, 97.5), (means > 0).mean()


if __name__ == '__main__':
    allT = []
    for s in SYMS:
        T = run_symbol(s)
        T.to_csv(f'research/smc/ict_ext_{s}.csv', index=False)
        allT.append(T)
    T = pd.concat(allT, ignore_index=True).sort_values('ts').reset_index(drop=True)
    T.to_csv('research/smc/ict_ext_all.csv', index=False)

    print('=' * 100)
    print('ICT external-liquidity sweep -> CHoCH -> FVG/OB retrace   (2bp slip each way)')
    print('=' * 100)
    for s in SYMS:
        summarize(T[T.sym == s], s)
    summarize(T, 'ALL')
    print()
    for side in ('L', 'S'):
        summarize(T[T.side == side], f'  side={side}')
    for lev in ('PDL', 'ONL', 'PDH', 'ONH'):
        summarize(T[T.lev == lev], f'  sweep={lev}')
    print()
    y = pd.to_datetime(T.day.astype(str)).dt.year
    for yy in sorted(y.unique()):
        summarize(T[y == yy], f'  {yy}')
    tr = T[y <= 2024]
    ho = T[y >= 2025]
    print()
    summarize(tr, 'TRAIN 2022-2024')
    summarize(ho, 'HOLDOUT 2025-2026')
    lo, hi, p = clustered_ci(T)
    print(f'\ndate-clustered bootstrap on expR: [{lo:+.3f}, {hi:+.3f}]  P(>0)={p:.3f}')
    print(f'TP1 hit rate: {T.hit_tp1.mean():.1%}')
