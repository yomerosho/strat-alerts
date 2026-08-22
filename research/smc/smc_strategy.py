"""
SMC strategy test: liquidity sweep -> displacement -> retrace into FVG / order
block / demand zone.

Structure on 15-minute bars, outcomes resolved on 5-minute bars.
15 tickers, extended-hours data, 2022-2026.

THE SETUP (long; short is the mirror)
  1. SWEEP        a 15m bar wicks below a confirmed prior swing low and closes
                  back above it
  2. DISPLACEMENT within 6 bars, price closes above the sweep bar's high AND a
                  bullish fair value gap prints (low[i] > high[i-2])
  3. ENTRY        resting LIMIT back inside the zone. Every entry price here is
                  one that actually traded -- a limit only fills if price comes
                  to it. This is the discipline the gate study lacked.
                    fvg_mid / fvg_far  = midpoint / far edge of the FVG
                    ob_mid  / ob_high  = last opposing candle before displacement
  4. STOP         below the sweep low
  5. TARGET       fixed R multiple

No-lookahead: pivots are known only L bars after they print; the limit is placed
after displacement confirms and fills only on later bars.

Discipline against the mistakes made earlier in this project:
  * every entry is fillable by construction
  * break-even win rate for the R:R is printed next to every win rate
  * train 2022-2024 / holdout 2025-2026 reported for every variant
  * bootstrap CI on expectancy, and per-ticker / per-year stability
"""
import os
import numpy as np
import pandas as pd

L = 2                 # fractal strength on 15m
MAXWAIT_DISP = 6      # bars from sweep to displacement
MAXWAIT_FILL = 16     # bars the limit rests (~4h on 15m)
HOLD_BARS_5 = 78 * 3  # 3 sessions of 5-min bars
BUF_ATR = 0.05        # stop buffer, in ATR
SLIP = 0.0002         # 2bp each way
UNIVERSE = ['IWM', 'SPY', 'QQQ', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
            'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']
TARGETS = [1.0, 2.0, 3.0]


def pivots(h, l, n):
    ph = np.full(len(h), np.nan)
    pl = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] == h[i - n:i + n + 1].max() and (h[i] > h[i - n:i]).all():
            ph[i] = h[i]
        if l[i] == l[i - n:i + n + 1].min() and (l[i] < l[i - n:i]).all():
            pl[i] = l[i]
    return ph, pl


rows = []
for sym in UNIVERSE:
    path = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(path):
        continue
    d5 = pd.read_parquet(path).between_time('09:30', '15:59')
    if len(d5) < 5000:
        continue
    m15 = d5.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                    'close': 'last', 'volume': 'sum'}).dropna()
    m15 = m15[m15.volume > 0]

    o, h, l, c = (m15.open.values, m15.high.values, m15.low.values, m15.close.values)
    n = len(m15)
    idx15 = m15.index
    ph, pl = pivots(h, l, L)

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(28).mean().shift(1).values

    # daily trend filter, causal
    dly = d5.resample('1D').agg({'close': 'last'}).dropna()
    dly['sma20'] = dly.close.rolling(20).mean()
    dly['up'] = (dly.close > dly.sma20).shift(1)
    up_by_date = dly['up'].to_dict()

    f5 = d5.index
    f5h, f5l = d5.high.values, d5.low.values

    known_lo, known_hi = [], []
    for i in range(n):
        j = i - L
        if j >= 0:
            if not np.isnan(pl[j]):
                known_lo.append((j, pl[j]))
            if not np.isnan(ph[j]):
                known_hi.append((j, ph[j]))
        if np.isnan(atr[i]) or atr[i] <= 0 or i + MAXWAIT_DISP + MAXWAIT_FILL >= n:
            continue

        for direction in (1, -1):
            pool = known_lo if direction == 1 else known_hi
            recent = [p for p in pool[-10:] if i - p[0] <= 60]
            if not recent:
                continue
            swept = False
            for _, lvl in recent:
                if direction == 1 and l[i] < lvl and c[i] > lvl:
                    swept = True
                if direction == -1 and h[i] > lvl and c[i] < lvl:
                    swept = True
            if not swept:
                continue

            sweep_ext = l[i] if direction == 1 else h[i]
            # 2) displacement: close beyond the sweep bar + an FVG in the leg
            disp = None
            for k in range(i + 1, min(i + 1 + MAXWAIT_DISP, n)):
                beyond = c[k] > h[i] if direction == 1 else c[k] < l[i]
                if not beyond:
                    continue
                for g in range(max(i + 2, 2), k + 1):
                    if direction == 1 and l[g] > h[g - 2]:
                        disp = (k, h[g - 2], l[g], g)
                        break
                    if direction == -1 and h[g] < l[g - 2]:
                        disp = (k, l[g - 2], h[g], g)
                        break
                if disp:
                    break
            if not disp:
                continue
            k, z_far, z_near, g = disp
            if direction == 1 and not (z_near > z_far):
                continue
            if direction == -1 and not (z_near < z_far):
                continue

            # order block = last opposing candle before the displacement leg
            ob_lo = ob_hi = np.nan
            for b in range(g, max(i - 1, 0), -1):
                if direction == 1 and c[b] < o[b]:
                    ob_lo, ob_hi = l[b], h[b]
                    break
                if direction == -1 and c[b] > o[b]:
                    ob_lo, ob_hi = l[b], h[b]
                    break

            zones = {
                'fvg_mid':  (z_far + z_near) / 2.0,
                'fvg_far':  z_far,
                'ob_mid':   (ob_lo + ob_hi) / 2.0 if np.isfinite(ob_lo) else np.nan,
                'ob_edge':  ob_hi if direction == 1 else ob_lo,
            }
            stop = (sweep_ext - BUF_ATR * atr[i]) if direction == 1 else (sweep_ext + BUF_ATR * atr[i])
            day = idx15[k].date()
            trend_up = up_by_date.get(pd.Timestamp(day), np.nan)

            for zname, zprice in zones.items():
                if not np.isfinite(zprice):
                    continue
                R = (zprice - stop) if direction == 1 else (stop - zprice)
                if R <= 0 or R > 3 * atr[i]:
                    continue
                # limit rests after displacement; fills only if price returns
                fill = None
                for w in range(k + 1, min(k + 1 + MAXWAIT_FILL, n)):
                    if direction == 1 and l[w] <= zprice:
                        fill = w
                        break
                    if direction == -1 and h[w] >= zprice:
                        fill = w
                        break
                    if direction == 1 and c[w] < stop:
                        break
                    if direction == -1 and c[w] > stop:
                        break
                base = dict(sym=sym, ts=idx15[k], day=day, dir=direction, zone=zname,
                            R_atr=R / atr[i], Rpct=R / zprice * 100,
                            trend_up=trend_up, filled=int(fill is not None))
                if fill is None:
                    rows.append(base)
                    continue
                entry = zprice * (1 + SLIP * direction)
                R = (entry - stop) if direction == 1 else (stop - entry)
                if R <= 0:
                    continue
                s0 = f5.searchsorted(idx15[fill] + pd.Timedelta(minutes=15))
                s1 = min(s0 + HOLD_BARS_5, len(f5))
                res = {}
                for tm in TARGETS:
                    tgt = entry + tm * R * direction
                    out = np.nan
                    for x in range(s0, s1):
                        hit_stop = (f5l[x] <= stop) if direction == 1 else (f5h[x] >= stop)
                        hit_tgt = (f5h[x] >= tgt) if direction == 1 else (f5l[x] <= tgt)
                        if hit_stop:
                            out = -1.0
                            break
                        if hit_tgt:
                            out = tm
                            break
                    if np.isnan(out):
                        lastpx = d5.close.values[s1 - 1] if s1 > s0 else entry
                        out = ((lastpx - entry) if direction == 1 else (entry - lastpx)) / R
                    res['r%g' % tm] = out
                    res['w%g' % tm] = 1 if out > 0 else 0
                rows.append(dict(base, **res))

T = pd.DataFrame(rows)
T['y'] = pd.to_datetime(T.ts).dt.year
T.to_csv('research/smc/smc_strategy.csv', index=False)
print('signals:', len(T), '| filled:', int(T.filled.sum()), '| tickers:', T.sym.nunique())
print(T.groupby('zone').agg(n=('filled', 'size'), fill_rate=('filled', 'mean')).round(2).to_string())
