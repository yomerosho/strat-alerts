"""
AMN 6-Point Sequence -- backtest of BOTH alert conditions.

The Pine state machine (pine/amn_5point.pine) is ported line for line so the
signals tested are the ones the indicator actually fires:

  ALERT A  "AMN 1-5 sequence qualified"   -> `fresh`
           the moment the ATR zigzag confirms a 5th pivot that satisfies
           P3>P1, P5>P3, P4>P2  (bullish; mirrored for bearish)

  ALERT B  "AMN point 6 - sweep and reclaim"  -> `entry`
           price sweeps the point-4 level, then a candle CLOSES back through it

Both enter at the CLOSE of the signal bar (that is when the alert fires), so
every fill is a price that actually traded. Outcomes resolve on the same 5-min
series, intrabar ties given to the stop (adverse).

Stops, taken from his own rules:
  B : beyond the sweep extreme  (the extreme printed while sweeping point 4)
  A : point 2 -- the level whose break invalidates the sequence

Controls: for every real signal, a matched control trades the SAME ticker, the
SAME direction and the SAME stop distance, entered at a random bar on the same
day. That is the test that separated real edges from imagined ones in every
prior study here; a setup that cannot beat it is selecting nothing.
"""
import os
import numpy as np
import pandas as pd

LEG_ATR   = 1.5
MAX_WAIT  = 120          # bars to wait for the point-6 sweep
HOLD_BARS = 78 * 3       # 3 sessions of 5-min bars
SLIP      = 0.0002       # 2bp each way
UNIVERSE  = ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
             'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']
TARGETS   = [1.0, 2.0, 3.0]
rng = np.random.default_rng(20260824)


def rma_atr(h, l, c, n=14):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(tr), np.nan)
    a = tr[:n].mean()
    out[n - 1] = a
    for i in range(n, len(tr)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def htf_zones(df):
    """Same order block rule as f_htfZones(), on 60-min bars, mapped back to 5m.

    Uses only bars that have CLOSED, so nothing is knowable early.
    """
    h1 = df.resample('60min').agg({'open': 'first', 'high': 'max',
                                   'low': 'min', 'close': 'last'}).dropna()
    o, h, l, c = h1.open.values, h1.high.values, h1.low.values, h1.close.values
    dT = dB = sT = sB = np.nan
    rows = []
    for i in range(len(h1)):
        if i >= 1:
            if c[i - 1] < o[i - 1] and c[i] > h[i - 1]:
                dT, dB = h[i - 1], l[i - 1]
            if c[i - 1] > o[i - 1] and c[i] < l[i - 1]:
                sT, sB = h[i - 1], l[i - 1]
        rows.append((dT, dB, sT, sB))
    z = pd.DataFrame(rows, index=h1.index, columns=['dT', 'dB', 'sT', 'sB'])
    z.index = z.index + pd.Timedelta(minutes=60)      # known only after close
    return z.reindex(df.index, method='ffill')


def signals(df):
    """Port of the Pine. Returns one row per alert."""
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(c)
    A = rma_atr(h, l, c)
    Z = htf_zones(df)
    zdT, zdB, zsT, zsB = Z.dT.values, Z.dB.values, Z.sT.values, Z.sB.values

    pxs, bis, tps = [], [], []

    def push(p, b, t):
        if tps and tps[-1] == t:
            if (p > pxs[-1]) if t == 1 else (p < pxs[-1]):
                pxs[-1], bis[-1] = p, b
        else:
            pxs.append(p); bis.append(b); tps.append(t)
        while len(tps) > 12:
            pxs.pop(0); bis.pop(0); tps.pop(0)

    zdir, zext, zbar = 0, np.nan, 0
    aDir = 0
    aLvl = aInv = np.nan
    aBar = -1
    aSwept = False
    aP5Bar = None
    sweepExt = np.nan
    out = []

    for i in range(n):
        ml = LEG_ATR * A[i] if not np.isnan(A[i]) else np.nan
        if zdir == 0 and not np.isnan(ml):
            zdir, zext, zbar = 1, h[i], i
        if zdir == 1:
            if h[i] >= zext:
                zext, zbar = h[i], i
            if not np.isnan(ml) and l[i] < zext - ml:
                push(zext, zbar, 1)
                zdir, zext, zbar = -1, l[i], i
        elif zdir == -1:
            if l[i] <= zext:
                zext, zbar = l[i], i
            if not np.isnan(ml) and h[i] > zext + ml:
                push(zext, zbar, -1)
                zdir, zext, zbar = 1, h[i], i

        # ---- f_seq()
        seq = 0
        if len(tps) >= 5:
            t5 = tps[-1]
            if tps[-2] == -t5 and tps[-3] == t5 and tps[-4] == -t5 and tps[-5] == t5:
                v1, v2, v3, v4, v5 = pxs[-5], pxs[-4], pxs[-3], pxs[-2], pxs[-1]
                if t5 == 1 and v3 > v1 and v5 > v3 and v4 > v2:
                    seq = 1
                if t5 == -1 and v3 < v1 and v5 < v3 and v4 < v2:
                    seq = -1
        p5b = bis[-1] if bis else None
        fresh = seq != 0 and (aP5Bar is None or p5b != aP5Bar)

        if fresh:
            aP5Bar = p5b
            aDir, aBar, aSwept = seq, i, False
            aLvl, aInv = pxs[-2], pxs[-4]
            p5px = pxs[-1]
            zT = zdT[i] if seq == 1 else zsT[i]
            zB = zdB[i] if seq == 1 else zsB[i]
            inzone = (not np.isnan(zT)) and (zB <= aLvl <= zT)
            # retro-scan for a sweep during the zigzag confirmation lag
            sweepExt = np.nan
            lag = i - bis[-1]
            for k in range(0, min(lag, 300) + 1):
                j = i - k
                if j < 0:
                    break
                if (l[j] < aLvl) if seq == 1 else (h[j] > aLvl):
                    aSwept = True
                    e = l[j] if seq == 1 else h[j]
                    sweepExt = e if np.isnan(sweepExt) else (min(sweepExt, e) if seq == 1 else max(sweepExt, e))
            out.append(dict(i=i, kind='A', dir=seq, entry=c[i],
                            stop=aInv, p5=p5px, p4=aLvl, inzone=inzone))

        # ---- point 6
        if aDir != 0:
            if aDir == 1 and l[i] < aLvl:
                aSwept = True
                sweepExt = l[i] if np.isnan(sweepExt) else min(sweepExt, l[i])
            if aDir == -1 and h[i] > aLvl:
                aSwept = True
                sweepExt = h[i] if np.isnan(sweepExt) else max(sweepExt, h[i])
            fired = aSwept and ((c[i] > aLvl) if aDir == 1 else (c[i] < aLvl))
            if fired:
                out.append(dict(i=i, kind='B', dir=aDir, entry=c[i],
                                stop=sweepExt, p5=np.nan, p4=aLvl, inzone=np.nan))
                aDir, aSwept = 0, False
            elif ((c[i] < aInv) if aDir == 1 else (c[i] > aInv)) or (i - aBar > MAX_WAIT):
                aDir, aSwept = 0, False
    return pd.DataFrame(out)


def resolve(df, i0, entry, stop, d):
    """Walk forward; return dict of R outcomes under several exit models."""
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(c)
    R = (entry - stop) if d == 1 else (stop - entry)
    if not np.isfinite(R) or R <= 0:
        return None
    e = entry * (1 + SLIP * d)
    R = (e - stop) if d == 1 else (stop - e)
    if R <= 0:
        return None
    end = min(i0 + 1 + HOLD_BARS, n)
    BIG = 10 ** 9
    t_stop = BIG
    t = {m: BIG for m in TARGETS}
    t_be = BIG
    lastc = e
    for k in range(i0 + 1, end):
        up = ((h[k] - e) / R) if d == 1 else ((e - l[k]) / R)
        dn = ((l[k] - e) / R) if d == 1 else ((e - h[k]) / R)
        lastc = c[k]
        if dn <= -1.0 and t_stop == BIG:
            t_stop = k
        for m in TARGETS:
            if up >= m and t[m] == BIG:
                t[m] = k
        if t[1.0] < BIG and k > t[1.0] and dn <= 0.0 and t_be == BIG:
            t_be = k
    endR = ((lastc - e) / R) if d == 1 else ((e - lastc) / R)
    endR -= SLIP * e / R                                  # exit cost
    res = {}
    for m in TARGETS:
        if t_stop < BIG and t_stop <= t[m]:
            res['r%g' % m] = -1.0
        elif t[m] < BIG:
            res['r%g' % m] = m - (SLIP * e / R)
        else:
            res['r%g' % m] = endR
    # committed scale-out: half at 1R, stop to BE, runner to 2R
    if t_stop < BIG and t_stop <= t[1.0]:
        res['rSO'] = -1.0
    elif t[1.0] == BIG:
        res['rSO'] = endR
    else:
        runner = 2.0 if (t[2.0] < BIG and t[2.0] <= t_be) else (0.0 if t_be < BIG else endR)
        res['rSO'] = 0.5 + 0.5 * runner - (SLIP * e / R)
    res['Rpct'] = R / e * 100
    return res


rows = []
for sym in UNIVERSE:
    p = f'research/smc/data/{sym}_5m_ext.parquet'
    if not os.path.exists(p):
        continue
    df = pd.read_parquet(p).between_time('09:30', '15:59')
    if len(df) < 5000:
        continue
    S = signals(df)
    if S.empty:
        continue
    dates = df.index.normalize()
    day_start = pd.Series(np.arange(len(df)), index=df.index).groupby(dates).min()
    day_end = pd.Series(np.arange(len(df)), index=df.index).groupby(dates).max()
    for _, s in S.iterrows():
        i0 = int(s['i'])
        r = resolve(df, i0, s['entry'], s['stop'], int(s['dir']))
        if r is None:
            continue
        ts = df.index[i0]
        day = ts.normalize()
        # matched control: same day, same direction, same stop DISTANCE
        lo, hi = int(day_start[day]), int(day_end[day])
        base = dict(sym=sym, ts=ts, day=day.date(), kind=s['kind'], dir=int(s['dir']),
                    inzone=s['inzone'], **r)
        if hi - lo > 5:
            j = int(rng.integers(lo, hi))
            ce = df.close.values[j]
            dist = abs(s['entry'] - s['stop'])
            cstop = ce - dist * int(s['dir'])
            cr = resolve(df, j, ce, cstop, int(s['dir']))
            if cr:
                for k in ['r1', 'r2', 'r3', 'rSO']:
                    base['c_' + k] = cr[k]
        rows.append(base)

T = pd.DataFrame(rows)
T['year'] = pd.to_datetime(T.ts).dt.year
T.to_csv('research/smc/amn_signals.csv', index=False)
print(f'signals {len(T)}  |  tickers {T.sym.nunique()}  |  '
      f'{T.ts.min():%Y-%m-%d} -> {T.ts.max():%Y-%m-%d}')
print(T.groupby(['kind', 'dir']).size().to_string())
