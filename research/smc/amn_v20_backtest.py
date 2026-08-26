"""
AMN point-6 (sweep + reclaim) -- backtest of the CURRENT v20 logic, plus the
FTFC question.

Port matches pine/amn_5point.pine v20 exactly:
  * ATR zigzag, legATR 1.5, minBars 2   (minBars kills the same-bar swings)
  * sequences may NOT share pivots      (new P1 must be at/after previous P5)
  * point 6 = sweep of the point-4 level, then a close back through it
  * entry  = close of the reclaim bar   (that is when the alert fires)
  * stop   = the sweep extreme

FTFC ("full timeframe continuity", The Strat sense): a timeframe is UP when the
CURRENT, still-forming bar on that timeframe has close > open. Computed from the
open of the higher-TF bar containing the entry bar, so it is knowable live.

Gates tested:
  ftfc3 : 5m + 15m + 1h all agree with the trade direction
  ftfc5 : ftfc3 + 4h + 1D
  zone  : point 4 tapped a 1H demand/supply zone
"""
import os
import numpy as np
import pandas as pd

LEG_ATR, MIN_BARS, MAX_WAIT = 1.5, 2, 120
HOLD_BARS = 78 * 3
SLIP = 0.0002
TARGETS = [1.0, 2.0, 3.0]
UNIVERSE = ['SPY', 'QQQ', 'IWM', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT',
            'NVDA', 'PLTR', 'TSLA', 'NFLX', 'INTC', 'QCOM', 'ORCL']
rng = np.random.default_rng(20260826)


def rma_atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(tr), np.nan)
    a = tr[:n].mean(); out[n - 1] = a
    for i in range(n, len(tr)):
        a = (a * (n - 1) + tr[i]) / n; out[i] = a
    return out


def htf_zones(df):
    """Two-deep 1H order blocks, as f_htfZones() now does."""
    h1 = df.resample('60min').agg({'open': 'first', 'high': 'max',
                                   'low': 'min', 'close': 'last'}).dropna()
    o, h, l, c = h1.open.values, h1.high.values, h1.low.values, h1.close.values
    d1T = d1B = d2T = d2B = s1T = s1B = s2T = s2B = np.nan
    rows = []
    for i in range(len(h1)):
        if i >= 1:
            if c[i - 1] < o[i - 1] and c[i] > h[i - 1]:
                d2T, d2B = d1T, d1B
                d1T, d1B = h[i - 1], l[i - 1]
            if c[i - 1] > o[i - 1] and c[i] < l[i - 1]:
                s2T, s2B = s1T, s1B
                s1T, s1B = h[i - 1], l[i - 1]
        rows.append((d1T, d1B, d2T, d2B, s1T, s1B, s2T, s2B))
    z = pd.DataFrame(rows, index=h1.index,
                     columns=['d1T', 'd1B', 'd2T', 'd2B', 's1T', 's1B', 's2T', 's2B'])
    z.index = z.index + pd.Timedelta(minutes=60)     # known only after the bar closes
    return z.reindex(df.index, method='ffill')


def ftfc_dirs(df):
    """Direction of the in-progress bar on each higher timeframe, per 5m bar."""
    out = {}
    # TradingView anchors intraday HTF bars to the SESSION open (09:30), not to
    # clock hours. 15m coincides either way; 1h and 4h do not, and the two
    # anchorings give opposite FTFC verdicts on real signals.
    for name, rule, off in [('m15','15min','0min'), ('h1','60min','30min'),
                            ('h4','240min','30min'), ('d1','1D','0min')]:
        op = df.open.resample(rule, offset=off).first().reindex(df.index, method='ffill')
        out[name] = np.sign(df.close.values - op.values)
    out['m5'] = np.sign(df.close.values - df.open.values)
    return out


def signals(df):
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(c)
    A = rma_atr(h, l, c)
    Z = htf_zones(df)
    F = ftfc_dirs(df)
    zc = {k: Z[k].values for k in Z.columns}

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
    aDir = 0; aLvl = aInv = np.nan; aBar = -1
    aSwept = False; aSwExt = np.nan
    aP5Bar = None; lastEnd = None; aInZone = False; aZDist = np.nan
    out = []

    for i in range(n):
        ml = LEG_ATR * A[i] if not np.isnan(A[i]) else np.nan
        if zdir == 0 and not np.isnan(ml):
            zdir, zext, zbar = 1, h[i], i
        if zdir == 1:
            if h[i] >= zext: zext, zbar = h[i], i
            if not np.isnan(ml) and i - zbar >= MIN_BARS and l[i] < zext - ml:
                push(zext, zbar, 1); zdir, zext, zbar = -1, l[i], i
        elif zdir == -1:
            if l[i] <= zext: zext, zbar = l[i], i
            if not np.isnan(ml) and i - zbar >= MIN_BARS and h[i] > zext + ml:
                push(zext, zbar, -1); zdir, zext, zbar = 1, h[i], i

        seq = 0
        if len(tps) >= 5:
            t5 = tps[-1]
            if tps[-2] == -t5 and tps[-3] == t5 and tps[-4] == -t5 and tps[-5] == t5:
                v1, v2, v3, v4, v5 = pxs[-5], pxs[-4], pxs[-3], pxs[-2], pxs[-1]
                if t5 == 1 and v3 > v1 and v5 > v3 and v4 > v2: seq = 1
                if t5 == -1 and v3 < v1 and v5 < v3 and v4 < v2: seq = -1

        p5b = bis[-1] if bis else None
        p1b = bis[-5] if len(bis) >= 5 else None
        fresh = (seq != 0 and (aP5Bar is None or p5b != aP5Bar)
                 and (lastEnd is None or (p1b is not None and p1b >= lastEnd)))

        if fresh:
            aP5Bar = p5b; lastEnd = p5b
            aDir, aBar, aSwept, aSwExt = seq, i, False, np.nan
            aLvl, aInv = pxs[-2], pxs[-4]
            if seq == 1:
                z1 = (zc['d1T'][i], zc['d1B'][i]); z2 = (zc['d2T'][i], zc['d2B'][i])
            else:
                z1 = (zc['s1T'][i], zc['s1B'][i]); z2 = (zc['s2T'][i], zc['s2B'][i])
            aInZone = any((not np.isnan(t)) and (b <= aLvl <= t) for t, b in (z1, z2))
            # how FAR point 4 is from the nearest zone, in ATR. 0 = inside.
            # A binary in/out test on a crude detector can hide a real proximity
            # effect, so measure the distance and let the data speak.
            _best = np.inf
            for _t, _b in (z1, z2):
                if np.isnan(_t):
                    continue
                if _b <= aLvl <= _t:
                    _best = 0.0
                    break
                _best = min(_best, min(abs(aLvl - _t), abs(aLvl - _b)))
            aZDist = (_best / A[i]) if (np.isfinite(_best) and A[i] > 0) else np.nan
            lag = i - bis[-1]
            for k in range(0, min(lag, 300) + 1):
                j = i - k
                if j < 0: break
                if (l[j] < aLvl) if seq == 1 else (h[j] > aLvl):
                    aSwept = True
                    e = l[j] if seq == 1 else h[j]
                    aSwExt = e if np.isnan(aSwExt) else (min(aSwExt, e) if seq == 1 else max(aSwExt, e))

        if aDir != 0:
            if aDir == 1 and l[i] < aLvl:
                aSwept = True
                aSwExt = l[i] if np.isnan(aSwExt) else min(aSwExt, l[i])
            if aDir == -1 and h[i] > aLvl:
                aSwept = True
                aSwExt = h[i] if np.isnan(aSwExt) else max(aSwExt, h[i])
            if aSwept and ((c[i] > aLvl) if aDir == 1 else (c[i] < aLvl)):
                d = aDir
                out.append(dict(i=i, dir=d, entry=c[i], stop=aSwExt, inzone=aInZone,
                                zdist=aZDist,
                                m5=F['m5'][i], m15=F['m15'][i], h1=F['h1'][i],
                                h4=F['h4'][i], d1=F['d1'][i]))
                aDir, aSwept = 0, False
            elif ((c[i] < aInv) if aDir == 1 else (c[i] > aInv)) or (i - aBar > MAX_WAIT):
                aDir, aSwept = 0, False
    return pd.DataFrame(out)


def resolve(df, i0, entry, stop, d):
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(c)
    e = entry * (1 + SLIP * d)
    R = (e - stop) if d == 1 else (stop - e)
    if not np.isfinite(R) or R <= 0:
        return None
    end = min(i0 + 1 + HOLD_BARS, n)
    BIG = 10 ** 9
    t_stop = BIG; t = {m: BIG for m in TARGETS}; t_be = BIG
    lastc = e
    for k in range(i0 + 1, end):
        up = ((h[k] - e) / R) if d == 1 else ((e - l[k]) / R)
        dn = ((l[k] - e) / R) if d == 1 else ((e - h[k]) / R)
        lastc = c[k]
        if dn <= -1.0 and t_stop == BIG: t_stop = k
        for m in TARGETS:
            if up >= m and t[m] == BIG: t[m] = k
        if t[1.0] < BIG and k > t[1.0] and dn <= 0.0 and t_be == BIG: t_be = k
    endR = (((lastc - e) / R) if d == 1 else ((e - lastc) / R)) - SLIP * e / R
    res = {}
    # best favourable excursion BEFORE the stop was hit - what a discretionary
    # trader could actually have taken off the table
    mfe = 0.0
    for k in range(i0 + 1, min(t_stop + 1, end)):
        u = ((h[k] - e) / R) if d == 1 else ((e - l[k]) / R)
        mfe = max(mfe, u)
    res['mfe'] = mfe
    for m in TARGETS:
        res['r%g' % m] = -1.0 if (t_stop < BIG and t_stop <= t[m]) else \
            (m - SLIP * e / R if t[m] < BIG else endR)
    if t_stop < BIG and t_stop <= t[1.0]:
        res['rSO'] = -1.0
    elif t[1.0] == BIG:
        res['rSO'] = endR
    else:
        runner = 2.0 if (t[2.0] < BIG and t[2.0] <= t_be) else (0.0 if t_be < BIG else endR)
        res['rSO'] = 0.5 + 0.5 * runner - SLIP * e / R
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
    ds = pd.Series(np.arange(len(df)), index=df.index).groupby(dates).min()
    de = pd.Series(np.arange(len(df)), index=df.index).groupby(dates).max()
    for _, s in S.iterrows():
        i0 = int(s['i']); d = int(s['dir'])
        r = resolve(df, i0, s['entry'], s['stop'], d)
        if r is None:
            continue
        ts = df.index[i0]; day = ts.normalize()
        base = dict(sym=sym, ts=ts, day=day.date(), dir=d, inzone=bool(s['inzone']),
                    zdist=float(s['zdist']),
                    ftfc3=int(s['m5'] == d and s['m15'] == d and s['h1'] == d),
                    ftfc5=int(s['m5'] == d and s['m15'] == d and s['h1'] == d
                              and s['h4'] == d and s['d1'] == d),
                    agree=int((s['m5'] == d) + (s['m15'] == d) + (s['h1'] == d)
                              + (s['h4'] == d) + (s['d1'] == d)),
                    m5=int(s['m5']), m15=int(s['m15']), h1=int(s['h1']),
                    h4=int(s['h4']), d1=int(s['d1']), **r)
        lo, hi = int(ds[day]), int(de[day])
        if hi - lo > 5:
            j = int(rng.integers(lo, hi))
            ce = df.close.values[j]
            cstop = ce - abs(s['entry'] - s['stop']) * d
            cr = resolve(df, j, ce, cstop, d)
            if cr:
                for k in ['r1', 'r2', 'r3', 'rSO']:
                    base['c_' + k] = cr[k]
        rows.append(base)

T = pd.DataFrame(rows)
T['year'] = pd.to_datetime(T.ts).dt.year
T.to_csv('research/smc/amn_v20.csv', index=False)
print(f'point-6 entries: {len(T)}   tickers {T.sym.nunique()}   '
      f'{T.ts.min():%Y-%m} -> {T.ts.max():%Y-%m}')
sess = T.groupby('sym').day.nunique().mean()
print(f'about {len(T)/T.sym.nunique()/1150:.2f} signals per ticker per session')
print(T.groupby('dir').size().to_string())
