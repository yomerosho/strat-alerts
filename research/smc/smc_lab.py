"""
SMC structure lab: liquidity sweeps + BOS/CHoCH on SPY 1H, gated by HTF bias.

Data: 5-min RTH bars (Alpaca SIP, raw) 2022-01 -> 2026-07, resampled up.
No lookahead: pivots are only "known" L bars after they print; HTF bias at a
1H event uses only HTF bars that CLOSED before that 1H bar closed.
Trade path resolution uses the 5-min series so 1R/2R vs stop ordering is real,
not an intrabar guess.
"""
import numpy as np
import pandas as pd

L = 2                 # fractal strength (pivot confirmed L bars later)
BUF = 0.05            # stop buffer, $
SLIP = 0.02           # slippage, $
MAX_HOLD_BARS = 78 * 5            # 5 RTH sessions of 5-min bars
SWEEP_LOOKBACK = 6    # 1H bars: how recent a sweep must be to "count"

# ---------- data ----------
df5 = pd.read_parquet('research/.lab_cache/SPY_5Min_2022-01-01.parquet')
df5 = df5.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
df5 = df5.set_index('ts').sort_index()[['open', 'high', 'low', 'close', 'volume']]

AGG = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

h1 = df5.resample('60min', offset='30min', label='left', closed='left').agg(AGG).dropna()
h4 = df5.resample('240min', offset='30min', label='left', closed='left').agg(AGG).dropna()
d1 = df5.resample('1D', label='left', closed='left').agg(AGG).dropna()
w1 = df5.resample('W-MON', label='left', closed='left').agg(AGG).dropna()


# ---------- structure engine ----------
def pivots(h, l, L):
    """Fractal pivots: strict on the left, >= on the right."""
    n = len(h)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(L, n - L):
        if h[i] == h[i - L:i + L + 1].max() and (h[i] > h[i - L:i]).all():
            ph[i] = h[i]
        if l[i] == l[i - L:i + L + 1].min() and (l[i] < l[i - L:i]).all():
            pl[i] = l[i]
    return ph, pl


def structure(df, L=L):
    """State machine -> per-bar trend + event list. Events fire on bar CLOSE."""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    n = len(df)
    ph, pl = pivots(h, l, L)
    trend = np.zeros(n, dtype=int)
    ref_h = ref_l = np.nan
    ref_h_i = ref_l_i = -1
    t = 0
    events = []
    for i in range(n):
        j = i - L                      # a pivot at j becomes known only now
        if j >= 0:
            if not np.isnan(ph[j]):
                ref_h, ref_h_i = ph[j], j
            if not np.isnan(pl[j]):
                ref_l, ref_l_i = pl[j], j
        if not np.isnan(ref_h) and c[i] > ref_h:
            events.append(dict(i=i, ts=df.index[i], dir=1,
                               kind='BOS' if t == 1 else 'CHoCH',
                               level=ref_h, prot=ref_l, prot_i=ref_l_i, prev_trend=t))
            t = 1
            ref_h, ref_h_i = np.nan, -1
        elif not np.isnan(ref_l) and c[i] < ref_l:
            events.append(dict(i=i, ts=df.index[i], dir=-1,
                               kind='BOS' if t == -1 else 'CHoCH',
                               level=ref_l, prot=ref_h, prot_i=ref_h_i, prev_trend=t))
            t = -1
            ref_l, ref_l_i = np.nan, -1
        trend[i] = t
    return trend, pd.DataFrame(events), ph, pl


tr_1h, ev_1h, ph1, pl1 = structure(h1)
tr_4h, _, _, _ = structure(h4)
tr_1d, _, _, _ = structure(d1)
tr_1w, _, _, _ = structure(w1)

# bar-close timestamps for HTF (index is bar OPEN)
h4_close = h4.index + pd.Timedelta(minutes=240)
d1_close = d1.index + pd.Timedelta(days=1)
w1_close = w1.index + pd.Timedelta(days=7)


def bias_at(ts, closes, trend):
    pos = closes.searchsorted(ts, side='right')
    return int(trend[pos - 1]) if pos > 0 else 0


# ---------- sweep detection on 1H ----------
hh, ll, cc = h1['high'].values, h1['low'].values, h1['close'].values
n1 = len(h1)
sweep_dn = np.zeros(n1, dtype=bool)   # sell-side liquidity taken -> bullish
sweep_up = np.zeros(n1, dtype=bool)
sweep_dn_lo = np.full(n1, np.nan)
sweep_up_hi = np.full(n1, np.nan)
known_pl, known_ph = [], []
for i in range(n1):
    j = i - L
    if j >= 0:
        if not np.isnan(pl1[j]):
            known_pl.append(pl1[j])
        if not np.isnan(ph1[j]):
            known_ph.append(ph1[j])
    for lvl in known_pl[-8:]:
        if ll[i] < lvl and cc[i] > lvl:
            sweep_dn[i] = True
            sweep_dn_lo[i] = ll[i]
    for lvl in known_ph[-8:]:
        if hh[i] > lvl and cc[i] < lvl:
            sweep_up[i] = True
            sweep_up_hi[i] = hh[i]

five_h = df5['high'].values
five_l = df5['low'].values
five_c = df5['close'].values


BIG = 10**9


def simulate(entry_ts, entry, stop, direction, max_bars=MAX_HOLD_BARS):
    """Walk 5-min bars, recording first-touch bar index for each R level.

    Full path is walked (no early break) so a break-even-after-1R rule can be
    resolved. Intrabar ties are given to the adverse level.
    """
    start = df5.index.searchsorted(entry_ts, side='left')
    end = min(start + max_bars, len(df5))
    R = abs(entry - stop)
    if start >= len(df5) or start >= end or R <= 0:
        return None
    t_stop = t1 = t2 = t3 = BIG
    t_be = BIG                       # first return to entry AFTER +1R is tagged
    mfe = mae = 0.0
    endR = 0.0
    for k in range(start, end):
        hi, lo = five_h[k], five_l[k]
        if direction == 1:
            up, dn = (hi - entry) / R, (lo - entry) / R
            endR = (five_c[k] - entry) / R
        else:
            up, dn = (entry - lo) / R, (entry - hi) / R
            endR = (entry - five_c[k]) / R
        mfe = max(mfe, up)
        mae = min(mae, dn)
        if dn <= -1.0 and t_stop == BIG:
            t_stop = k
        if up >= 1.0 and t1 == BIG:
            t1 = k
        if up >= 2.0 and t2 == BIG:
            t2 = k
        if up >= 3.0 and t3 == BIG:
            t3 = k
        if t1 < BIG and k > t1 and dn <= 0.0 and t_be == BIG:
            t_be = k
    # ---- exit models ----
    # A: flat 1R target
    stopped_first_1 = t_stop < BIG and t_stop <= t1
    r_1r = -1.0 if stopped_first_1 else (1.0 if t1 < BIG else endR)
    # B: flat 2R target
    stopped_first_2 = t_stop < BIG and t_stop <= t2
    r_2r = -1.0 if stopped_first_2 else (2.0 if t2 < BIG else endR)
    # C: committed scale-out -- half at +1R, stop to BE, runner to +2R
    if stopped_first_1:
        r_so = -1.0
    else:
        runner = 2.0 if (t2 < BIG and t2 <= t_be) else (0.0 if t_be < BIG else endR)
        if t1 == BIG:
            r_so = endR
        else:
            r_so = 0.5 + 0.5 * runner
    return dict(R=R, mfe=mfe, mae=mae,
                w1=t1 < BIG and t_stop > t1, w2=t2 < BIG and t_stop > t2,
                w3=t3 < BIG and t_stop > t3, stopped=t_stop < BIG,
                r_1r=r_1r, r_2r=r_2r, r_so=r_so)


# ---------- build the event table ----------
rows = []
for _, e in ev_1h.iterrows():
    i = int(e['i'])
    if i + 1 >= n1:
        continue
    ts = e['ts']
    d = int(e['dir'])
    b4 = bias_at(ts, h4_close, tr_4h)
    bd = bias_at(ts, d1_close, tr_1d)
    bw = bias_at(ts, w1_close, tr_1w)
    win = slice(max(0, i - SWEEP_LOOKBACK + 1), i + 1)
    swept = bool(sweep_dn[win].any()) if d == 1 else bool(sweep_up[win].any())

    prot_i = int(e['prot_i']) if e['prot_i'] == e['prot_i'] and e['prot_i'] >= 0 else i
    if d == 1:
        stop = float(np.nanmin(ll[prot_i:i + 1])) - BUF
        entry = cc[i] + SLIP
    else:
        stop = float(np.nanmax(hh[prot_i:i + 1])) + BUF
        entry = cc[i] - SLIP

    sim = simulate(ts + pd.Timedelta(minutes=60), entry, stop, d)
    if sim is None:
        continue
    rows.append(dict(ts=ts, i=i, dir=d, kind=e['kind'], b4=b4, bd=bd, bw=bw,
                     stacked=int(b4 == d and bd == d and bw == d),
                     against=int(b4 == -d and bd == -d and bw == -d),
                     swept=int(swept), entry=entry, stop=stop,
                     Rdollar=sim['R'], Rpct=sim['R'] / entry * 100,
                     mfe=sim['mfe'], mae=sim['mae'],
                     w1=sim['w1'], w2=sim['w2'], w3=sim['w3'], stopped=sim['stopped'],
                     r_1r=sim['r_1r'], r_2r=sim['r_2r'], r_so=sim['r_so']))

T = pd.DataFrame(rows)
T.to_csv('research/smc/events.csv', index=False)
print(f"1H bars {len(h1)}  4H {len(h4)}  1D {len(d1)}  1W {len(w1)}")
print(f"events: {len(T)}   span {T.ts.min()} -> {T.ts.max()}")
print(T.groupby(['dir', 'kind']).size().to_string())
print("\nstacked-with-trend events:", int(T.stacked.sum()),
      " of which swept:", int(T[T['stacked'] == 1].swept.sum()))

# ---------- baseline: long ANY 1H close while the HTF stack is bullish ----------
# Same stop logic (5-bar swing low) and same exit models. If the structure event
# does not beat this, the event is decoration.
b4_all = np.array([bias_at(t, h4_close, tr_4h) for t in h1.index])
bd_all = np.array([bias_at(t, d1_close, tr_1d) for t in h1.index])
bw_all = np.array([bias_at(t, w1_close, tr_1w) for t in h1.index])
base = []
for i in range(5, n1 - 1):
    for d in (1, -1):
        if not (b4_all[i] == d and bd_all[i] == d and bw_all[i] == d):
            continue
        if d == 1:
            stop = float(ll[i - 4:i + 1].min()) - BUF
            entry = cc[i] + SLIP
        else:
            stop = float(hh[i - 4:i + 1].max()) + BUF
            entry = cc[i] - SLIP
        sim = simulate(h1.index[i] + pd.Timedelta(minutes=60), entry, stop, d)
        if sim is None:
            continue
        base.append(dict(ts=h1.index[i], dir=d, Rpct=sim['R'] / entry * 100,
                         mfe=sim['mfe'], w1=sim['w1'], w2=sim['w2'],
                         r_1r=sim['r_1r'], r_2r=sim['r_2r'], r_so=sim['r_so']))
B = pd.DataFrame(base)
B.to_csv('research/smc/baseline.csv', index=False)
print(f"baseline entries: {len(B)}  (long {int((B.dir==1).sum())} / short {int((B.dir==-1).sum())})")
