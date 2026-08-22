"""
Two follow-ups to smc_lab.py:

A) Mechanics-free: what does SPY DO after a 1H BOS/CHoCH? Forward returns in bps
   at +1h/+4h/+1d/+3d, against the unconditional drift over the same horizons.
B) The entry the method actually prescribes: don't buy the break, buy the
   retracement into the impulse leg (0.5 / 0.618 / 0.786), stop under the sweep.
"""
import numpy as np
import pandas as pd

exec(open('research/smc/smc_lab.py').read().split('# ---------- build the event table')[0])

RETRACE = [0.5, 0.618, 0.786]
VALID_BARS = 12          # 1H bars a resting limit stays live (~2 sessions)

# ---------------------------------------------------------------------------
# A) forward returns
# ---------------------------------------------------------------------------
HOR = {'+1h': 1, '+4h': 4, '+1d': 7, '+3d': 20}   # in 1H RTH bars
rows = []
for _, e in ev_1h.iterrows():
    i = int(e['i'])
    d = int(e['dir'])
    if i + max(HOR.values()) >= n1:
        continue
    ts = e['ts']
    b4, bd, bw = (bias_at(ts, h4_close, tr_4h), bias_at(ts, d1_close, tr_1d),
                  bias_at(ts, w1_close, tr_1w))
    win = slice(max(0, i - SWEEP_LOOKBACK + 1), i + 1)
    swept = bool(sweep_dn[win].any()) if d == 1 else bool(sweep_up[win].any())
    r = dict(ts=ts, i=i, dir=d, kind=e['kind'], bw=bw, bd=bd, b4=b4,
             stacked=int(b4 == d and bd == d and bw == d), swept=int(swept))
    for k, h in HOR.items():
        r[k] = (cc[i + h] / cc[i] - 1) * 1e4 * d      # bps, signed to the event
    rows.append(r)
F = pd.DataFrame(rows)

uncond = {k: (cc[h:] / cc[:-h] - 1).mean() * 1e4 for k, h in HOR.items()}
print("=" * 100)
print("A  Forward move AFTER the event, in bps, signed in the event's direction")
print("=" * 100)
print(f"{'unconditional 1H drift (long)':<44} " +
      "  ".join(f"{k}={v:+7.1f}" for k, v in uncond.items()))
print("-" * 100)


def frow(df, label):
    if len(df) < 5:
        return
    cells = "  ".join(f"{k}={df[k].mean():+7.1f}" for k in HOR)
    pos = "  ".join(f"{df[k].gt(0).mean()*100:3.0f}%" for k in HOR)
    print(f"{label:<44} {cells}   n={len(df):<4} up%: {pos}")


for d, tag in ((1, 'LONG'), (-1, 'SHORT')):
    g = F[F.dir == d]
    frow(g[g.kind == 'BOS'], f"{tag} BOS   any HTF")
    frow(g[g.kind == 'CHoCH'], f"{tag} CHoCH any HTF")
    frow(g[(g.kind == 'BOS') & (g.stacked == 1)], f"{tag} BOS   HTF stacked")
    frow(g[(g.kind == 'CHoCH') & (g.stacked == 1)], f"{tag} CHoCH HTF stacked")
    frow(g[(g.kind == 'CHoCH') & (g.stacked == 1) & (g.swept == 1)],
         f"{tag} CHoCH HTF stacked + SWEPT")
    print("-" * 100)

# ---------------------------------------------------------------------------
# B) retracement entry
# ---------------------------------------------------------------------------
h1_idx = h1.index
out = []
for _, e in ev_1h.iterrows():
    i = int(e['i'])
    d = int(e['dir'])
    if i + 1 >= n1:
        continue
    ts = e['ts']
    b4, bd, bw = (bias_at(ts, h4_close, tr_4h), bias_at(ts, d1_close, tr_1d),
                  bias_at(ts, w1_close, tr_1w))
    win = slice(max(0, i - SWEEP_LOOKBACK + 1), i + 1)
    swept = bool(sweep_dn[win].any()) if d == 1 else bool(sweep_up[win].any())
    prot_i = int(e['prot_i']) if e['prot_i'] == e['prot_i'] and e['prot_i'] >= 0 else i
    lo_leg = float(ll[prot_i:i + 1].min())
    hi_leg = float(hh[prot_i:i + 1].max())
    if hi_leg <= lo_leg:
        continue

    for f in RETRACE:
        lvl = hi_leg - f * (hi_leg - lo_leg) if d == 1 else lo_leg + f * (hi_leg - lo_leg)
        # find the first 1H bar within VALID_BARS that trades through the limit
        fill_i = None
        for k in range(i + 1, min(i + 1 + VALID_BARS, n1)):
            if (d == 1 and ll[k] <= lvl) or (d == -1 and hh[k] >= lvl):
                fill_i = k
                break
            # invalidated: leg low/high taken out before the fill
            if (d == 1 and cc[k] < lo_leg) or (d == -1 and cc[k] > hi_leg):
                break
        if fill_i is None:
            out.append(dict(ts=ts, dir=d, kind=e['kind'], f=f, bw=bw, bd=bd, b4=b4,
                            stacked=int(b4 == d and bd == d and bw == d),
                            swept=int(swept), filled=0))
            continue
        entry = lvl + (SLIP if d == 1 else -SLIP)
        stop = (lo_leg - BUF) if d == 1 else (hi_leg + BUF)
        sim = simulate(h1_idx[fill_i] + pd.Timedelta(minutes=60), entry, stop, d)
        if sim is None:
            continue
        out.append(dict(ts=ts, dir=d, kind=e['kind'], f=f, bw=bw, bd=bd, b4=b4,
                        stacked=int(b4 == d and bd == d and bw == d),
                        swept=int(swept), filled=1,
                        Rpct=sim['R'] / entry * 100, mfe=sim['mfe'],
                        w1=sim['w1'], w2=sim['w2'], w3=sim['w3'],
                        stopped=sim['stopped'], r_1r=sim['r_1r'],
                        r_2r=sim['r_2r'], r_so=sim['r_so']))

Rt = pd.DataFrame(out)
Rt.to_csv('research/smc/retrace.csv', index=False)

print("\n" + "=" * 112)
print("B  Buy the RETRACEMENT into the leg instead of the break close  (long side)")
print("=" * 112)
print(f"{'setup':<46}{'fill%':>7}{'n':>6}{'hit1R':>7}{'hit2R':>7}{'stop%':>7}"
      f"{'exp1R':>8}{'exp2R':>8}{'expSO':>8}{'R%px':>7}")


def line(df, label):
    if len(df) == 0:
        return
    fills = df[df.filled == 1]
    if len(fills) < 5:
        return
    print(f"{label:<46}{df.filled.mean()*100:6.0f}%{len(fills):>6}"
          f"{fills.w1.mean()*100:6.0f}%{fills.w2.mean()*100:6.0f}%"
          f"{fills.stopped.mean()*100:6.0f}%{fills.r_1r.mean():>8.3f}"
          f"{fills.r_2r.mean():>8.3f}{fills.r_so.mean():>8.3f}"
          f"{fills.Rpct.median():>7.2f}")


LG = Rt[Rt.dir == 1]
for f in RETRACE:
    g = LG[LG.f == f]
    print(f"--- retrace to {f:.3f} of the leg " + "-" * 74)
    line(g[g.kind == 'CHoCH'], "CHoCH  any HTF")
    line(g[(g.kind == 'CHoCH') & (g.bw == 1)], "CHoCH  1W bullish")
    line(g[(g.kind == 'CHoCH') & (g.bw == 1) & (g.swept == 1)], "CHoCH  1W bullish + SWEPT")
    line(g[(g.kind == 'CHoCH') & (g.stacked == 1) & (g.swept == 1)], "CHoCH  4H+1D+1W stacked + SWEPT")
    line(g[g.kind == 'BOS'], "BOS    any HTF")
    line(g[(g.kind == 'BOS') & (g.bw == 1)], "BOS    1W bullish")
    line(g[(g.kind == 'BOS') & (g.bw == 1) & (g.swept == 1)], "BOS    1W bullish + SWEPT")

print("\n" + "=" * 112)
print("B2 Short side, retrace to 0.618")
print("=" * 112)
SH = Rt[(Rt.dir == -1) & (Rt.f == 0.618)]
line(SH[SH.kind == 'CHoCH'], "CHoCH  any HTF")
line(SH[(SH.kind == 'CHoCH') & (SH.bw == -1)], "CHoCH  1W bearish")
line(SH[(SH.kind == 'CHoCH') & (SH.bw == -1) & (SH.swept == 1)], "CHoCH  1W bearish + SWEPT")
line(SH[SH.kind == 'BOS'], "BOS    any HTF")
line(SH[(SH.kind == 'BOS') & (SH.bw == -1)], "BOS    1W bearish")
