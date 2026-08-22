"""
Equal-lows sweep, standalone -- no BOS/CHoCH required.

Signal: a cluster of 2+ confirmed swing lows inside a tight band forms a pool.
A bar wicks BELOW the pool and CLOSES back above it -> enter at that close, stop
under the sweep wick. The stop is the sweep bar itself, not the protected swing,
so R is a fraction of what the break-entry versions had to risk.

Discipline: parameters are chosen on 2022-2024 and validated on 2025-2026.
The split is reported for every variant, not just the winner.
"""
import numpy as np
import pandas as pd

exec(open('research/smc/smc_lab.py').read().split('# ---------- build the event table')[0])

POOL_LOOKBACK = 60      # 1H bars a swing low stays eligible for a pool (~2 weeks)
COOLDOWN = 6            # bars before the same pool may fire again
TRAIN_END = 2025        # < TRAIN_END is train, >= is holdout

atr_tr = np.maximum(hh - ll, np.maximum(np.abs(hh - np.roll(cc, 1)),
                                        np.abs(ll - np.roll(cc, 1))))
atr_tr[0] = hh[0] - ll[0]
ATR = pd.Series(atr_tr).rolling(14).mean().shift(1).values


def find_pools(band, min_touch, direction):
    """Yield (bar_i, level, touches, depth_atr, reclaim) for each sweep+reclaim."""
    pivot = pl1 if direction == 1 else ph1
    known = []                     # (idx, price) of confirmed swings
    last_fire = {}
    out = []
    for i in range(n1):
        j = i - L
        if j >= 0 and not np.isnan(pivot[j]):
            known.append((j, pivot[j]))
        known = [(k, p) for k, p in known if k >= i - POOL_LOOKBACK]
        if len(known) < min_touch or np.isnan(ATR[i]):
            continue
        # build clusters
        prices = sorted(p for _, p in known)
        pools = []
        for a in range(len(prices)):
            grp = [p for p in prices if abs(p - prices[a]) / prices[a] <= band]
            if len(grp) >= min_touch:
                lvl = min(grp) if direction == 1 else max(grp)
                pools.append((round(lvl, 2), len(grp)))
        for lvl, touches in set(pools):
            if i - last_fire.get(lvl, -999) < COOLDOWN:
                continue
            if direction == 1:
                hit = ll[i] < lvl and cc[i] > lvl
                depth = (lvl - ll[i]) / ATR[i] if hit else 0
                reclaim = (cc[i] - lvl) / ATR[i] if hit else 0
            else:
                hit = hh[i] > lvl and cc[i] < lvl
                depth = (hh[i] - lvl) / ATR[i] if hit else 0
                reclaim = (lvl - cc[i]) / ATR[i] if hit else 0
            if hit:
                last_fire[lvl] = i
                out.append((i, lvl, touches, depth, reclaim))
    return out


def build(band, min_touch, direction):
    rows = []
    for i, lvl, touches, depth, reclaim in find_pools(band, min_touch, direction):
        if i + 1 >= n1:
            continue
        ts = h1.index[i]
        if direction == 1:
            entry = cc[i] + SLIP
            stop = float(ll[i]) - BUF
        else:
            entry = cc[i] - SLIP
            stop = float(hh[i]) + BUF
        s = simulate(ts + pd.Timedelta(minutes=60), entry, stop, direction)
        if s is None:
            continue
        bw = bias_at(ts, w1_close, tr_1w)
        bd = bias_at(ts, d1_close, tr_1d)
        b4 = bias_at(ts, h4_close, tr_4h)
        rows.append(dict(ts=ts, i=i, dir=direction, level=lvl, touches=touches,
                         depth=depth, reclaim=reclaim, bw=bw, bd=bd, b4=b4,
                         w_ok=int(bw == direction),
                         stacked=int(bw == direction and bd == direction and b4 == direction),
                         Rpct=s['R'] / entry * 100, mfe=s['mfe'], mae=s['mae'],
                         w1=s['w1'], w2=s['w2'], w3=s['w3'], stopped=s['stopped'],
                         r_1r=s['r_1r'], r_2r=s['r_2r'], r_so=s['r_so']))
    d = pd.DataFrame(rows)
    if len(d):
        d['y'] = d.ts.dt.year
    return d


def boot(x, seed=17):
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()


def dd(x):
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(x) else 0.0


HDR = (f"{'variant':<40}{'n':>5}{'/yr':>5}{'hit1R':>7}{'hit2R':>7}"
       f"{'exp2R':>8}{'expSO':>8}{'R%px':>7}   {'train exp2R':>12}{'holdout':>10}")


def row(g, label, quiet=False):
    if len(g) < 10:
        return None
    tr_ = g[g.y < TRAIN_END].r_2r.values
    ho = g[g.y >= TRAIN_END].r_2r.values
    if not quiet:
        print(f"{label:<40}{len(g):>5}{len(g)/4.55:>5.0f}{g.w1.mean()*100:6.0f}%"
              f"{g.w2.mean()*100:6.0f}%{g.r_2r.mean():>8.3f}{g.r_so.mean():>8.3f}"
              f"{g.Rpct.median():>7.2f}   {tr_.mean():>12.3f}"
              f"{(ho.mean() if len(ho) else np.nan):>10.3f}")
    return g


# ---------------------------------------------------------------------------
print("=" * 118)
print("A  Equal-lows sweep alone (long).  Entry = reclaim close.  Stop = under the sweep wick.")
print("=" * 118)
print(HDR)
LONG = {}
for band in (0.0010, 0.0015, 0.0025):
    for touch in (2, 3):
        g = build(band, touch, 1)
        LONG[(band, touch)] = g
        row(g, f"band {band*100:.2f}%  {touch}+ touches")

print("\n" + "=" * 118)
print("B  Equal-highs sweep alone (short)")
print("=" * 118)
print(HDR)
SHORT = {}
for band in (0.0010, 0.0015, 0.0025):
    for touch in (2, 3):
        g = build(band, touch, -1)
        SHORT[(band, touch)] = g
        row(g, f"band {band*100:.2f}%  {touch}+ touches")

# ---------------------------------------------------------------------------
G = LONG[(0.0015, 2)]
print("\n" + "=" * 118)
print("C  Base variant (band 0.15%, 2+ touches, long) sliced")
print("=" * 118)
print(HDR)
row(G, "all")
row(G[G.w_ok == 1], "1W bullish")
row(G[G.w_ok != 1], "1W not bullish")
row(G[G.stacked == 1], "4H+1D+1W stacked")
row(G[G.depth >= 0.25], "wick >= 0.25 ATR below the pool")
row(G[G.depth < 0.25], "shallow poke (< 0.25 ATR)")
row(G[G.reclaim >= 0.5], "closes >= 0.5 ATR back above")
row(G[(G.w_ok == 1) & (G.depth >= 0.25)], "1W bullish + real flush")
row(G[(G.w_ok == 1) & (G.reclaim >= 0.5)], "1W bullish + strong reclaim")
row(G[G.touches >= 3], "3+ lows in the pool")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("D  Does the BOS add anything, or just lag?  (same sweeps, entry deferred)")
print("=" * 118)
ev_i = {int(e['i']): e['dir'] for _, e in ev_1h.iterrows()}
comp = []
for _, r in G.iterrows():
    i = int(r['i'])
    brk = next((k for k in range(i + 1, min(i + 13, n1)) if ev_i.get(k) == 1), None)
    if brk is None:
        continue
    entry = cc[brk] + SLIP
    stop = float(ll[i]) - BUF                       # same stop: the sweep wick
    s = simulate(h1.index[brk] + pd.Timedelta(minutes=60), entry, stop, 1)
    if s is None:
        continue
    comp.append(dict(ts=r['ts'], y=r['y'], w_ok=r['w_ok'], lag=brk - i,
                     Rpct=s['R'] / entry * 100, mfe=s['mfe'], w1=s['w1'], w2=s['w2'],
                     w3=s['w3'], stopped=s['stopped'], r_1r=s['r_1r'],
                     r_2r=s['r_2r'], r_so=s['r_so']))
C = pd.DataFrame(comp)
print(HDR)
paired = G[G.ts.isin(C.ts)]
row(paired, "enter at the sweep reclaim")
row(C, "wait for the BOS, same stop")
if len(C):
    print(f"\nmedian lag to the break: {C.lag.median():.0f} hours   "
          f"sweeps that never produced one within 12h: {1 - len(C)/len(G):.0%}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 118)
print("E  Robustness of the base long variant")
print("=" * 118)
for lbl, g in (("all", G), ("1W bullish", G[G.w_ok == 1])):
    x = g.r_2r.values
    lo, hi, p = boot(x)
    print(f"\n{lbl}:  n={len(g)}  exp2R={x.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(>0)={p:.2f}  totR={x.sum():+.1f}  maxDD={dd(x):.1f}  "
          f"ret/DD={x.sum()/dd(x) if dd(x) else float('nan'):.2f}")
    print(g.groupby('y').agg(n=('r_2r', 'size'), hit1R=('w1', 'mean'),
                             hit2R=('w2', 'mean'), exp2R=('r_2r', 'mean'),
                             totR=('r_2r', 'sum'), expSO=('r_so', 'mean')).round(2).to_string())

print("\n" + "=" * 118)
print("F  Benchmark check -- is any of this better than a random entry?")
print("=" * 118)
B = pd.read_csv('research/smc/baseline.csv')
bl = B[B.dir == 1]
print(f"{'random hour, stack bullish':<40}{len(bl):>5}{'':>5}{bl.w1.mean()*100:6.0f}%"
      f"{bl.w2.mean()*100:6.0f}%{bl.r_2r.mean():>8.3f}{bl.r_so.mean():>8.3f}{bl.Rpct.median():>7.2f}")
rng = np.random.default_rng(23)
for lbl, g in (("EQL sweep, all", G), ("EQL sweep, 1W bullish", G[G.w_ok == 1])):
    a, c = g.w1.values.astype(float), bl.w1.values.astype(float)
    d = (rng.choice(a, (10000, len(a)), replace=True).mean(1)
         - rng.choice(c, (10000, len(c)), replace=True).mean(1))
    print(f"{lbl}: hit1R {a.mean():.1%} vs {c.mean():.1%}  diff {a.mean()-c.mean():+.1%}  "
          f"95% CI [{np.percentile(d,2.5):+.1%}, {np.percentile(d,97.5):+.1%}]  P(better)={(d>0).mean():.2f}")

G.to_csv('research/smc/eql_signals.csv', index=False)
