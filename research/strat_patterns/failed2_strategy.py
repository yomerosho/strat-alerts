"""
Failed-2 depth strategy build-out.

Signal (P6 long shown; P7 short is the mirror):
  Day D undercuts yesterday's low while still <= yesterday's high (goes 2D),
  then reclaims. Depth = (yesterday_low - lowest_low_before_entry) / ATR14.

Triggers kept from the pattern study:
  T1: first 15m bar closing back above yesterday's low, entry at that close,
      no entries after 14:00 ET. Depth measured pre-entry (no lookahead).
  T3: daily 2D bar closes green in the top 40% of range; enter next open.
      Depth = full signal-day undercut.

Sweeps (in-sample only; OOS run once on the locked spec):
  A. depth threshold          0 / 0.10 / 0.25 / 0.35 / 0.50 ATR
  B. stop buffer below the failure low   0 / 0.10 / 0.25 / 0.50 ATR
  C. exit model:
       mag       stop/target = failure low / yesterday's high, 2-day time stop
       scaleout  half off at +1R, stop to breakeven, runner to yesterday's
                 high, same 2-day time stop (user's committed exit)
       hold1..3  stop active, otherwise exit at close of Nth day after entry

Conservative fill rules match backtest.py: any 5m bar touching both stop and
target counts as STOP; scale-out runner ignores a same-bar target fill when the
breakeven stop was also touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest import ENTRY_CUTOFF, IS_START, OOS_START, SYMBOLS, Tape, ns  # noqa: E402

OUT = Path(__file__).parent
ET = "America/New_York"
EVENTS = OUT / "f2_events.parquet"


# ------------------------------------------------------------- events -------

def build_events() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        tape = Tape(sym)
        D = tape.daily
        dates = D.index
        for di in range(1, len(D)):
            d = dates[di]
            if d < IS_START:
                continue
            p_hi = D["high"].iloc[di - 1]
            p_lo = D["low"].iloc[di - 1]
            atr = tape.atr14.iloc[di - 1]
            if not np.isfinite(atr) or atr <= 0:
                continue
            j0 = np.searchsorted(tape.t15, ns(d + pd.Timedelta(hours=9, minutes=30)))
            j1 = np.searchsorted(tape.t15, ns(d + pd.Timedelta(hours=16)))
            if j1 <= j0:
                continue
            cutoff = d + ENTRY_CUTOFF
            k_day0 = np.searchsorted(tape.t5, ns(d + pd.Timedelta(hours=9, minutes=30)))

            for side in ("L", "S"):
                pat = "P6" if side == "L" else "P7"
                # --- 2D/2U break with the outside-day guard
                run_hi, run_lo, brk_j = -np.inf, np.inf, -1
                for j in range(j0, j1):
                    if side == "L":
                        if run_hi > p_hi:
                            break
                        run_hi = max(run_hi, tape.h15[j])
                        if tape.l15[j] < p_lo:
                            if tape.h15[j] > p_hi:
                                break
                            brk_j = j
                            break
                    else:
                        if run_lo < p_lo:
                            break
                        run_lo = min(run_lo, tape.l15[j])
                        if tape.h15[j] > p_hi:
                            if tape.l15[j] < p_lo:
                                break
                            brk_j = j
                            break
                if brk_j < 0:
                    continue
                # --- T1 reclaim
                lvl = p_lo if side == "L" else p_hi
                for j in range(brk_j, j1):
                    end_j = pd.Timestamp(tape.end15[j], tz="UTC").tz_convert(ET)
                    if end_j > cutoff:
                        break
                    c = tape.c15[j]
                    if (side == "L" and c > lvl) or (side == "S" and c < lvl):
                        k_e = np.searchsorted(tape.t5, tape.end15[j])
                        if side == "L":
                            fail = tape.l5[k_day0:k_e].min()
                            depth = (p_lo - fail) / atr
                            tgt = p_hi
                            ok = c < tgt and c > fail
                        else:
                            fail = tape.h5[k_day0:k_e].max()
                            depth = (fail - p_hi) / atr
                            tgt = p_lo
                            ok = c > tgt and c < fail
                        if ok:
                            rows.append(dict(
                                symbol=sym, pattern=pat, side=side, trigger="T1",
                                date=d, di_entry=di, entry_time=end_j, entry=c,
                                fail=fail, target=tgt, atr=atr, depth=depth))
                        break

            # --- T3 (needs next day)
            lab = D["label"].iloc[di]
            o_, h_, l_, c_ = D[["open", "high", "low", "close"]].iloc[di]
            rng = h_ - l_
            if rng <= 0 or di + 1 >= len(D):
                continue
            pos = (c_ - l_) / rng
            for side, pat in (("L", "P6"), ("S", "P7")):
                if side == "L":
                    ok = lab == "2D" and c_ > o_ and pos >= 0.6
                    fail, tgt, depth = l_, p_hi, (p_lo - l_) / atr
                else:
                    ok = lab == "2U" and c_ < o_ and pos <= 0.4
                    fail, tgt, depth = h_, p_lo, (h_ - p_hi) / atr
                if not ok:
                    continue
                e_px = D["open"].iloc[di + 1]
                if (side == "L" and (e_px >= tgt or e_px <= fail)) or (
                    side == "S" and (e_px <= tgt or e_px >= fail)
                ):
                    continue
                rows.append(dict(
                    symbol=sym, pattern=pat, side=side, trigger="T3", date=d,
                    di_entry=di + 1,
                    entry_time=dates[di + 1] + pd.Timedelta(hours=9, minutes=30),
                    entry=e_px, fail=fail, target=tgt, atr=atr, depth=depth))
    ev = pd.DataFrame(rows)
    ev["sample"] = np.where(pd.to_datetime(ev.date).dt.tz_convert(ET) >= OOS_START,
                            "OOS", "IS")
    ev.to_parquet(EVENTS)
    return ev


# ---------------------------------------------------------- simulation ------

def simulate(tape: Tape, ev, buf: float, mode: str):
    """Return dict(r=..., outcome=...) or None if degenerate."""
    side, entry, atr, tgt = ev.side, ev.entry, ev.atr, ev.target
    sgn = 1.0 if side == "L" else -1.0
    stop = ev.fail - sgn * buf * atr
    risk = sgn * (entry - stop)
    if risk <= 0:
        return None
    D = tape.daily
    hold = {"mag": 2, "scaleout": 2, "hold1": 1, "hold2": 2, "hold3": 3}[mode]
    di_x = min(ev.di_entry + hold, len(D) - 1)
    t_end = D["bar_end"].iloc[di_x]
    k0 = np.searchsorted(tape.t5, ns(ev.entry_time))
    kend = min(np.searchsorted(tape.t5, ns(t_end)), len(tape.t5))
    one_r = entry + sgn * risk

    half_done = False
    stop_cur = stop
    for k in range(k0, kend):
        h, l = tape.h5[k], tape.l5[k]
        px_stop = (l <= stop_cur) if side == "L" else (h >= stop_cur)
        if mode in ("hold1", "hold2", "hold3"):
            if px_stop:
                return dict(r=-1.0, outcome="stop")
            continue
        px_tgt = (h >= tgt) if side == "L" else (l <= tgt)
        if mode == "mag":
            if px_stop:
                return dict(r=-1.0, outcome="stop")
            if px_tgt:
                return dict(r=sgn * (tgt - entry) / risk, outcome="target")
            continue
        # scaleout
        if not half_done:
            px_1r = (h >= one_r) if side == "L" else (l <= one_r)
            if px_stop:
                return dict(r=-1.0, outcome="stop")
            if px_1r:
                half_done = True
                stop_cur = entry
                # same-bar runner fill only if breakeven stop was untouched
                be_touch = (l <= entry) if side == "L" else (h >= entry)
                if px_tgt and not be_touch:
                    return dict(r=0.5 + 0.5 * sgn * (tgt - entry) / risk,
                                outcome="target")
            continue
        if px_stop:
            return dict(r=0.5, outcome="breakeven")
        if px_tgt:
            return dict(r=0.5 + 0.5 * sgn * (tgt - entry) / risk, outcome="target")
    # time exit at the daily close
    close = D["close"].iloc[di_x]
    r_open = sgn * (close - entry) / risk
    if mode == "scaleout" and half_done:
        return dict(r=0.5 + 0.5 * r_open, outcome="time_half")
    return dict(r=r_open, outcome="time")


def run(ev: pd.DataFrame, tapes: dict, buf: float, mode: str) -> pd.DataFrame:
    out = []
    for _, e in ev.iterrows():
        res = simulate(tapes[e.symbol], e, buf, mode)
        if res:
            out.append(dict(e, **res))
    return pd.DataFrame(out)


def metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return dict(n=0)
    wins = df.r[df.r > 0].sum()
    losses = -df.r[df.r < 0].sum()
    eq = df.sort_values("entry_time").r.cumsum()
    return dict(n=n, win=round((df.r > 0).mean() * 100, 1),
                avg_r=round(df.r.mean(), 3),
                pf=round(wins / losses, 2) if losses > 0 else np.inf,
                maxdd=round((eq - eq.cummax()).min(), 1))


# --------------------------------------------------------------- main -------

def main():
    ev = pd.read_parquet(EVENTS) if EVENTS.exists() else build_events()
    print(f"events: {len(ev)}  (IS {(ev['sample']=='IS').sum()}, "
          f"OOS {(ev['sample']=='OOS').sum()})")
    tapes = {s: Tape(s) for s in SYMBOLS}
    ev_is = ev[ev["sample"] == "IS"]

    print("\n=== A. depth threshold sweep (IS, buf=0, mag exit) ===")
    rows = []
    base = {}
    for (pat, trig), g in ev_is.groupby(["pattern", "trigger"]):
        base[(pat, trig)] = run(g, tapes, 0.0, "mag")
    for thr in (0.0, 0.10, 0.25, 0.35, 0.50):
        for (pat, trig), sim in base.items():
            m = metrics(sim[sim.depth >= thr])
            rows.append(dict(pattern=pat, trigger=trig, depth=thr, **m))
    print(pd.DataFrame(rows).sort_values(["pattern", "trigger", "depth"])
          .to_string(index=False))

    print("\n=== B. stop buffer sweep (IS, depth>=0.25, mag exit) ===")
    rows = []
    sel = ev_is[ev_is.depth >= 0.25]
    for buf in (0.0, 0.10, 0.25, 0.50):
        sim = run(sel, tapes, buf, "mag")
        for (pat, trig), g in sim.groupby(["pattern", "trigger"]):
            rows.append(dict(pattern=pat, trigger=trig, buf=buf, **metrics(g)))
    print(pd.DataFrame(rows).sort_values(["pattern", "trigger", "buf"])
          .to_string(index=False))

    print("\n=== C. exit model (IS, depth>=0.25, buf per B readout) ===")
    rows = []
    for buf in (0.0, 0.10):
        for mode in ("mag", "scaleout", "hold1", "hold2", "hold3"):
            sim = run(sel, tapes, buf, mode)
            for (pat, trig), g in sim.groupby(["pattern", "trigger"]):
                rows.append(dict(pattern=pat, trigger=trig, buf=buf, mode=mode,
                                 **metrics(g)))
    print(pd.DataFrame(rows).sort_values(["pattern", "trigger", "buf", "mode"])
          .to_string(index=False))


if __name__ == "__main__":
    main()
