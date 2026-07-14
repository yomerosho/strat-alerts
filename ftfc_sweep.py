"""
ftfc_sweep.py
-------------
Does Full Timeframe Continuity actually earn its cut of the trades?

Same v5 entries, committed scale-out exit. The ONLY thing that varies is a
Full-Timeframe-Continuity gate: require at least k of the watched frames
(1H / 2H / 4H / 1D / 1W) to be trading in the trade's direction at entry.

    python ftfc_sweep.py                 # full default watchlist
    python ftfc_sweep.py SPY QQQ NVDA    # specific symbols

For each threshold it reports win rate, expectancy, and -- the number that
keeps us honest -- how many trades survive. A gate that lifts win rate by
starving you of trades has not helped; it has just shrunk the sample until the
noise looks like signal.

2D is deliberately absent: the system builds 1H/2H/4H/1D/1W cleanly and has no
2-day bar. Add it only if continuity here proves worth the build.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import exit_sweep
import magnitude
import replay
from bars import BarProvider
from config import load_tickers
from strat import BULL, label_bars

FTFC_TFS = ["1H", "2H", "4H", "1D", "1W"]


def collect(provider: BarProvider, symbol: str):
    """v5 TIER2 signals + each one's FTFC alignment count at entry."""
    frames = provider.fetch(symbol)
    if not frames or "15Min" not in frames:
        return [], None
    labeled = {tf: label_bars(df) for tf, df in frames.items() if not tf.startswith("_")}
    df5 = labeled["5Min"]
    df15 = labeled["15Min"]

    out = []
    fired = set()
    for bar_end in df15["bar_end"]:
        now = pd.Timestamp(bar_end)
        # min_ftfc=0: we sweep FTFC ourselves below, so we must see every
        # signal that cleared the OTHER gates, pre-FTFC.
        for lv in replay.scan_at(symbol, labeled, now, apply_gates=True, min_ftfc=0):
            if lv.tier != replay.TIER2 or lv.tier2_time != now:
                continue
            if lv.key in fired or lv.target is None or lv.invalidation is None:
                continue
            fired.add(lv.key)

            # FTFC at the moment of entry: forming bar agreeing with direction.
            closed_by_tf, forming_by_tf = replay.state_at(labeled, now)
            aligned = total = 0
            for tf in FTFC_TFS:
                c = closed_by_tf.get(tf)
                if c is None or c.empty:
                    continue
                total += 1
                aligned += int(magnitude.bar_agrees(forming_by_tf.get(tf), c, lv.direction))

            out.append({
                "symbol": symbol, "entry": now, "bull": lv.direction == BULL,
                "trigger": float(lv.level), "stop0": float(lv.invalidation),
                "rung": float(lv.target), "aligned": aligned, "total": total,
                # The ACTUAL fill: the 15m close that confirmed. It sits through
                # the trigger, so measuring R from here (not the trigger) is the
                # honest version -- no free head-start toward +1R.
                "entry_price": float(lv.current_price),
            })
    return out, df5


def resolve_honest(s, df5, target_r=None):
    """
    Scale-out resolved from the REAL entry fill, not the trigger.

    risk = fill -> stop (what you actually risk). +1R scale and breakeven are
    measured from the fill. This strips out the trigger-vs-fill head start that
    flatters the as-built numbers. Returns (outcome, r).

    target_r: fixed R runner target measured from the fill. None -> the gating
    rung. A tighter target trades R for a higher hit rate.
    """
    entry, stop, bull, rung = s["entry_price"], s["stop0"], s["bull"], s["rung"]
    risk = abs(entry - stop)
    if risk <= 0:
        return "INVALID", 0.0
    if target_r is not None:
        target = entry + target_r * risk if bull else entry - target_r * risk
    else:
        target = rung
    target_R = abs(target - entry) / risk
    if target_R <= 0:            # target already at/behind the fill -> no runner
        return "INVALID", 0.0

    fwd = df5[df5.index > s["entry"]]
    deadline = s["entry"] + pd.Timedelta(days=exit_sweep.MAX_HOLD * 1.5)
    fwd = fwd[fwd.index <= deadline]
    if fwd.empty:
        return "TIMEOUT", 0.0

    cur_stop, banked, weight, moved_be = stop, 0.0, 1.0, False
    for _, bar in fwd.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        fav = (hi - entry) / risk if bull else (entry - lo) / risk
        hit_stop = (lo <= cur_stop) if bull else (hi >= cur_stop)
        hit_tgt = (hi >= target) if bull else (lo <= target)

        if not moved_be and fav >= 1.0:
            banked, weight, cur_stop, moved_be = 0.5, 0.5, entry, True
            if hit_tgt:
                return "WIN", banked + weight * target_R
            continue
        if hit_stop:
            if moved_be:
                return ("WIN" if banked else "SCRATCH"), banked
            return "LOSS", -1.0
        if hit_tgt:
            return "WIN", banked + weight * target_R
    return "TIMEOUT", banked


def combo(sigs, df5_by, min_aligned, target_r):
    sel = [s for s in sigs if s["aligned"] >= min_aligned]
    outs = [resolve_honest(s, df5_by[s["symbol"]], target_r=target_r) for s in sel]
    wins = [r for o, r in outs if o == "WIN"]
    losses = [r for o, r in outs if o == "LOSS"]
    booked = [r for o, r in outs if o in ("WIN", "LOSS", "SCRATCH")]
    decided = len(wins) + len(losses)
    return {
        "n": len(sel), "win%": 100 * len(wins) / decided if decided else 0.0,
        "exp_R": sum(booked) / len(booked) if booked else 0.0,
        "total_R": sum(booked),
    }


def stats(sigs, df5_by, min_aligned, honest=False):
    sel = [s for s in sigs if s["aligned"] >= min_aligned]
    if honest:
        outs = [resolve_honest(s, df5_by[s["symbol"]]) for s in sel]
    else:
        outs = [exit_sweep.resolve(s, df5_by[s["symbol"]], scale=True)[:2] for s in sel]
    wins = [r for o, r in outs if o == "WIN"]
    losses = [r for o, r in outs if o == "LOSS"]
    booked = [r for o, r in outs if o in ("WIN", "LOSS", "SCRATCH")]
    decided = len(wins) + len(losses)
    return {
        "k": min_aligned, "n": len(sel),
        "win%": 100 * len(wins) / decided if decided else 0.0,
        "W": len(wins), "L": len(losses),
        "exp_R": sum(booked) / len(booked) if booked else 0.0,
        "total_R": sum(booked),
    }


def main() -> int:
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set", file=sys.stderr)
        return 1
    provider = BarProvider(key, secret, os.getenv("ALPACA_DATA_FEED", "sip"))

    symbols = sys.argv[1:] or load_tickers()
    sigs, df5_by = [], {}
    for sym in symbols:
        s, df5 = collect(provider, sym)
        if df5 is not None:
            sigs.extend(s)
            df5_by[sym] = df5
        print(f"{sym:<6} {len(s)} v5 signals")

    if not sigs:
        print("no signals")
        return 0

    # distribution of alignment, so we can see WHERE trades actually live
    from collections import Counter
    dist = Counter(s["aligned"] for s in sigs)
    print(f"\n{len(sigs)} v5 entries across {len(df5_by)} symbols")
    print("FTFC alignment (of 5 frames 1H/2H/4H/1D/1W) at entry:")
    for a in range(6):
        print(f"  {a}/5 aligned: {dist.get(a,0)}")

    for honest in (False, True):
        title = ("R from ACTUAL ENTRY FILL (honest -- no head start)" if honest
                 else "R from trigger (as-built)")
        print(f"\nFTFC gate sweep -- {title}:")
        hdr = f"{'require >=k aligned':<22}{'n':>5}{'win%':>8}{'W':>5}{'L':>5}{'exp_R':>8}{'total_R':>9}"
        print(hdr)
        print("-" * len(hdr))
        for k in range(0, 6):
            r = stats(sigs, df5_by, k, honest=honest)
            label = "no FTFC gate (all)" if k == 0 else f">= {k}/5"
            print(f"{label:<22}{r['n']:>5}{r['win%']:>7.1f}%{r['W']:>5}{r['L']:>5}"
                  f"{r['exp_R']:>+8.2f}{r['total_R']:>+9.1f}")

    # --- the path to 85%: FTFC gate x tighter runner target (all honest) ---
    print("\nPath to 85% -- FTFC gate x runner target (honest, R from fill):")
    hdr = f"{'config':<34}{'n':>5}{'win%':>8}{'exp_R':>8}{'total_R':>9}"
    print(hdr)
    print("-" * len(hdr))
    combos = [
        ("no gate, runner->rung", 0, None),
        (">=4/5, runner->rung", 4, None),
        (">=4/5, runner 2R", 4, 2.0),
        (">=4/5, runner 1R", 4, 1.0),
        (">=5/5, runner->rung", 5, None),
        (">=5/5, runner 2R", 5, 2.0),
        (">=5/5, runner 1R", 5, 1.0),
    ]
    for label, k, tr in combos:
        r = combo(sigs, df5_by, k, tr)
        print(f"{label:<34}{r['n']:>5}{r['win%']:>7.1f}%{r['exp_R']:>+8.2f}{r['total_R']:>+9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
