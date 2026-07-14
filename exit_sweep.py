"""
exit_sweep.py
-------------
Same v5 entries. Different EXITS. One question: where does win rate actually
live, and what R do you pay for each point of it?

It reuses replay.scan_at -- identical scanner, identical anti-lookahead -- to
generate the exact v5 TIER2 signals once per symbol, caches the 5-minute tape,
then re-resolves every signal under a menu of exit policies. Nothing about the
entry changes between rows; only the management does.

    python exit_sweep.py SPY QQQ NVDA TSLA

Pessimism is preserved from replay.resolve: when a single 5m bar touches both
the stop and the target, it is scored against you. That makes every win-rate
number here a FLOOR, not a hope.
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

import replay
from bars import BarProvider
from strat import BULL, label_bars

MAX_HOLD = replay.MAX_HOLD_SESSIONS


def collect_signals(provider: BarProvider, symbol: str):
    """Every v5 TIER2 signal for `symbol`, plus the 5m tape to resolve on."""
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
        # min_ftfc=0: the exit experiment predates the FTFC gate and is judged
        # on the pre-FTFC entry set, so it must not be filtered by it.
        for lv in replay.scan_at(symbol, labeled, now, apply_gates=True, min_ftfc=0):
            if lv.tier != replay.TIER2 or lv.tier2_time != now:
                continue
            if lv.key in fired or lv.target is None or lv.invalidation is None:
                continue
            fired.add(lv.key)
            out.append({
                "symbol": symbol,
                "entry": now,
                "bull": lv.direction == BULL,
                "trigger": float(lv.level),
                "stop0": float(lv.invalidation),        # original 1R stop
                "rung": float(lv.target),               # first gating rung
            })
    return out, df5


def resolve(sig, df5, *, target_r=None, breakeven_r=None, scale=False):
    """
    Walk 5m bars forward. Returns realised R.

    target_r   : fixed R target. None -> use the gating rung.
    breakeven_r: once price runs this far in R, stop jumps to entry (BE).
    scale      : book HALF at +1R and move stop to BE; runner exits at target.

    outcome in {WIN, LOSS, SCRATCH, TIMEOUT}. A SCRATCH is a breakeven exit --
    it is not a loss, which is the entire point of stop management.
    """
    trig, bull = sig["trigger"], sig["bull"]
    risk = abs(trig - sig["stop0"])
    if risk <= 0:
        return "INVALID", 0.0, 0.0

    if target_r is not None:
        target = trig + target_r * risk if bull else trig - target_r * risk
    else:
        target = sig["rung"]
    target_R = abs(target - trig) / risk

    fwd = df5[df5.index > sig["entry"]]
    deadline = sig["entry"] + pd.Timedelta(days=MAX_HOLD * 1.5)
    fwd = fwd[fwd.index <= deadline]
    if fwd.empty:
        return "TIMEOUT", 0.0, 0.0

    stop = sig["stop0"]
    banked = 0.0          # R already locked from a scaled-out half
    weight = 1.0          # remaining position size
    moved_be = False
    mfe = 0.0

    for ts, bar in fwd.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        fav = (hi - trig) / risk if bull else (trig - lo) / risk
        mfe = max(mfe, fav)

        hit_stop = (lo <= stop) if bull else (hi >= stop)
        hit_tgt = (hi >= target) if bull else (lo <= target)

        # --- scale: book half at +1R, then trail the rest at BE ---
        if scale and not moved_be and fav >= 1.0:
            banked = 0.5 * 1.0        # half the position, +1R
            weight = 0.5
            stop = trig               # BE on the runner
            moved_be = True
            # a bar can reach +1R AND target in the same candle
            if hit_tgt:
                return "WIN", banked + weight * target_R, round(mfe, 2)
            continue

        # --- plain breakeven management ---
        if breakeven_r is not None and not moved_be and fav >= breakeven_r:
            stop = trig
            moved_be = True

        if hit_stop:                  # pessimistic: stop checked first
            r = banked + weight * ((stop - trig) / risk if bull else (trig - stop) / risk)
            if moved_be and abs(stop - trig) < 1e-9:
                return ("SCRATCH" if banked == 0 else "WIN"), round(r, 3), round(mfe, 2)
            return "LOSS", round(r, 3), round(mfe, 2)
        if hit_tgt:
            return "WIN", round(banked + weight * target_R, 3), round(mfe, 2)

    return "TIMEOUT", round(banked, 3), round(mfe, 2)


def run(sigs, df5_by_sym, label, **kw):
    outs = [resolve(s, df5_by_sym[s["symbol"]], **kw) for s in sigs]
    wins = [r for o, r, _ in outs if o == "WIN"]
    losses = [r for o, r, _ in outs if o == "LOSS"]
    scratches = [o for o, _, _ in outs if o == "SCRATCH"]
    timeouts = [o for o, _, _ in outs if o == "TIMEOUT"]
    decided = len(wins) + len(losses)                 # scratches excluded from win%
    wr = 100 * len(wins) / decided if decided else 0.0
    booked = [r for o, r, _ in outs if o in ("WIN", "LOSS", "SCRATCH")]
    exp = sum(booked) / len(booked) if booked else 0.0
    return {
        "policy": label, "n": len(outs), "win%": wr,
        "W": len(wins), "L": len(losses), "scr": len(scratches), "to": len(timeouts),
        "exp_R": exp, "total_R": sum(booked),
    }


def main() -> int:
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set", file=sys.stderr)
        return 1
    provider = BarProvider(key, secret, os.getenv("ALPACA_DATA_FEED", "sip"))

    sigs, df5_by_sym = [], {}
    for sym in sys.argv[1:] or ["SPY", "QQQ", "NVDA", "TSLA"]:
        s, df5 = collect_signals(provider, sym)
        if df5 is not None:
            sigs.extend(s)
            df5_by_sym[sym] = df5
        print(f"{sym:<6} {len(s)} v5 signals")

    policies = [
        ("baseline: rung target, fixed stop", dict()),
        ("fixed target 3R", dict(target_r=3.0)),
        ("fixed target 2R", dict(target_r=2.0)),
        ("fixed target 1R (scalp)", dict(target_r=1.0)),
        ("fixed target 0.5R (tight scalp)", dict(target_r=0.5)),
        ("breakeven @1R, rung target", dict(breakeven_r=1.0)),
        ("breakeven @0.5R, rung target", dict(breakeven_r=0.5)),
        ("SCALE: half @1R + BE, runner to rung", dict(scale=True)),
    ]

    rows = [run(sigs, df5_by_sym, lbl, **kw) for lbl, kw in policies]

    print(f"\n{len(sigs)} v5 entries, re-resolved under each exit policy")
    print(f"(win% excludes scratches; exp_R is per signal incl. scratches as 0R)\n")
    hdr = f"{'policy':<40}{'n':>4}{'win%':>7}{'W':>5}{'L':>5}{'scr':>5}{'exp_R':>8}{'total_R':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['policy']:<40}{r['n']:>4}{r['win%']:>6.1f}%{r['W']:>5}{r['L']:>5}"
              f"{r['scr']:>5}{r['exp_R']:>+8.2f}{r['total_R']:>+9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
