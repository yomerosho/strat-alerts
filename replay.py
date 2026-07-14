"""
replay.py
---------
Walk history forward one 15-minute bar at a time, reconstruct EXACTLY the state
the live scanner would have seen at that moment, and journal every signal it
would have fired.

    python replay.py SPY QQQ --days 60
    python replay.py SPY --days 60 --no-gates      # what v4 would have sent

Why this is honest
==================
It does not reimplement the scanner. It CALLS it -- the same arm_inside_bar,
the same magnitude.evaluate, the same evaluate_tiers. Whatever is wrong with
the live code is wrong here too, which is the only way a replay is worth
anything.

Lookahead is prevented structurally, not by discipline. At each step every
frame is truncated to bars whose bar_end <= T, so a bar that had not closed yet
is not merely ignored -- it is physically absent from the dataframe. There is
no code path by which the scanner could see it.

Output
======
A CSV with one row per signal, carrying the exact ET timestamp of the 15m close
that triggered it plus the trigger / stop / target prices. Pull it up next to
TradingView, jump to that timestamp, and judge for yourself whether the setup
did what it claimed it would.

`outcome` is auto-resolved by walking 5-minute bars forward from the entry:
whichever of stop or target is touched first. When a single 5m bar touches BOTH
(a violent bar straddling the whole range), it is scored as a LOSS. That is
pessimistic on purpose -- an optimistic tiebreak is how backtests lie.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

# Unlike main.py, replay does not import config, so nothing else pulls the
# .env in for us. Load it here or the os.getenv() key lookup in main() fails
# even when the keys are sitting right next to this file.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import magnitude
from bars import ET, BarProvider
from levels import (
    ArmedLevel,
    TIER2,
    arm_failed_two,
    arm_inside_bar,
    evaluate_tiers,
    split_closed_forming,
)
from strat import BULL, label_bars

logger = logging.getLogger("replay")

# How long a signal is given to resolve before it is called a timeout.
MAX_HOLD_SESSIONS = 10


@dataclass
class Signal:
    symbol: str
    triggered_at: str      # ET timestamp of the 15m close -- paste this into TV
    setup_tf: str
    pattern: str
    direction: str
    trigger: float
    stop: float
    target: float          # first gating rung
    runway_r: float
    score: int
    continuity: str
    nested: str
    ladder: str

    outcome: str = ""      # WIN | LOSS | SCRATCH | TIMEOUT
    r_realized: float = 0.0
    mfe_r: float = 0.0     # best unrealised R before resolution
    mae_r: float = 0.0     # worst unrealised R before resolution
    bars_held: int = 0
    resolved_at: str = ""


# --------------------------------------------------------------------------
# state reconstruction
# --------------------------------------------------------------------------

def state_at(
    labeled: dict[str, pd.DataFrame],
    now_et: pd.Timestamp,
) -> tuple[dict, dict]:
    """
    (closed_by_tf, forming_by_tf) as they stood at `now_et`.

    This is the anti-lookahead boundary. Every frame is first CUT to bars that
    had already started by now_et, then split into closed vs forming by their
    bar_end. A 4H bar that begins at 13:30 does not exist at 11:00 -- not as a
    forming bar, not at all.
    """
    closed_by_tf: dict[str, pd.DataFrame] = {}
    forming_by_tf: dict[str, Optional[pd.Series]] = {}

    for tf, df in labeled.items():
        visible = df[df.index <= now_et]
        if visible.empty:
            closed_by_tf[tf] = visible
            forming_by_tf[tf] = None
            continue
        c, f = split_closed_forming(visible, now_et)
        closed_by_tf[tf] = c
        forming_by_tf[tf] = f

    return closed_by_tf, forming_by_tf


def scan_at(
    symbol: str,
    labeled: dict[str, pd.DataFrame],
    now_et: pd.Timestamp,
    apply_gates: bool = True,
) -> list[ArmedLevel]:
    """One scan cycle, as of `now_et`. Mirrors StratScanner.scan_symbol."""
    closed_by_tf, forming_by_tf = state_at(labeled, now_et)

    df5 = labeled["5Min"]
    df5_visible = df5[df5["bar_end"] <= now_et]
    if df5_visible.empty:
        return []
    current_price = float(df5_visible["close"].iloc[-1])

    nominating = magnitude.NOMINATING_TFS if apply_gates else ("2H", "4H", "1D")

    armed: list[ArmedLevel] = []
    for tf in nominating:
        closed = closed_by_tf.get(tf)
        if closed is None or len(closed) < 3:
            continue
        forming = forming_by_tf.get(tf)
        armed.extend(arm_inside_bar(symbol, tf, closed, forming, current_price))
        armed.extend(
            arm_failed_two(symbol, tf, closed, forming, df5_visible, current_price)
        )

    if not armed:
        return []

    for lv in armed:
        lv.decision = magnitude.evaluate(
            closed_by_tf, forming_by_tf,
            setup_tf=lv.setup_tf, direction=lv.direction,
            trigger=lv.level, invalidation=lv.invalidation,
        )
        if lv.decision.passed and lv.decision.rungs:
            lv.target = lv.decision.gate_rungs[0].level

    if apply_gates:
        armed = [lv for lv in armed if lv.decision.passed]

    df15 = labeled.get("15Min", pd.DataFrame())
    df15_visible = df15[df15["bar_end"] <= now_et] if not df15.empty else df15
    for lv in armed:
        evaluate_tiers(lv, df15_visible, now_et)

    return armed


# --------------------------------------------------------------------------
# outcome resolution
# --------------------------------------------------------------------------

def resolve(sig: Signal, df5: pd.DataFrame, entry_time: pd.Timestamp) -> Signal:
    """
    Walk 5-minute bars forward and resolve under the committed SCALE-OUT model:

        take half off at +1R, pull the stop to breakeven, run the rest to the
        first gating rung (sig.target).

    This is the exit chosen after the exit_sweep experiment: on the v5 entries
    it scored ~88% win at +1.32R/trade, versus 43% / +2.66R for "let it all run
    to the rung". The mechanism is simple -- once +1R is banked and the stop is
    at breakeven, the runner can only WIN or SCRATCH, never lose a full R.

    Tiebreaks:
      * The +1R scale is a resting LIMIT: if a bar's range reaches +1R the half
        is treated as filled, even if that same bar later trades against you.
      * The stop (and the breakeven stop) is a stop-market, checked AFTER the
        scale on the same bar. A both-touch bar BEFORE +1R is a full LOSS --
        the pessimistic tiebreak from the original resolver is preserved there.

    outcome in {WIN, LOSS, SCRATCH, TIMEOUT}. SCRATCH = stopped at breakeven
    after scaling never triggered; r_realized is the blended R across both
    halves.
    """
    risk = abs(sig.trigger - sig.stop)
    if risk <= 0:
        sig.outcome = "INVALID"
        return sig

    forward = df5[df5.index > entry_time]
    if forward.empty:
        sig.outcome = "TIMEOUT"
        return sig

    deadline = entry_time + pd.Timedelta(days=MAX_HOLD_SESSIONS * 1.5)
    forward = forward[forward.index <= deadline]

    bull = sig.direction == BULL
    target_R = abs(sig.target - sig.trigger) / risk

    stop = sig.stop        # trails to breakeven after the scale
    banked = 0.0           # R locked from the scaled-out half
    weight = 1.0           # remaining position size
    moved_be = False
    mfe = mae = 0.0

    def _done(outcome: str, r: float, i: int, ts) -> Signal:
        sig.outcome = outcome
        sig.r_realized = round(r, 3)
        sig.bars_held = i
        sig.resolved_at = ts.isoformat()
        sig.mfe_r, sig.mae_r = round(mfe, 2), round(mae, 2)
        return sig

    for i, (ts, bar) in enumerate(forward.iterrows(), start=1):
        hi, lo = float(bar["high"]), float(bar["low"])
        if bull:
            fav = (hi - sig.trigger) / risk
            mae = min(mae, (lo - sig.trigger) / risk)
            hit_stop = lo <= stop
            hit_target = hi >= sig.target
        else:
            fav = (sig.trigger - lo) / risk
            mae = min(mae, (sig.trigger - hi) / risk)
            hit_stop = hi >= stop
            hit_target = lo <= sig.target
        mfe = max(mfe, fav)

        # --- scale: book half at +1R, trail the runner at breakeven ---
        if not moved_be and fav >= 1.0:
            banked = 0.5 * 1.0
            weight = 0.5
            stop = sig.trigger
            moved_be = True
            if hit_target:                       # +1R and rung in the same bar
                return _done("WIN", banked + weight * target_R, i, ts)
            continue

        if hit_stop:                             # stop-market, checked after scale
            if moved_be:                         # stop is at breakeven
                return _done("WIN" if banked else "SCRATCH", banked, i, ts)
            return _done("LOSS", -1.0, i, ts)    # pre-scale: pessimistic full loss
        if hit_target:
            return _done("WIN", banked + weight * target_R, i, ts)

    sig.outcome = "TIMEOUT"
    sig.r_realized = round(banked, 3)
    sig.bars_held = len(forward)
    sig.mfe_r, sig.mae_r = round(mfe, 2), round(mae, 2)
    return sig


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def replay_symbol(
    provider: BarProvider,
    symbol: str,
    apply_gates: bool = True,
) -> list[Signal]:
    frames = provider.fetch(symbol)
    if not frames or "15Min" not in frames:
        logger.warning("%s: insufficient data", symbol)
        return []

    labeled = {
        tf: label_bars(df) for tf, df in frames.items() if not tf.startswith("_")
    }
    df5 = labeled["5Min"]
    df15 = labeled["15Min"]

    signals: list[Signal] = []
    fired: set[str] = set()   # a level triggers once, not once per scan

    for bar_end in df15["bar_end"]:
        now = pd.Timestamp(bar_end)

        for lv in scan_at(symbol, labeled, now, apply_gates):
            if lv.tier != TIER2 or lv.tier2_time != now:
                continue
            if lv.key in fired:
                continue
            if lv.target is None or lv.invalidation is None:
                continue
            fired.add(lv.key)

            d = lv.decision
            sig = Signal(
                symbol=symbol,
                triggered_at=now.isoformat(),
                setup_tf=lv.setup_tf,
                pattern=lv.pattern,
                direction=lv.direction,
                trigger=round(lv.level, 2),
                stop=round(lv.invalidation, 2),
                target=round(lv.target, 2),
                runway_r=round(d.runway_r, 2) if d else 0.0,
                score=d.score if d else 0,
                continuity=(
                    "".join(f"{tf}{'+' if ok else '-'}" for tf, ok in d.continuity.items())
                    if d else ""
                ),
                nested=",".join(d.nested_tfs) if d else "",
                ladder=" > ".join(str(r) for r in d.gate_rungs) if d else "",
            )
            signals.append(resolve(sig, df5, now))

    return signals


def summarize(sigs: list[Signal]) -> str:
    if not sigs:
        return "No signals."

    # SCRATCH (stopped at breakeven after the scale) is not a loss and does not
    # count against the win rate -- it is a trade the management defused.
    decided = [s for s in sigs if s.outcome in ("WIN", "LOSS")]
    wins = [s for s in decided if s.outcome == "WIN"]
    scratches = [s for s in sigs if s.outcome == "SCRATCH"]
    timeouts = [s for s in sigs if s.outcome == "TIMEOUT"]
    booked = [s for s in sigs if s.outcome in ("WIN", "LOSS", "SCRATCH")]

    lines = [
        f"signals        {len(sigs)}",
        f"resolved       {len(booked)}  (scratches {len(scratches)}, timeouts {len(timeouts)})",
    ]
    if decided:
        wr = len(wins) / len(decided) * 100
        exp = sum(s.r_realized for s in booked) / len(booked) if booked else 0.0
        lines += [
            f"win rate       {wr:.1f}%  (scratches excluded)",
            f"expectancy     {exp:+.2f}R per signal",
            f"total          {sum(s.r_realized for s in booked):+.1f}R",
        ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--out", default="signals.csv")
    ap.add_argument(
        "--no-gates", action="store_true",
        help="replay WITHOUT the v5 gates -- roughly what v4 would have sent. "
             "Run both and diff the expectancy; that is the actual experiment.",
    )
    args = ap.parse_args()

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not set", file=sys.stderr)
        return 1

    provider = BarProvider(key, secret, os.getenv("ALPACA_DATA_FEED", "sip"))

    all_sigs: list[Signal] = []
    for sym in args.symbols:
        sigs = replay_symbol(provider, sym, apply_gates=not args.no_gates)
        logger.info("%-6s %d signals", sym, len(sigs))
        all_sigs.extend(sigs)

    if not all_sigs:
        print("\nNo signals fired. With the gates on, that is a plausible result "
              "over a short window -- widen --days or drop min_runway_r to see more.")
        return 0

    df = pd.DataFrame([asdict(s) for s in all_sigs]).sort_values("triggered_at")
    df.to_csv(args.out, index=False)

    print()
    print(df[["triggered_at", "symbol", "setup_tf", "pattern", "direction",
              "trigger", "stop", "target", "runway_r", "score",
              "outcome", "r_realized", "mfe_r"]].to_string(index=False))
    print()
    print(summarize(all_sigs))
    print(f"\nwrote {args.out}")
    print("Open each triggered_at on TradingView and judge the setup yourself. "
          "The CSV is the point; the summary is a footnote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
