"""
magnitude.py
------------
Open magnitude, continuity as a GATE, and the v5 alert gate stack.

The v4 problem
==============
v4 arms a level, computes a target on the SAME timeframe, and alerts. It never
asks the only question that separates a scalp from a swing:

    "Once price reaches my target, is there anything left above it?"

A 2H 2-1-2 targets the 2H prior bar's high. If the DAILY prior high sits 0.4%
above the trigger, price arrives at your target and the daily magnetism is
already spent -- there is nothing pulling it further. You have a scalp. If the
Daily AND Weekly prior highs are both still untouched and spread out, the same
pattern is a swing. Identical geometry, different trade.

Open magnitude
==============
    A higher timeframe's prior-bar extreme is UNTOUCHED if the currently
    FORMING bar on that timeframe has not yet traded through it.

Once the forming bar exceeds it, that rung is SPENT -- the timeframe has
already made its move and has no unfinished business in your direction.

Premarket falls out for free: at 08:00 there is no forming daily bar yet, so
the last closed daily bar is yesterday and its high is trivially untouched.
That is PDH, and it is exactly the right target. No special-casing needed.

Contract
========
This module consumes the SAME structures scanner.py already builds:

    closed_by_tf : dict[str, pd.DataFrame]      -- bars whose bar_end has passed
    forming_by_tf: dict[str, Optional[pd.Series]] -- the live bar, or None

which is why nothing here re-derives bars. Pass what you already have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from strat import BEAR, BULL

# Timeframes permitted to NOMINATE a level (Gate 1).
# 2H is deliberately absent. It confirms; it does not nominate.
NOMINATING_TFS: tuple[str, ...] = ("4H", "1D")

# Timeframes whose forming bar must AGREE with the trade (Gate 3).
# These two govern whether an overnight hold is survivable.
CONTINUITY_TFS: tuple[str, ...] = ("4H", "1D")

# Rungs of the magnitude ladder, ascending in scale.
LADDER_TFS: tuple[str, ...] = ("2H", "4H", "1D", "1W")

MIN_RUNWAY_R: float = 2.0
NEST_TOLERANCE_PCT: float = 0.002   # 0.2%
ALERT_BUDGET: int = 3


@dataclass(frozen=True)
class Rung:
    """One untouched higher-timeframe prior-bar extreme, priced in R."""
    timeframe: str
    level: float
    r_multiple: float

    def __str__(self) -> str:
        return f"{self.timeframe} {self.level:.2f} ({self.r_multiple:.1f}R)"


@dataclass
class Decision:
    """Gate-by-gate accounting, in the style of diagnose.py."""
    setup_tf: str = ""
    passed: bool = False
    score: int = 0
    rungs: list[Rung] = field(default_factory=list)
    continuity: dict[str, bool] = field(default_factory=dict)
    nested_tfs: list[str] = field(default_factory=list)
    compressed: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def gate_rungs(self) -> list[Rung]:
        """
        Rungs at or ABOVE the nominating timeframe. These are the ones that gate.

        A 2H prior high is a speed bump for a 4H setup, not a wall -- price
        pauses there and carries on. Gating on it would kill everything, because
        there is nearly always a 2H extreme sitting within a fraction of an R.
        The levels that actually STOP a 4H move are 4H, Daily and Weekly.

        Lower rungs are still reported (they tell you where price may hesitate);
        they just don't get a veto.
        """
        if self.setup_tf not in LADDER_TFS:
            return self.rungs
        floor = LADDER_TFS.index(self.setup_tf)
        return [r for r in self.rungs if LADDER_TFS.index(r.timeframe) >= floor]

    @property
    def runway_r(self) -> float:
        """R to the FURTHEST gating rung -- how far this can run."""
        return max((r.r_multiple for r in self.gate_rungs), default=0.0)

    @property
    def nearest_r(self) -> float:
        """
        R to the NEAREST gating rung. THIS is what Gate 2 tests, not runway.

        A weekly high 6R away behind a daily high 0.5R away is not a 6R trade.
        Price arrives at the daily level, the daily magnetism is spent, and you
        bleed the move back waiting for a weekly target that was never going to
        be reached in one push. The nearest gating rung is the realistic first
        target, so it is the one that has to be worth taking.
        """
        return min((r.r_multiple for r in self.gate_rungs), default=0.0)

    def fail(self, reason: str) -> "Decision":
        self.passed = False
        self.reasons.append(reason)
        return self

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "runway_r": round(self.runway_r, 2),
            "nearest_r": round(self.nearest_r, 2),
            "rungs": [
                {"tf": r.timeframe, "level": round(r.level, 2),
                 "r": round(r.r_multiple, 2),
                 "gating": r in self.gate_rungs}
                for r in self.rungs
            ],
            "continuity": self.continuity,
            "nested_tfs": self.nested_tfs,
            "compressed": self.compressed,
            "reasons": self.reasons,
        }


# --------------------------------------------------------------------------
# primitives -- all operate on (closed, forming), never on raw frames
# --------------------------------------------------------------------------

def prior_extreme(closed: pd.DataFrame, direction: str) -> Optional[float]:
    """The last CLOSED bar's extreme in the trade direction."""
    if closed is None or closed.empty:
        return None
    last = closed.iloc[-1]
    return float(last["high"] if direction == BULL else last["low"])


def is_untouched(
    closed: pd.DataFrame,
    forming: Optional[pd.Series],
    direction: str,
) -> bool:
    """
    Has the forming bar NOT yet traded through the prior closed bar's extreme?

    forming is None (premarket, or after the close) -> nothing has been touched
    on this timeframe today. Trivially untouched. This is correct, not a
    loophole: at 08:00 the daily magnitude to PDH really is fully open.
    """
    level = prior_extreme(closed, direction)
    if level is None:
        return False
    if forming is None:
        return True
    if direction == BULL:
        return float(forming["high"]) < level
    return float(forming["low"]) > level


def bar_agrees(forming: Optional[pd.Series], closed: pd.DataFrame, direction: str) -> bool:
    """
    Continuity: is the FORMING bar trading on the correct side of its own open?

    Falls back to the last closed bar when there is no forming bar, which is
    what strat.continuity already does. Same semantics, one place.
    """
    bar = forming
    if bar is None:
        if closed is None or closed.empty:
            return False
        bar = closed.iloc[-1]
    green = float(bar["close"]) > float(bar["open"])
    return green if direction == BULL else (not green)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def open_magnitude(
    closed_by_tf: dict[str, pd.DataFrame],
    forming_by_tf: dict[str, Optional[pd.Series]],
    direction: str,
    trigger: float,
    invalidation: float,
    ladder_tfs: tuple[str, ...] = LADDER_TFS,
) -> list[Rung]:
    """
    Untouched higher-TF extremes beyond the trigger, expressed in R.

    A rung is included only if BOTH:
      1. the forming bar has not already traded through it (still untouched)
      2. it lies BEYOND the trigger in the trade direction -- a level price has
         already cleared is not a target, it is history.
    """
    risk = abs(trigger - invalidation)
    if risk <= 0:
        return []

    rungs: list[Rung] = []
    for tf in ladder_tfs:
        closed = closed_by_tf.get(tf)
        if closed is None or closed.empty:
            continue
        if not is_untouched(closed, forming_by_tf.get(tf), direction):
            continue

        level = prior_extreme(closed, direction)
        beyond = level > trigger if direction == BULL else level < trigger
        if not beyond:
            continue

        rungs.append(Rung(tf, level, abs(level - trigger) / risk))

    return sorted(rungs, key=lambda r: r.r_multiple)


def continuity_gate(
    closed_by_tf: dict[str, pd.DataFrame],
    forming_by_tf: dict[str, Optional[pd.Series]],
    direction: str,
    tfs: tuple[str, ...] = CONTINUITY_TFS,
) -> dict[str, bool]:
    """Per-timeframe agreement. Unlike strat.continuity, this one is a GATE."""
    out: dict[str, bool] = {}
    for tf in tfs:
        closed = closed_by_tf.get(tf)
        if closed is None or closed.empty:
            continue
        out[tf] = bar_agrees(forming_by_tf.get(tf), closed, direction)
    return out


def nested_at(
    closed_by_tf: dict[str, pd.DataFrame],
    trigger: float,
    tfs: tuple[str, ...] = ("4H", "1D", "1W"),
    tolerance_pct: float = NEST_TOLERANCE_PCT,
) -> list[str]:
    """
    Timeframes whose prior-bar extreme sits ON the trigger (within tolerance).

    The compression-release check. When the trigger price IS also the 4H prior
    high, firing it doesn't just break the setup bar -- it converts the 4H bar
    into a 2U as well. One order-flow event, several timeframes expanding at
    once. That simultaneity is where the outsized moves come from.
    """
    hit: list[str] = []
    for tf in tfs:
        closed = closed_by_tf.get(tf)
        if closed is None or closed.empty or trigger == 0:
            continue
        last = closed.iloc[-1]
        for level in (float(last["high"]), float(last["low"])):
            if abs(level - trigger) / trigger <= tolerance_pct:
                hit.append(tf)
                break
    return hit


def higher_tf_compressed(
    closed_by_tf: dict[str, pd.DataFrame],
    tfs: tuple[str, ...] = ("4H", "1D"),
) -> bool:
    """Is a 4H or Daily bar currently an inside bar (a '1')? Coiled spring."""
    for tf in tfs:
        closed = closed_by_tf.get(tf)
        if closed is None or len(closed) < 1:
            continue
        if closed["label"].iloc[-1] == "1":
            return True
    return False


# --------------------------------------------------------------------------
# the gate stack
# --------------------------------------------------------------------------

def evaluate(
    closed_by_tf: dict[str, pd.DataFrame],
    forming_by_tf: dict[str, Optional[pd.Series]],
    *,
    setup_tf: str,
    direction: str,
    trigger: float,
    invalidation: Optional[float],
    min_runway_r: float = MIN_RUNWAY_R,
) -> Decision:
    """
    Run every gate. Returns a Decision carrying the full reason trail so
    diagnose.py can show you exactly why a level did or did not alert.
    """
    d = Decision(setup_tf=setup_tf)

    # Gate 1 -- higher-timeframe nomination.
    # This is the entire noise fix. 2H/15m/5m do not produce levels; they
    # confirm levels that a 4H or Daily bar nominated.
    if setup_tf not in NOMINATING_TFS:
        return d.fail(f"gate1: {setup_tf} may not nominate a level")

    if invalidation is None:
        return d.fail("gate1: no invalidation, cannot size risk")

    # Ladder + context
    d.rungs = open_magnitude(
        closed_by_tf, forming_by_tf, direction, trigger, invalidation
    )
    d.continuity = continuity_gate(closed_by_tf, forming_by_tf, direction)
    d.nested_tfs = nested_at(closed_by_tf, trigger)
    d.compressed = higher_tf_compressed(closed_by_tf)

    # Gate 2 -- open magnitude, measured only on rungs at or above setup_tf
    if not d.gate_rungs:
        return d.fail("gate2: ladder empty -- all higher-TF magnitude spent")
    if d.nearest_r < min_runway_r:
        return d.fail(
            f"gate2: nearest gating rung {d.nearest_r:.1f}R < {min_runway_r:.1f}R "
            f"({d.gate_rungs[0]})"
        )

    # Gate 3 -- continuity on 4H and Daily
    against = [tf for tf, ok in d.continuity.items() if not ok]
    if against:
        return d.fail(f"gate3: {', '.join(against)} forming bar against direction")

    # Score -- the sort key for the daily alert budget (Gate 5).
    # These weights are a first guess, NOT a validated model. Treat the number
    # as an ordering, not as truth, until it has been backtested.
    score = sum(d.continuity.values())        # 0-2
    score += 2 * len(d.nested_tfs)            # +2 per nested timeframe
    score += len(d.gate_rungs)                # untouched gating rungs remaining
    if d.compressed:
        score += 2

    if d.runway_r > 4.0:
        score *= 2
    elif d.runway_r >= 2.0:
        score = int(score * 1.5)

    d.score = int(score)
    d.passed = True
    d.reasons.append(
        f"pass: {len(d.gate_rungs)} gating rungs, nearest {d.nearest_r:.1f}R, "
        f"runway {d.runway_r:.1f}R, score {d.score}"
    )
    return d


def rank_and_budget(levels: list, budget: int = ALERT_BUDGET) -> list:
    """
    Gate 5 -- the hard daily cap.

    Crude on purpose. A cap forces the score to do real work, and it keeps the
    feed small enough that you READ it rather than tune it out. Tuning out your
    own alerts is how you end up chasing.

    Expects each level to carry a `.decision` (a Decision).
    """
    passing = [lv for lv in levels if getattr(lv, "decision", None) and lv.decision.passed]
    passing.sort(
        key=lambda lv: (lv.decision.score, lv.decision.runway_r),
        reverse=True,
    )
    return passing[:budget]
