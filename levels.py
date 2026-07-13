"""
levels.py
---------
The scanner core: ARMED LEVELS, not completed patterns.

The v3 mistake
==============
v3 alerted on patterns that had already COMPLETED. A confirmed 4H 2-1-2 means
the third candle has closed -- the move is over. You were being told about
trades you'd already missed.

The v4 model
============
Detect the setup while it's still ARMED, publish the trigger level, and stage
alerts as lower timeframes confirm.

Every Strat setup collapses to the same object:

    "If a bar closes on THIS SIDE of THIS LEVEL, the trade is on."

Six named patterns, one machine. The pattern name is a label on the alert,
not a branch in the logic.

    ARMED  --5m close on trigger side-->  TIER 1  --15m close-->  TIER 2

Two arming families
===================
FAMILY A -- inside-bar setups (2-1-2, 3-1-2, 1-1-2)
    Arms off the LAST CLOSED setup bar.
    An inside bar has no direction, so BOTH sides arm:
        high -> bullish breakout,  low -> bearish breakdown
    Trigger = close BEYOND the level (breakout).

FAMILY B -- Failed 2 (F2U / F2D)
    Arms off the CURRENTLY FORMING setup bar, referencing the prior closed one.
    Trigger = close BACK INSIDE the level (reversal).

That direction difference is why `trigger_side` is an explicit field rather
than something derived from `direction`. Family A closes away from the level;
Family B closes back through it.

State is a PURE FUNCTION of bar data
====================================
Armed levels and tier status are both re-derived from scratch every scan.
Nothing is carried forward in a database. This matters because the scanner
runs on a fresh GitHub Actions VM each cycle -- persisted state was always
the fragile part. The only thing the store tracks now is "did I already send
this alert", which is a much smaller thing to get wrong.

Expiry falls out for free: if the last closed setup bar is no longer an inside
bar, no level arms. Nothing to expire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from bars import is_closed
from strat import BEAR, BULL, DOWN, INSIDE, OUTSIDE, UP

# Tier names
ARMED = "ARMED"
TIER1 = "TIER1"
TIER2 = "TIER2"

FAMILY_INSIDE = "inside"
FAMILY_F2 = "f2"


@dataclass
class ArmedLevel:
    symbol: str
    setup_tf: str
    family: str            # FAMILY_INSIDE | FAMILY_F2
    pattern: str           # "2-1-2" | "3-1-2" | "1-1-2" | "F2U" | "F2D"
    direction: str         # BULL | BEAR  -- which way the trade goes
    trigger_side: str      # "above" | "below" -- where a close must land
    level: float           # the trigger price
    invalidation: Optional[float]  # the other side of the setup
    target: Optional[float]

    setup_bar_time: pd.Timestamp   # identity of the bar that armed this
    arm_time: pd.Timestamp         # 5m/15m closes only count from here

    current_price: float
    setup_bar_closes_at: Optional[pd.Timestamp] = None  # when the forming setup bar ends

    tier: str = ARMED
    tier1_time: Optional[pd.Timestamp] = None
    tier2_time: Optional[pd.Timestamp] = None
    minutes_to_next_15m: Optional[int] = None
    continuity: str = "0/0"
    invalidated_by_opposite: bool = False

    # The v5 gate verdict: ladder, continuity, nesting, score. Set by
    # scanner.py via magnitude.evaluate(). A level whose decision did not pass
    # is kept in the snapshot (so diagnose.py can show you WHY it was dropped)
    # but is never sent.
    decision: Optional[object] = None

    @property
    def key(self) -> str:
        """Stable identity across scans. The setup bar's timestamp is what
        makes this unique -- a new inside bar tomorrow is a different level
        even if the price happens to be identical."""
        return (
            f"{self.symbol}|{self.setup_tf}|{self.setup_bar_time.isoformat()}"
            f"|{self.pattern}|{self.direction}"
        )

    @property
    def distance_pct(self) -> float:
        """Signed % of current price through the trigger level. Positive
        means price is on the trigger side. A close 0.02% through is a very
        different thing from a close 0.4% through -- this is what tells you
        which."""
        if self.level == 0:
            return 0.0
        raw = (self.current_price - self.level) / self.level * 100
        return raw if self.trigger_side == "above" else -raw

    @property
    def is_through(self) -> bool:
        return self.distance_pct > 0

    @property
    def risk_reward(self) -> Optional[float]:
        """
        Reward-to-risk, measured from the TRIGGER (not from current price).

        Risk  = trigger -> invalidation (the other edge of the setup)
        Reward = trigger -> target

        This is worth computing because the Strat's own magnitude convention
        (target = the opposing extreme of the bar before the inside bar) can
        produce a target that sits almost on top of the trigger, whenever the
        inside bar is nearly as wide as the bar containing it. Seen live: a
        QQQ setup with 0.92 of risk and 0.21 of reward. Geometrically valid,
        completely unplayable.

        The pattern being "real" doesn't make the trade worth taking. This is
        the number that tells them apart.
        """
        if self.target is None or self.invalidation is None:
            return None
        risk = abs(self.level - self.invalidation)
        reward = abs(self.target - self.level)
        if risk <= 0:
            return None
        return reward / risk

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "setup_tf": self.setup_tf,
            "family": self.family,
            "pattern": self.pattern,
            "direction": self.direction,
            "trigger_side": self.trigger_side,
            "level": round(self.level, 2),
            "invalidation": round(self.invalidation, 2) if self.invalidation else None,
            "target": round(self.target, 2) if self.target else None,
            "risk_reward": round(self.risk_reward, 2) if self.risk_reward is not None else None,
            "current_price": round(self.current_price, 2),
            "distance_pct": round(self.distance_pct, 3),
            "tier": self.tier,
            "tier1_time": self.tier1_time.isoformat() if self.tier1_time is not None else None,
            "tier2_time": self.tier2_time.isoformat() if self.tier2_time is not None else None,
            "setup_bar_time": self.setup_bar_time.isoformat(),
            "setup_bar_closes_at": (
                self.setup_bar_closes_at.isoformat() if self.setup_bar_closes_at is not None else None
            ),
            "minutes_to_next_15m": self.minutes_to_next_15m,
            "continuity": self.continuity,
            "invalidated_by_opposite": self.invalidated_by_opposite,
            "decision": self.decision.to_dict() if self.decision is not None else None,
        }


# --------------------------------------------------------------------------
# Splitting closed vs forming
# --------------------------------------------------------------------------

def split_closed_forming(df: pd.DataFrame, now_et: pd.Timestamp) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Returns (closed_bars, forming_bar_or_None).

    A bar is closed once real time has passed its `bar_end`. Note the last
    row Alpaca gives you is NOT automatically "still forming" -- if you scan
    at 8pm, the day's final bar is long done. v3 got this right and it's
    preserved here.
    """
    if df.empty:
        return df, None

    closed_mask = df["bar_end"].apply(lambda e: is_closed(e, now_et))
    closed = df[closed_mask]
    forming_rows = df[~closed_mask]
    forming = forming_rows.iloc[-1] if len(forming_rows) else None
    return closed, forming


# --------------------------------------------------------------------------
# Arming -- Family A: inside bars
# --------------------------------------------------------------------------

def _inside_bar_pattern_name(labels: list[str | None]) -> str:
    """The label is cosmetic -- the levels are identical either way. It just
    tells you what kind of setup you're looking at when the alert lands."""
    if len(labels) < 2:
        return "?-1-2"
    prior = labels[-2]
    if prior in (UP, DOWN):
        return "2-1-2"
    if prior == OUTSIDE:
        return "3-1-2"
    if prior == INSIDE:
        return "1-1-2"
    return "?-1-2"


def arm_inside_bar(
    symbol: str,
    setup_tf: str,
    closed: pd.DataFrame,
    forming: Optional[pd.Series],
    current_price: float,
) -> list[ArmedLevel]:
    """
    If the last CLOSED setup bar is an inside bar, arm both of its edges.

    We do not guess direction. An inside bar is coiled, not biased. Both
    triggers go live and whichever breaks first is the trade -- continuity
    and your own read decide which one you actually want.
    """
    if len(closed) < 3:
        return []

    labels = list(closed["label"])
    if labels[-1] != INSIDE:
        return []

    inside = closed.iloc[-1]
    prior = closed.iloc[-2]  # the bar the inside bar is inside of
    pattern = _inside_bar_pattern_name(labels)

    hi = float(inside["high"])
    lo = float(inside["low"])
    arm_time = inside["bar_end"]

    # Classic Strat magnitude: the setup targets the opposing extreme of the
    # bar that came before the inside bar.
    target_up = float(prior["high"])
    target_down = float(prior["low"])

    setup_bar_closes_at = forming["bar_end"] if forming is not None else None

    return [
        ArmedLevel(
            symbol=symbol, setup_tf=setup_tf, family=FAMILY_INSIDE, pattern=pattern,
            direction=BULL, trigger_side="above", level=hi,
            invalidation=lo, target=target_up if target_up > hi else None,
            setup_bar_time=inside.name, arm_time=arm_time,
            current_price=current_price, setup_bar_closes_at=setup_bar_closes_at,
        ),
        ArmedLevel(
            symbol=symbol, setup_tf=setup_tf, family=FAMILY_INSIDE, pattern=pattern,
            direction=BEAR, trigger_side="below", level=lo,
            invalidation=hi, target=target_down if target_down < lo else None,
            setup_bar_time=inside.name, arm_time=arm_time,
            current_price=current_price, setup_bar_closes_at=setup_bar_closes_at,
        ),
    ]


# --------------------------------------------------------------------------
# Arming -- Family B: Failed 2
# --------------------------------------------------------------------------

def arm_failed_two(
    symbol: str,
    setup_tf: str,
    closed: pd.DataFrame,
    forming: Optional[pd.Series],
    df5: pd.DataFrame,
    current_price: float,
) -> list[ArmedLevel]:
    """
    F2 is the only setup that arms MID-BAR. It's inherently live.

    Mechanics: the forming setup bar pokes through the prior bar's high
    (a 2U attempt), then price falls back below that high. Breakout buyers
    are now trapped above. A close back below the prior high is the trigger,
    and the unwind is those longs getting flushed.

    The breach time matters. We can't just say "any 5m close below the prior
    high counts" -- price was below the prior high BEFORE the poke too. We
    find the exact 5m bar inside the forming setup bar that first exceeded
    the level, and only closes AFTER that count as a failure.

    F2 can arm and disarm several times within one setup bar as price chops
    around the level. That's expected. Each re-arm is the same key (same
    setup bar), so the store won't double-alert you.
    """
    # F2 references exactly one bar -- the last closed one. Unlike the
    # inside-bar family it needs no label history, so one closed bar is
    # genuinely enough.
    if forming is None or len(closed) < 1:
        return []

    prior = closed.iloc[-1]
    prior_high = float(prior["high"])
    prior_low = float(prior["low"])

    forming_start = forming.name
    # 5-minute bars that live inside the currently-forming setup bar
    inner = df5[df5.index >= forming_start]
    if inner.empty:
        return []

    out: list[ArmedLevel] = []

    # --- F2D: poked above prior high, now back below -> trapped longs ---
    if float(forming["high"]) > prior_high and current_price <= prior_high:
        breaches = inner[inner["high"] > prior_high]
        if not breaches.empty:
            out.append(ArmedLevel(
                symbol=symbol, setup_tf=setup_tf, family=FAMILY_F2, pattern="F2D",
                direction=BEAR, trigger_side="below", level=prior_high,
                invalidation=float(forming["high"]),
                target=prior_low,
                setup_bar_time=forming_start,
                arm_time=breaches.iloc[0]["bar_end"],
                current_price=current_price,
                setup_bar_closes_at=forming["bar_end"],
            ))

    # --- F2U: poked below prior low, now back above -> trapped shorts ---
    if float(forming["low"]) < prior_low and current_price >= prior_low:
        breaches = inner[inner["low"] < prior_low]
        if not breaches.empty:
            out.append(ArmedLevel(
                symbol=symbol, setup_tf=setup_tf, family=FAMILY_F2, pattern="F2U",
                direction=BULL, trigger_side="above", level=prior_low,
                invalidation=float(forming["low"]),
                target=prior_high,
                setup_bar_time=forming_start,
                arm_time=breaches.iloc[0]["bar_end"],
                current_price=current_price,
                setup_bar_closes_at=forming["bar_end"],
            ))

    return out


# --------------------------------------------------------------------------
# Tier evaluation
# --------------------------------------------------------------------------

def _first_confirming_close(
    df: pd.DataFrame,
    level: ArmedLevel,
    now_et: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    """First CLOSED bar at or after arm_time that closed on the trigger side."""
    if df.empty:
        return None

    window = df[df.index >= level.arm_time]
    if window.empty:
        return None

    closed = window[window["bar_end"].apply(lambda e: is_closed(e, now_et))]
    if closed.empty:
        return None

    if level.trigger_side == "above":
        hits = closed[closed["close"] > level.level]
    else:
        hits = closed[closed["close"] < level.level]

    return hits.iloc[0]["bar_end"] if not hits.empty else None


def minutes_until_next_15m_close(now_et: pd.Timestamp) -> int:
    """
    The nesting gotcha, quantified.

    5-minute bars nest inside 15-minute bars, so the gap between a Tier 1
    and a Tier 2 alert is anywhere from 0 to 10 minutes depending on WHERE
    in the 15m bar the triggering 5m close landed.

    If the 5m that triggered was the third of its 15m group, Tier 2 arrives
    almost immediately and tells you nothing you didn't already know. If it
    was the first, you have a real 10-minute decision window.

    You need to know which. So every Tier 1 alert carries this number.
    """
    from bars import session_open

    minutes_in = int((now_et - session_open(now_et)).total_seconds() // 60)
    if minutes_in < 0:
        return 15
    return 15 - (minutes_in % 15)


def evaluate_tiers(
    level: ArmedLevel,
    df15: pd.DataFrame,
    now_et: pd.Timestamp,
) -> ArmedLevel:
    """
    Promote ARMED -> TIER1 -> TIER2. The 5-minute close is GONE.

    v4 promoted to Tier 1 on a 5m close through the level. That is the single
    biggest generator of nuisance alerts in the system: a 5m close through a
    4H level, with no 15m confirmation, resolves back inside the level often
    enough that acting on it is a coin flip with commissions.

    The v5 tiers:

        TIER1  price is CURRENTLY through the level, intrabar, 15m not yet
               closed. This is a HEADS-UP, not an entry. It carries
               minutes_to_next_15m so you know how long your decision window is.

        TIER2  a 15-minute bar has CLOSED through the level. This is the entry.
               Nothing else is.

    Tier 1 is now derived from live price rather than from a closed 5m bar,
    which means it is also strictly more current: it fires the moment price is
    through, not up to five minutes later.
    """
    t2 = _first_confirming_close(df15, level, now_et)
    if t2 is not None:
        level.tier = TIER2
        level.tier2_time = t2
        level.tier1_time = level.tier1_time or t2
    elif level.is_through:
        level.tier = TIER1
        level.tier1_time = now_et

    level.minutes_to_next_15m = minutes_until_next_15m_close(now_et)
    return level


def mark_opposite_invalidation(levels: list[ArmedLevel]) -> list[ArmedLevel]:
    """
    Within one inside bar, if one side has confirmed at Tier 2, the other
    side is dead. The bar resolved. Don't fire the loser.
    """
    by_bar: dict[tuple, list[ArmedLevel]] = {}
    for lv in levels:
        if lv.family != FAMILY_INSIDE:
            continue
        by_bar.setdefault((lv.symbol, lv.setup_tf, lv.setup_bar_time), []).append(lv)

    for group in by_bar.values():
        if len(group) != 2:
            continue
        a, b = group
        if a.tier == TIER2 and b.tier == ARMED:
            b.invalidated_by_opposite = True
        if b.tier == TIER2 and a.tier == ARMED:
            a.invalidated_by_opposite = True

    return levels
