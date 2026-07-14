"""
scanner.py
----------
Per-symbol orchestration: fetch -> label -> arm -> GATE -> tier.

What changed in v5
==================
v4 armed levels on 2H and 4H, computed a same-timeframe target, and alerted.
Continuity was printed on the alert but gated nothing, and the Daily frame --
the best nominator in the system -- was fetched, labelled, and then never
passed to the arming functions at all.

v5 inverts the flow. A level is not something we FIND and then rank. It is
something a HIGHER TIMEFRAME NOMINATES, and everything below it confirms:

    4H / Daily   nominate the level          (Gate 1)
    the ladder   says how far it can run     (Gate 2)
    continuity   says whether to hold it     (Gate 3)
    the 15m      says when to enter          (tiers)
    the budget   says how many you get       (Gate 5, in main.py)

2H no longer arms. It is still fetched, still labelled, still a rung on the
magnitude ladder, still on the dashboard -- it simply cannot originate a trade.
That one change removes roughly half the armed levels outright, and the
magnitude gate removes most of what survives.

Expect this to go quiet. A scanner that sends nothing on a chop day is working
correctly. That is the hardest part to sit with, and it is the entire point.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import magnitude
from bars import ET, BarProvider
from config import CONFIG
from levels import (
    ArmedLevel,
    arm_failed_two,
    arm_inside_bar,
    evaluate_tiers,
    mark_opposite_invalidation,
    split_closed_forming,
)
from strat import continuity, label_bars

logger = logging.getLogger("strat_scanner.scanner")


class StratScanner:
    def __init__(self, api_key: str, secret_key: str, data_feed: str = "iex"):
        self.provider = BarProvider(api_key, secret_key, data_feed)

    def scan_symbol(self, symbol: str) -> tuple[list[ArmedLevel], dict]:
        """
        Returns (passing_levels, snapshot).

        `passing_levels` are ONLY those that cleared every gate. Levels that
        armed but were rejected are not discarded -- they go into the snapshot
        under "rejected" with their reason trail, so diagnose.py can tell you
        exactly which gate killed which setup.

        A silent scanner you cannot interrogate is indistinguishable from a
        broken one. That distinction is the whole reason the reject log exists.
        """
        now_et = pd.Timestamp.now(tz=ET)

        frames = self.provider.fetch(symbol)
        if not frames or "5Min" not in frames:
            return [], {}

        labeled: dict[str, pd.DataFrame] = {}
        closed_by_tf: dict[str, pd.DataFrame] = {}
        forming_by_tf: dict[str, Optional[pd.Series]] = {}

        for tf, df in frames.items():
            if tf.startswith("_"):
                continue
            lab = label_bars(df)
            labeled[tf] = lab
            closed, forming = split_closed_forming(lab, now_et)
            closed_by_tf[tf] = closed
            forming_by_tf[tf] = forming

        df5 = labeled["5Min"]

        # Price from the full tape (extended hours included); candles from RTH.
        last = frames.get("_last")
        if last is not None and not last.empty:
            current_price = float(last["close"].iloc[-1])
        else:
            current_price = float(df5["close"].iloc[-1])

        # --- Gate 1: arm on 4H and Daily only ---
        #
        # CONFIG.setup_timeframes must now be ("4H", "1D"). magnitude.evaluate()
        # would reject a 2H level anyway, but arming one only to throw it away
        # wastes the scan and clutters the reject log. Enforce it at the source.
        armed: list[ArmedLevel] = []
        for tf in CONFIG.setup_timeframes:
            if tf not in magnitude.NOMINATING_TFS:
                logger.warning(
                    'CONFIG.setup_timeframes contains %r, which cannot nominate '
                    'a level. Set setup_timeframes = ("4H", "1D").', tf,
                )
                continue

            closed = closed_by_tf.get(tf)
            if closed is None or len(closed) < 3:
                continue
            forming = forming_by_tf.get(tf)

            armed.extend(arm_inside_bar(symbol, tf, closed, forming, current_price))

            if CONFIG.enable_failed_two:
                armed.extend(
                    arm_failed_two(symbol, tf, closed, forming, df5, current_price)
                )

        snapshot = self._snapshot(symbol, current_price, closed_by_tf, now_et)

        if not armed:
            return [], snapshot

        # --- Gates 2 and 3: magnitude and continuity ---
        for lv in armed:
            lv.decision = magnitude.evaluate(
                closed_by_tf,
                forming_by_tf,
                setup_tf=lv.setup_tf,
                direction=lv.direction,
                trigger=lv.level,
                invalidation=lv.invalidation,
                min_runway_r=getattr(CONFIG, "min_runway_r", magnitude.MIN_RUNWAY_R),
                min_ftfc=getattr(CONFIG, "min_ftfc", magnitude.MIN_FTFC),
                ftfc_tfs=getattr(CONFIG, "ftfc_timeframes", magnitude.FTFC_TFS),
            )

            # The Strat's native target is the same-timeframe prior extreme.
            # Now that there is a real ladder, the FIRST untouched rung is the
            # honest first target -- it is the level price actually has to fight
            # through, and it may well be lower than the pattern's own target.
            if lv.decision.passed and lv.decision.rungs:
                lv.target = lv.decision.rungs[0].level

            agree, total = continuity(
                labeled, list(CONFIG.continuity_timeframes), lv.direction
            )
            lv.continuity = f"{agree}/{total}"

        passing = [lv for lv in armed if lv.decision.passed]
        rejected = [lv for lv in armed if not lv.decision.passed]

        snapshot["rejected"] = [
            {
                "pattern": lv.pattern,
                "setup_tf": lv.setup_tf,
                "direction": lv.direction,
                "level": round(lv.level, 2),
                "reasons": lv.decision.reasons,
            }
            for lv in rejected
        ]

        if not passing:
            logger.info(
                "%s: %d armed, 0 passed | %s",
                symbol,
                len(armed),
                "; ".join(r["reasons"][0] for r in snapshot["rejected"][:3]),
            )
            return [], snapshot

        # --- Tiers: 15m only. The 5m tier is gone. ---
        df15 = labeled.get("15Min", pd.DataFrame())
        for lv in passing:
            evaluate_tiers(lv, df15, now_et)

        passing = mark_opposite_invalidation(passing)

        logger.info(
            "%s: %d armed, %d passed (best score %d)",
            symbol,
            len(armed),
            len(passing),
            max(lv.decision.score for lv in passing),
        )
        return passing, snapshot

    @staticmethod
    def _snapshot(
        symbol: str,
        price: float,
        closed_by_tf: dict[str, pd.DataFrame],
        now_et: pd.Timestamp,
    ) -> dict:
        tf_state = {}
        for tf, df in closed_by_tf.items():
            if df.empty:
                continue
            labels = [x for x in df["label"].tail(3) if x]
            tf_state[tf] = {
                "labels": labels,
                "high": round(float(df["high"].iloc[-1]), 2),
                "low": round(float(df["low"].iloc[-1]), 2),
                "last_bar": df.index[-1].isoformat(),
            }
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "scanned_at": now_et.isoformat(),
            "timeframes": tf_state,
            "rejected": [],
        }
