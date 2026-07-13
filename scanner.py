"""
scanner.py
----------
Per-symbol orchestration: fetch bars -> label -> arm levels -> evaluate tiers.

Setup timeframes are 2H and 4H only. 1H, 30m, and the 5m/15m confirmation
frames are still computed -- they feed the continuity score and the tier
logic -- but they never arm a setup themselves. That's the noise reduction:
you're no longer being alerted to a 15-minute 2-1-2 that resolves in eleven
minutes and means nothing.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

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
        Returns (armed_levels, snapshot) for one symbol.

        `snapshot` is dashboard context -- current price, bar labels per
        timeframe -- and is written even when nothing is armed. An empty
        board is information too: it tells you the scanner ran and found
        nothing, rather than leaving you wondering if it's broken.
        """
        now_et = pd.Timestamp.now(tz=ET)

        frames = self.provider.fetch(symbol)
        if not frames or "5Min" not in frames:
            return [], {}

        labeled: dict[str, pd.DataFrame] = {}
        closed_by_tf: dict[str, pd.DataFrame] = {}
        forming_by_tf: dict[str, Optional[pd.Series]] = {}

        for tf, df in frames.items():
            lab = label_bars(df)
            labeled[tf] = lab
            closed, forming = split_closed_forming(lab, now_et)
            closed_by_tf[tf] = closed
            forming_by_tf[tf] = forming

        df5 = labeled["5Min"]
        current_price = float(df5["close"].iloc[-1])

        # --- Arm setups on 2H / 4H only ---
        armed: list[ArmedLevel] = []
        for tf in CONFIG.setup_timeframes:
            closed = closed_by_tf.get(tf)
            if closed is None or len(closed) < 3:
                continue
            forming = forming_by_tf.get(tf)

            armed.extend(arm_inside_bar(symbol, tf, closed, forming, current_price))

            if CONFIG.enable_failed_two:
                armed.extend(
                    arm_failed_two(symbol, tf, closed, forming, df5, current_price)
                )

        if not armed:
            return [], self._snapshot(symbol, current_price, closed_by_tf, now_et)

        # --- Tier evaluation + continuity ---
        df15 = labeled.get("15Min", pd.DataFrame())
        for lv in armed:
            evaluate_tiers(lv, df5, df15, now_et)
            # Continuity reads the FORMING bar of each timeframe (price vs
            # that bar's own open), so pass the full frames, not just the
            # closed ones.
            agree, total = continuity(
                labeled, list(CONFIG.continuity_timeframes), lv.direction
            )
            lv.continuity = f"{agree}/{total}"

        armed = mark_opposite_invalidation(armed)

        return armed, self._snapshot(symbol, current_price, closed_by_tf, now_et)

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
        }
