"""
alerting.py
-----------
Telegram delivery + alert de-duplication.

Dedup, not debounce
===================
v3 tracked "the last setup key per symbol/timeframe" and compared it each
cycle. That's a debounce, and it was fragile: any flicker in the state string
re-fired the alert, which is why a cooldown had to be bolted on top.

v4 doesn't need any of that. An armed level has a stable identity -- symbol,
setup timeframe, the timestamp of the bar that armed it, pattern, direction.
That identity does not flicker. So the store answers exactly one question:

    "Have I already sent the TIER1 (or TIER2) alert for THIS level?"

Fire once, never again. No cooldown, no flapping, no state string to corrupt.
Price can chop back and forth across the trigger all afternoon and you will
not be spammed, because the level's identity never changed.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import aiohttp
import pandas as pd

from levels import FAMILY_F2, TIER1, TIER2, ArmedLevel

logger = logging.getLogger("strat_scanner.alerting")


class AlertStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_alerts (
                    level_key TEXT NOT NULL,
                    tier      TEXT NOT NULL,
                    sent_at   TEXT NOT NULL,
                    PRIMARY KEY (level_key, tier)
                )
                """
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def already_sent(self, level_key: str, tier: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sent_alerts WHERE level_key = ? AND tier = ?",
                (level_key, tier),
            ).fetchone()
            return row is not None

    def mark_sent(self, level_key: str, tier: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sent_alerts (level_key, tier, sent_at) "
                "VALUES (?, ?, datetime('now'))",
                (level_key, tier),
            )

    def prune(self, days: int = 7) -> None:
        """Keep the committed DB small -- it rides in the git repo."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM sent_alerts WHERE sent_at < datetime('now', ?)",
                (f"-{days} days",),
            )


class AlertManager:
    def __init__(self, telegram_bot_token: str = "", telegram_chat_id: list[str] | None = None):
        self.token = telegram_bot_token
        self.chat_ids = telegram_chat_id or []

    async def send(self, message: str) -> bool:
        """Returns True only if every recipient got it.

        The return value matters for --test-telegram: a 400 from Telegram's
        Markdown parser (unbalanced * or _) fails ONE message while the rest
        sail through, which is exactly the kind of bug that hides. The test
        needs to know, not just log it and move on.
        """
        if not self.token or not self.chat_ids:
            logger.info("No Telegram configured; would have sent:\n%s", message)
            return False
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *(self._send_telegram(session, cid, message) for cid in self.chat_ids)
            )
        return all(results)

    async def _send_telegram(self, session: aiohttp.ClientSession, chat_id: str, message: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            async with session.post(url, json=payload, timeout=15) as resp:
                if resp.status >= 300:
                    body = await resp.text()
                    logger.error("Telegram %s -> HTTP %s: %s", chat_id, resp.status, body)
                    return False
                return True
        except Exception:
            logger.exception("Telegram send failed for %s", chat_id)
            return False


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _fmt_time(ts: pd.Timestamp | None) -> str:
    return ts.strftime("%H:%M ET") if ts is not None else "--"


def _arrow(direction: str) -> str:
    return "🟢" if direction == "bull" else "🔴"


def _word(direction: str) -> str:
    return "BULLISH" if direction == "bull" else "BEARISH"


def _rr(lv: ArmedLevel) -> str:
    """R:R shown inline next to the target -- the number that separates a
    pattern that's real from a trade that's worth taking."""
    rr = lv.risk_reward
    return f"  (R:R {rr:.1f})" if rr is not None else ""


def _scale_plan(lv: ArmedLevel) -> list[str]:
    """
    The committed exit model, spelled out on the alert.

    Scale-out: half off at +1R, stop to breakeven, runner to the first gating
    rung. On the v5 replay this ran ~88% win at +1.32R/trade -- the point is
    that once +1R is paid, the runner can only scratch, never lose.

    Falls back to a plain stop/target block if the level is unsized (no
    invalidation), because then there is no 1R to scale at.
    """
    if lv.invalidation is None or lv.scale_level is None:
        out = []
        if lv.invalidation is not None:
            out.append(f"Stop: {lv.invalidation:.2f}")
        if lv.target is not None:
            out.append(f"Target: {lv.target:.2f}{_rr(lv)}")
        return out

    runner = f"  ({lv.runner_r:.1f}R)" if lv.runner_r is not None else ""
    lines = [
        "*Plan — scale-out:*",
        f"  Entry {lv.level:.2f}  ·  Stop {lv.invalidation:.2f}",
        f"  +1R at {lv.scale_level:.2f} → take half, stop to breakeven",
    ]
    if lv.target is not None:
        lines.append(f"  Runner → {lv.target:.2f}{runner}")
    return lines


def format_arm_alert(lv: ArmedLevel) -> str:
    """
    👀 Level is armed and price is approaching. No confirmation yet.

    This is the alert v3 never had, and it's the one that matters most: the
    trade hasn't happened. You have time to pull up the chart, check context,
    and decide whether you even want it before anything triggers.
    """
    side = "above" if lv.trigger_side == "above" else "below"
    lines = [
        f"👀 {_arrow(lv.direction)} *{lv.symbol}* — {lv.setup_tf} {lv.pattern} ARMED",
        f"{_word(lv.direction)} · needs a close {side} *{lv.level:.2f}*",
        "",
        f"Price:  {lv.current_price:.2f}  ({abs(lv.distance_pct):.2f}% away)",
    ]
    lines.extend(_scale_plan(lv))
    lines.append(f"Continuity: {lv.continuity}")
    if lv.setup_bar_closes_at is not None:
        lines.append(f"{lv.setup_tf} bar closes: {_fmt_time(lv.setup_bar_closes_at)}")
    lines.append("")
    lines.append("_Not triggered yet. Get the chart up._")
    return "\n".join(lines)


def format_tier1_alert(lv: ArmedLevel, f2_actionable: bool) -> str:
    """
    ⚡ 5-minute close on the trigger side.

    The level held for one bar. Early, and worth acting on -- but the 15m
    hasn't confirmed yet, and the nesting means that could be 10 minutes away
    or 30 seconds away. The alert tells you which, because that changes
    whether waiting costs you anything.
    """
    is_f2 = lv.family == FAMILY_F2
    hot = is_f2 and f2_actionable

    header = "🔥 ⚡" if hot else "⚡"
    lines = [
        f"{header} {_arrow(lv.direction)} *{lv.symbol}* — {lv.setup_tf} {lv.pattern} · TIER 1",
        f"5m closed {lv.trigger_side} *{lv.level:.2f}*  ({lv.distance_pct:+.2f}% through)",
        "",
        f"Price:  {lv.current_price:.2f}",
        f"Confirmed at: {_fmt_time(lv.tier1_time)}",
    ]

    if lv.minutes_to_next_15m is not None:
        if lv.minutes_to_next_15m <= 1:
            lines.append("15m closing *now* — Tier 2 lands immediately")
        else:
            lines.append(f"15m closes in *{lv.minutes_to_next_15m} min*")

    lines.extend(_scale_plan(lv))
    lines.append(f"Continuity: {lv.continuity}")
    if lv.setup_bar_closes_at is not None:
        lines.append(f"{lv.setup_tf} bar closes: {_fmt_time(lv.setup_bar_closes_at)}")

    lines.append("")
    if hot:
        lines.append("🔥 *F2 — trapped traders unwind fast. This is the actionable tier;*")
        lines.append("*waiting for the 15m may hand back most of the move.*")
    else:
        lines.append("_Starter size, or wait for Tier 2._")
    return "\n".join(lines)


def format_tier2_alert(lv: ArmedLevel) -> str:
    """
    🎯 15-minute close on the trigger side. The level held.

    For inside-bar setups this is the conviction entry. For an F2 it's a
    "still going" confirmation -- if you took the Tier 1, this says stay in.
    """
    is_f2 = lv.family == FAMILY_F2
    lines = [
        f"🎯 {_arrow(lv.direction)} *{lv.symbol}* — {lv.setup_tf} {lv.pattern} · TIER 2",
        f"15m closed {lv.trigger_side} *{lv.level:.2f}*  ({lv.distance_pct:+.2f}% through)",
        "",
        f"Price:  {lv.current_price:.2f}",
        f"Confirmed at: {_fmt_time(lv.tier2_time)}",
    ]
    if lv.tier1_time is not None:
        lines.append(f"Tier 1 was: {_fmt_time(lv.tier1_time)}")
    lines.extend(_scale_plan(lv))
    lines.append(f"Continuity: {lv.continuity}")
    if lv.setup_bar_closes_at is not None:
        lines.append(f"{lv.setup_tf} bar closes: {_fmt_time(lv.setup_bar_closes_at)}")

    lines.append("")
    lines.append("_Continuation confirmation — if you took Tier 1, it's holding._"
                 if is_f2 else "_Conviction tier._")
    return "\n".join(lines)


def format_for_tier(lv: ArmedLevel, tier: str, f2_actionable: bool) -> str:
    if tier == TIER2:
        return format_tier2_alert(lv)
    if tier == TIER1:
        return format_tier1_alert(lv, f2_actionable)
    return format_arm_alert(lv)
