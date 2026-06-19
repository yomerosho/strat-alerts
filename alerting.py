"""
alerting.py
-----------
AlertManager: sends Strat setup notifications to Telegram and/or WhatsApp.
StateStore: SQLite-backed "last known state" so alerts only fire once per
            new setup/trigger instead of spamming every scan cycle.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger("strat_scanner.alerting")


class StateStore:
    """Persists the last alerted setup-key per (symbol, timeframe) in SQLite,
    so the debounce survives restarts -- not just in-memory state.

    Two separate things are tracked:
    - setup_key / updated_at: the latest seen state, updated every cycle.
      Used to detect "did anything change since last time" (the debounce).
    - last_alert_at: only updated when an alert is actually SENT. Used for
      the cooldown, which is a different concept from debounce -- it
      suppresses rapid flapping (the same symbol crossing its trigger
      level back and forth) even though each flap technically counts as
      "a change," without blocking a genuinely new setup once enough time
      has passed.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS last_state (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    setup_key TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_alert_at TEXT,
                    PRIMARY KEY (symbol, timeframe)
                )
                """
            )
            # Migration for databases created before last_alert_at existed.
            try:
                conn.execute("ALTER TABLE last_state ADD COLUMN last_alert_at TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_last_setup_key(self, symbol: str, timeframe: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT setup_key FROM last_state WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
            return row[0] if row else None

    def set_last_setup_key(self, symbol: str, timeframe: str, setup_key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO last_state (symbol, timeframe, setup_key, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(symbol, timeframe)
                DO UPDATE SET setup_key = excluded.setup_key, updated_at = excluded.updated_at
                """,
                (symbol, timeframe, setup_key),
            )

    def minutes_since_last_alert(self, symbol: str, timeframe: str) -> Optional[float]:
        """Minutes since an alert was actually sent for this symbol, or
        None if one has never been sent."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_alert_at FROM last_state WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()
            if not row or not row[0]:
                return None
            last_alert = datetime.fromisoformat(row[0])
            return (datetime.utcnow() - last_alert).total_seconds() / 60

    def record_alert_sent(self, symbol: str, timeframe: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO last_state (symbol, timeframe, setup_key, updated_at, last_alert_at)
                VALUES (?, ?, '', datetime('now'), datetime('now'))
                ON CONFLICT(symbol, timeframe)
                DO UPDATE SET last_alert_at = excluded.last_alert_at
                """,
                (symbol, timeframe),
            )


class AlertManager:
    """Fan-out alert sender. Add more channels by adding a `_send_x` method
    and wiring it into `send()`."""

    def __init__(
        self,
        telegram_bot_token: str = "",
        telegram_chat_id: Optional[list] = None,
        twilio_account_sid: str = "",
        twilio_auth_token: str = "",
        twilio_whatsapp_from: str = "",
        twilio_whatsapp_to: Optional[list] = None,
    ):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_ids = telegram_chat_id or []
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_whatsapp_from = twilio_whatsapp_from
        self.twilio_whatsapp_to_numbers = twilio_whatsapp_to or []

    async def send(self, message: str) -> None:
        async with aiohttp.ClientSession() as session:
            tasks = []
            if self.telegram_bot_token:
                for chat_id in self.telegram_chat_ids:
                    tasks.append(self._send_telegram(session, chat_id, message))
            if self.twilio_account_sid and self.twilio_auth_token and self.twilio_whatsapp_from:
                for to_number in self.twilio_whatsapp_to_numbers:
                    tasks.append(self._send_whatsapp(session, to_number, message))
            if tasks:
                await asyncio.gather(*tasks)

    async def _send_telegram(self, session: aiohttp.ClientSession, chat_id: str, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status >= 300:
                    logger.error("Telegram send to %s failed (%s): %s", chat_id, resp.status, await resp.text())
        except Exception:
            logger.exception("Error sending Telegram alert to %s", chat_id)

    async def _send_whatsapp(self, session: aiohttp.ClientSession, to_number: str, message: str) -> None:
        """Sends via Twilio's WhatsApp API. Requires a Twilio account with a
        WhatsApp-enabled sender (sandbox for testing, or an approved
        WhatsApp Business sender for production). Each recipient must have
        individually joined the sandbox (or be approved on a production
        sender) -- there's no WhatsApp group equivalent here."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json"
        auth = aiohttp.BasicAuth(self.twilio_account_sid, self.twilio_auth_token)
        data = {
            "From": self.twilio_whatsapp_from,
            "To": to_number,
            "Body": message,
        }
        try:
            async with session.post(url, data=data, auth=auth, timeout=10) as resp:
                if resp.status >= 300:
                    logger.error("WhatsApp send to %s failed (%s): %s", to_number, resp.status, await resp.text())
        except Exception:
            logger.exception("Error sending WhatsApp alert to %s", to_number)


def format_confluence_alert(
    symbol: str,
    ftfc: str,
    entry_tf: str,
    trigger: str,
    states: dict,  # dict[str, scanner.StratState] -- avoiding circular import in type hints
    ftfc_timeframes: list,
) -> str:
    """A confluence alert: FTFC direction agrees across the higher
    timeframes AND a live trigger just fired on the entry timeframe in
    that same direction -- the actual "go" signal for a 0DTE-style entry."""
    direction_emoji = "🟢" if ftfc == "bull" else "🔴"
    direction_word = "BULLISH" if ftfc == "bull" else "BEARISH"

    ftfc_summary = " · ".join(
        f"{tf}:{states[tf].direction}" for tf in ftfc_timeframes if tf in states
    )
    entry_state = states[entry_tf]
    arrow = "↑" if trigger == "bullish_trigger" else "↓"
    level = entry_state.last_completed_high if trigger == "bullish_trigger" else entry_state.last_completed_low

    return (
        f"{direction_emoji} **{symbol}** — {direction_word} FTFC + entry trigger on {entry_tf}\n"
        f"Price {entry_state.current_price:.2f} broke {arrow} {level:.2f}\n"
        f"FTFC: {ftfc_summary}\n"
        f"Check your 5m/15m chart now to confirm entry."
    )
