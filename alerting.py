"""
alerting.py
-----------
AlertManager: sends Strat setup notifications to Telegram and/or WhatsApp.
StateStore: SQLite-backed "last known state" so alerts only fire once per
            new setup/trigger instead of spamming every scan cycle.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger("strat_scanner.alerting")


class StateStore:
    """Persists the last alerted setup-key per (symbol, timeframe) in SQLite,
    so the debounce survives restarts -- not just in-memory state."""

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
                    PRIMARY KEY (symbol, timeframe)
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


class AlertManager:
    """Fan-out alert sender. Add more channels by adding a `_send_x` method
    and wiring it into `send()`."""

    def __init__(
        self,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        twilio_account_sid: str = "",
        twilio_auth_token: str = "",
        twilio_whatsapp_from: str = "",
        twilio_whatsapp_to: str = "",
    ):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_whatsapp_from = twilio_whatsapp_from
        self.twilio_whatsapp_to = twilio_whatsapp_to

    async def send(self, message: str) -> None:
        async with aiohttp.ClientSession() as session:
            if self.telegram_bot_token and self.telegram_chat_id:
                await self._send_telegram(session, message)
            if self.twilio_account_sid and self.twilio_auth_token and self.twilio_whatsapp_from and self.twilio_whatsapp_to:
                await self._send_whatsapp(session, message)

    async def _send_telegram(self, session: aiohttp.ClientSession, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message, "parse_mode": "Markdown"}
        try:
            async with session.post(url, json=payload, timeout=10) as resp:
                if resp.status >= 300:
                    logger.error("Telegram send failed (%s): %s", resp.status, await resp.text())
        except Exception:
            logger.exception("Error sending Telegram alert")

    async def _send_whatsapp(self, session: aiohttp.ClientSession, message: str) -> None:
        """Sends via Twilio's WhatsApp API. Requires a Twilio account with a
        WhatsApp-enabled sender (sandbox for testing, or an approved
        WhatsApp Business sender for production)."""
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_account_sid}/Messages.json"
        auth = aiohttp.BasicAuth(self.twilio_account_sid, self.twilio_auth_token)
        data = {
            "From": self.twilio_whatsapp_from,
            "To": self.twilio_whatsapp_to,
            "Body": message,
        }
        try:
            async with session.post(url, data=data, auth=auth, timeout=10) as resp:
                if resp.status >= 300:
                    logger.error("WhatsApp (Twilio) send failed (%s): %s", resp.status, await resp.text())
        except Exception:
            logger.exception("Error sending WhatsApp alert")


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
