"""
config.py
---------
Central configuration for the Strat Scanner service.

All secrets/URLs are read from environment variables (use a `.env` file with
python-dotenv, or export them in your systemd unit / shell profile).

Ticker management:
- DEFAULT_TICKERS seeds the watchlist on first run.
- The live watchlist actually scanned is stored in `tickers.txt` (one symbol
  per line). On startup, if tickers.txt doesn't exist, it is created from
  DEFAULT_TICKERS. The file is re-read at the top of every scan cycle, so you
  can add/remove symbols while the service is running -- no restart needed.
- `main.py` also exposes a `--add-ticker` / `--remove-ticker` CLI flag that
  edits tickers.txt for you (see main.py --help).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can be set any other way.
    pass

BASE_DIR = Path(__file__).resolve().parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
STATE_DB_PATH = BASE_DIR / "scanner_state.db"

# Tickers pulled from your watchlist screenshots (indices + the stock list).
DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "AMD", "AMZN", "GOOG", "GOOGL", "META",
    "MSFT", "NVDA", "PLTR", "TSLA",
]


def ensure_tickers_file() -> None:
    """Create tickers.txt from DEFAULT_TICKERS if it doesn't exist yet."""
    if not TICKERS_FILE.exists():
        TICKERS_FILE.write_text("\n".join(DEFAULT_TICKERS) + "\n")


def load_tickers() -> list[str]:
    """Read the current watchlist from disk (case-insensitive, de-duped)."""
    ensure_tickers_file()
    raw = TICKERS_FILE.read_text().splitlines()
    seen: list[str] = []
    for line in raw:
        sym = line.strip().upper()
        if sym and not sym.startswith("#") and sym not in seen:
            seen.append(sym)
    return seen


def add_ticker(symbol: str) -> list[str]:
    tickers = load_tickers()
    symbol = symbol.strip().upper()
    if symbol and symbol not in tickers:
        tickers.append(symbol)
        TICKERS_FILE.write_text("\n".join(tickers) + "\n")
    return tickers


def remove_ticker(symbol: str) -> list[str]:
    tickers = load_tickers()
    symbol = symbol.strip().upper()
    tickers = [t for t in tickers if t != symbol]
    TICKERS_FILE.write_text("\n".join(tickers) + "\n")
    return tickers


@dataclass
class Config:
    # --- Alpaca credentials (data-only key works fine; no trading needed) ---
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))

    # --- Alerting ---
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # WhatsApp via Twilio (https://www.twilio.com/whatsapp) -- needs a Twilio
    # account with the WhatsApp sandbox/sender enabled.
    twilio_account_sid: str = field(default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", ""))
    twilio_auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    twilio_whatsapp_from: str = field(default_factory=lambda: os.getenv("TWILIO_WHATSAPP_FROM", ""))  # e.g. "whatsapp:+14155238886"
    twilio_whatsapp_to: str = field(default_factory=lambda: os.getenv("TWILIO_WHATSAPP_TO", ""))      # e.g. "whatsapp:+15551234567"

    # --- Scan behavior ---
    scan_interval_seconds: int = field(default_factory=lambda: int(os.getenv("SCAN_INTERVAL_SECONDS", "60")))
    timeframes: tuple[str, ...] = ("4H", "1D", "1W", "1M")

    # --- Misc ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with the current config."""
        problems = []
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            problems.append("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set.")

        has_telegram = bool(self.telegram_bot_token and self.telegram_chat_id)
        has_whatsapp = bool(
            self.twilio_account_sid and self.twilio_auth_token
            and self.twilio_whatsapp_from and self.twilio_whatsapp_to
        )
        if not has_telegram and not has_whatsapp:
            problems.append(
                "No alert channel configured: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, "
                "and/or TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN + TWILIO_WHATSAPP_FROM + TWILIO_WHATSAPP_TO."
            )
        return problems


CONFIG = Config()
