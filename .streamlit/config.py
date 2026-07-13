"""
config.py
---------
Central configuration.

The big v4 changes:
  - setup_timeframes is 2H/4H ONLY. Nothing arms on 15m or below anymore.
  - Confirmation happens on 5Min (Tier 1) and 15Min (Tier 2).
  - Continuity is context on the alert, not a hard gate by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
STATE_DB_PATH = BASE_DIR / "scanner_state.db"
SNAPSHOT_FILE = BASE_DIR / "latest_scan.json"

DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "AMD", "AMZN", "GOOGL", "META",
    "MSFT", "NVDA", "PLTR", "TSLA",
]


def ensure_tickers_file() -> None:
    if not TICKERS_FILE.exists():
        TICKERS_FILE.write_text("\n".join(DEFAULT_TICKERS) + "\n")


def load_tickers() -> list[str]:
    ensure_tickers_file()
    seen: list[str] = []
    for line in TICKERS_FILE.read_text().splitlines():
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
    tickers = [t for t in load_tickers() if t != symbol.strip().upper()]
    TICKERS_FILE.write_text("\n".join(tickers) + "\n")
    return tickers


def parse_recipients(env_value: str) -> list[str]:
    return [x.strip() for x in env_value.split(",") if x.strip()]


@dataclass
class Config:
    # --- Alpaca ---
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_data_feed: str = field(default_factory=lambda: (os.getenv("ALPACA_DATA_FEED") or "iex").lower())

    # --- Telegram ---
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: list[str] = field(
        default_factory=lambda: parse_recipients(os.getenv("TELEGRAM_CHAT_ID", ""))
    )

    # --- Setup timeframes: where patterns ARM ---
    # 2H and 4H only. This is the whole point of v4. A setup on 15m resolves
    # in fifteen minutes and isn't worth an alert; a setup on 4H gives you a
    # level that stays relevant for hours, which is what a +1/+2 DTE hold
    # actually needs.
    setup_timeframes: tuple[str, ...] = ("2H", "4H")

    # --- Confirmation timeframes: where triggers CONFIRM ---
    tier1_timeframe: str = "5Min"    # 👀 early -- the level held one bar
    tier2_timeframe: str = "15Min"   # 🎯 conviction -- it held a full 15m

    # --- Continuity (FTFC) ---
    # Reported on every alert. Gate is OFF by default: v3's 4/5 threshold was
    # silently eating setups, and with 2H/4H setups you already have far
    # fewer, higher-quality candidates. Turn the gate on only if the board
    # actually gets too busy.
    continuity_timeframes: tuple[str, ...] = ("15Min", "1H", "2H", "4H", "1D")
    min_continuity_tier1: int = field(default_factory=lambda: int(os.getenv("MIN_CONTINUITY_TIER1") or "0"))
    min_continuity_tier2: int = field(default_factory=lambda: int(os.getenv("MIN_CONTINUITY_TIER2") or "0"))

    # --- Failed-2 ---
    enable_failed_two: bool = field(
        default_factory=lambda: (os.getenv("ENABLE_FAILED_TWO") or "true").lower() == "true"
    )
    # Trapped-trader unwinds move fast -- that's the entire mechanism. By the
    # 15m close the snapback can be half over. So F2 sends its Tier 1 as the
    # ACTIONABLE alert and treats Tier 2 as "still going", not as a gate.
    #
    # This is a judgment call, not a proven edge. Watch it live for a few
    # weeks before you trust it. If F2 Tier 1s keep failing, set this false
    # and F2 will wait for the 15m like everything else.
    failed_two_tier1_actionable: bool = field(
        default_factory=lambda: (os.getenv("F2_TIER1_ACTIONABLE") or "true").lower() == "true"
    )

    # --- Alert hygiene ---
    # Only alert when price is within this % of an armed level, so a level
    # 3% away doesn't ping you at 9:31. Set 0 to disable.
    proximity_alert_pct: float = field(
        default_factory=lambda: float(os.getenv("PROXIMITY_ALERT_PCT") or "0.75")
    )
    # Suppress alerts whose reward:risk (trigger->target vs trigger->stop) is
    # below this. The Strat's magnitude convention can hand you a target that
    # sits right on top of the trigger when the inside bar is nearly as wide
    # as its parent -- geometrically a valid pattern, but not a trade.
    # Set 0 to disable and see everything.
    min_risk_reward: float = field(
        default_factory=lambda: float(os.getenv("MIN_RISK_REWARD") or "1.0")
    )
    scan_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("SCAN_INTERVAL_SECONDS") or "300")
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        problems = []
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            problems.append("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set.")
        if not (self.telegram_bot_token and self.telegram_chat_id):
            problems.append("No alert channel: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.")
        return problems


CONFIG = Config()
