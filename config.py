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

    # --- Setup timeframes: where patterns NOMINATE a level ---
    # v5: 4H and DAILY. 2H is demoted.
    #
    # v4 said "2H/4H only" and called that the noise fix, but it wasn't -- it
    # just moved the floor up one notch. 2H still produced roughly half the
    # armed levels, and the DAILY frame (fetched, labelled, and then never
    # passed to arm_inside_bar) produced none at all. The best nominator in the
    # system was disabled and the worst one was enabled.
    #
    # A level is not something the scanner FINDS. It is something a higher
    # timeframe NOMINATES and everything below confirms.
    setup_timeframes: tuple[str, ...] = ("4H", "1D")

    # --- Confirmation timeframe: where triggers CONFIRM ---
    # The 5Min tier is GONE. A 5m close through a 4H level resolves back inside
    # often enough that acting on it is a coin flip with commissions.
    #
    # tier1 = price is through the level intrabar, 15m not yet closed (heads-up)
    # tier2 = a 15m bar has CLOSED through the level (the entry)
    tier2_timeframe: str = "15Min"

    # --- Continuity (FTFC) ---
    # Reported on every alert. Gate is OFF by default: v3's 4/5 threshold was
    # silently eating setups, and with 2H/4H setups you already have far
    # fewer, higher-quality candidates. Turn the gate on only if the board
    # actually gets too busy.
    # These five are the DISPLAY string on the alert ("3/5"). Cosmetic.
    continuity_timeframes: tuple[str, ...] = ("15Min", "1H", "2H", "4H", "1D")
    min_continuity_tier1: int = field(default_factory=lambda: int(os.getenv("MIN_CONTINUITY_TIER1") or "0"))
    min_continuity_tier2: int = field(default_factory=lambda: int(os.getenv("MIN_CONTINUITY_TIER2") or "0"))

    # --- v5 gates (magnitude.py) ---
    # The REAL continuity gate lives in magnitude.CONTINUITY_TFS = ("4H", "1D")
    # and is not optional. The thresholds above defaulted to 0, which meant the
    # continuity gate has never once been on -- it was printed on every alert
    # and enforced on none of them.

    # Gate 2: the nearest untouched higher-TF extreme must be at least this far
    # away, in R. Below this there is no swing there, however clean the pattern.
    # If the reject log fills with "nearest gating rung 1.4R", this is well
    # calibrated. If nothing ever fires, try 1.5.
    min_runway_r: float = field(
        default_factory=lambda: float(os.getenv("MIN_RUNWAY_R") or "2.0")
    )

    # Gate 3b: Full Timeframe Continuity. Require at least `min_ftfc` of the
    # watched frames to be trading in the trade's direction at entry. Validated
    # on a 208-trade / 14-symbol replay (R measured from the ACTUAL fill, not
    # the trigger): no gate 73.6% win / +0.59R, >=4/5 75.8% / +0.66R keeping
    # 180 trades, >=5/5 80.3% / +0.88R but only 79 trades. 4/5 is the balanced
    # point -- nearly all the frequency, a real quality bump. This sits ON TOP
    # of the 4H+1D hard-continuity gate (magnitude.CONTINUITY_TFS).
    ftfc_timeframes: tuple[str, ...] = ("1H", "2H", "4H", "1D", "1W")
    min_ftfc: int = field(default_factory=lambda: int(os.getenv("MIN_FTFC") or "4"))

    # Gate 5: hard cap on alerts per scan, across the whole watchlist, ranked by
    # score. Crude on purpose -- it forces the ranking to do real work and keeps
    # the feed small enough that you READ it instead of tuning it out.
    alert_budget: int = field(
        default_factory=lambda: int(os.getenv("ALERT_BUDGET") or "3")
    )

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
