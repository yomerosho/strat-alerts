"""
main.py
-------
Long-running asyncio service that:
  1. Re-reads the watchlist (tickers.txt) every cycle -> dynamic custom tickers.
  2. For each symbol, fetches every configured timeframe concurrently.
  3. Computes FTFC (Full Timeframe Continuity) across the FTFC timeframes
     (15Min/30Min/1H/4H/1D by default) and checks the entry timeframes
     (5Min/15Min by default) for a trigger matching that direction.
  4. Sends a confluence alert to Telegram/WhatsApp only when FTFC agrees
     AND a matching entry trigger fires -- not on every single-timeframe blip.
  5. Writes latest_scan.json -- the data feed the Streamlit dashboard reads.

Run as a service:
    python main.py

Manage the watchlist without restarting the service:
    python main.py --add-ticker COIN
    python main.py --remove-ticker PLTR
    python main.py --list-tickers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Optional

from config import BASE_DIR, CONFIG, STATE_DB_PATH, add_ticker, load_tickers, remove_ticker
from scanner import StratScanner, StratState
from alerting import AlertManager, StateStore, format_confluence_alert

logger = logging.getLogger("strat_scanner.main")
SNAPSHOT_FILE = BASE_DIR / "latest_scan.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def compute_ftfc(states: dict[str, StratState]) -> str:
    """bull / bear / mixed, based on agreement across CONFIG.ftfc_timeframes."""
    dirs = {states[tf].direction for tf in CONFIG.ftfc_timeframes if tf in states}
    if dirs == {"bull"}:
        return "bull"
    if dirs == {"bear"}:
        return "bear"
    return "mixed"


def find_entry_signal(ftfc: str, states: dict[str, StratState]) -> Optional[tuple[str, str]]:
    """Returns (timeframe, trigger) for the first entry timeframe whose live
    trigger matches the FTFC direction, or None if there isn't one."""
    if ftfc == "mixed":
        return None
    wanted = "bullish_trigger" if ftfc == "bull" else "bearish_trigger"
    for tf in CONFIG.entry_timeframes:
        state = states.get(tf)
        if state and state.trigger == wanted:
            return (tf, wanted)
    return None


async def evaluate_symbol(
    scanner: StratScanner,
    alert_manager: AlertManager,
    store: StateStore,
    symbol: str,
) -> list[dict]:
    """Fetches every configured timeframe for one symbol, computes FTFC +
    entry-signal confluence, alerts on a fresh confluence event, and
    returns JSON-serializable snapshots for the dashboard feed."""
    states: dict[str, StratState] = {}
    for timeframe in CONFIG.timeframes:
        try:
            # alpaca-py's client is sync; offload to a thread so it doesn't block the loop.
            state = await asyncio.to_thread(scanner.get_state, symbol, timeframe)
        except Exception:
            logger.exception("Failed to fetch/evaluate %s [%s]", symbol, timeframe)
            continue
        if state is not None:
            states[timeframe] = state

    if not states:
        return []

    ftfc = compute_ftfc(states)
    entry_signal = find_entry_signal(ftfc, states)

    setup_key = f"ftfc={ftfc}|entry={entry_signal}"
    last_key = store.get_last_setup_key(symbol, "CONFLUENCE")
    if setup_key != last_key:
        if entry_signal is not None:
            minutes_since = store.minutes_since_last_alert(symbol, "CONFLUENCE")
            in_cooldown = minutes_since is not None and minutes_since < CONFIG.alert_cooldown_minutes
            if in_cooldown:
                logger.info(
                    "Suppressing alert for %s (cooldown: %.1f/%d min) -- %s",
                    symbol, minutes_since, CONFIG.alert_cooldown_minutes, setup_key,
                )
            else:
                entry_tf, trigger = entry_signal
                message = format_confluence_alert(symbol, ftfc, entry_tf, trigger, states, CONFIG.ftfc_timeframes)
                logger.info("Confluence alert for %s: %s", symbol, setup_key)
                await alert_manager.send(message)
                store.record_alert_sent(symbol, "CONFLUENCE")
        store.set_last_setup_key(symbol, "CONFLUENCE", setup_key)

    return [state.to_dict() for state in states.values()]


async def scan_cycle(scanner: StratScanner, alert_manager: AlertManager, store: StateStore) -> None:
    tickers = load_tickers()
    logger.info("Starting scan cycle for %d tickers x %d timeframes", len(tickers), len(CONFIG.timeframes))

    tasks: list[asyncio.Task] = []
    try:
        async with asyncio.TaskGroup() as tg:  # Python 3.11+
            for symbol in tickers:
                tasks.append(tg.create_task(evaluate_symbol(scanner, alert_manager, store, symbol)))
    except* Exception as eg:  # pragma: no cover -- safety net, individual tasks already self-handle
        for exc in eg.exceptions:
            logger.exception("Unhandled error in scan task", exc_info=exc)

    snapshots: list[dict] = []
    for t in tasks:
        if not t.cancelled():
            snapshots.extend(t.result())
    write_snapshot_file(tickers, snapshots)


def write_snapshot_file(tickers: list[str], snapshots: list[dict]) -> None:
    """Writes the dashboard's data feed: every symbol/timeframe's current
    Strat state, regardless of whether it triggered an alert this cycle."""
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "timeframes": list(CONFIG.timeframes),
        "ftfc_timeframes": list(CONFIG.ftfc_timeframes),
        "entry_timeframes": list(CONFIG.entry_timeframes),
        "states": snapshots,
    }
    SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote snapshot for %d states to %s", len(snapshots), SNAPSHOT_FILE)


async def run_service() -> None:
    """Continuous loop -- use this for a VPS/always-on deployment."""
    problems = CONFIG.validate()
    for p in problems:
        logger.warning("Config issue: %s", p)

    scanner = StratScanner(CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key, CONFIG.alpaca_data_feed)
    alert_manager = AlertManager(
        telegram_bot_token=CONFIG.telegram_bot_token,
        telegram_chat_id=CONFIG.telegram_chat_id,
        twilio_account_sid=CONFIG.twilio_account_sid,
        twilio_auth_token=CONFIG.twilio_auth_token,
        twilio_whatsapp_from=CONFIG.twilio_whatsapp_from,
        twilio_whatsapp_to=CONFIG.twilio_whatsapp_to,
    )
    store = StateStore(STATE_DB_PATH)

    logger.info("Strat Scanner service starting. Interval=%ss", CONFIG.scan_interval_seconds)
    while True:
        await scan_cycle(scanner, alert_manager, store)
        await asyncio.sleep(CONFIG.scan_interval_seconds)


async def run_once() -> None:
    """Single scan cycle then return -- use this for GitHub Actions cron,
    or any other external scheduler (cron, Cloud Scheduler, etc.)."""
    problems = CONFIG.validate()
    for p in problems:
        logger.warning("Config issue: %s", p)

    scanner = StratScanner(CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key, CONFIG.alpaca_data_feed)
    alert_manager = AlertManager(
        telegram_bot_token=CONFIG.telegram_bot_token,
        telegram_chat_id=CONFIG.telegram_chat_id,
        twilio_account_sid=CONFIG.twilio_account_sid,
        twilio_auth_token=CONFIG.twilio_auth_token,
        twilio_whatsapp_from=CONFIG.twilio_whatsapp_from,
        twilio_whatsapp_to=CONFIG.twilio_whatsapp_to,
    )
    store = StateStore(STATE_DB_PATH)

    logger.info("Strat Scanner running a single scan cycle (--once mode).")
    await scan_cycle(scanner, alert_manager, store)
    logger.info("Scan cycle complete. Exiting.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strat Scanner service / ticker manager")
    parser.add_argument("--add-ticker", metavar="SYMBOL", help="Add a symbol to the live watchlist and exit")
    parser.add_argument("--remove-ticker", metavar="SYMBOL", help="Remove a symbol from the watchlist and exit")
    parser.add_argument("--list-tickers", action="store_true", help="Print current watchlist and exit")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle then exit (for GitHub Actions / external cron). "
             "Without this flag, runs forever as a continuous service (for a VPS).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.add_ticker:
        tickers = add_ticker(args.add_ticker)
        print(f"Watchlist now: {', '.join(tickers)}")
        return
    if args.remove_ticker:
        tickers = remove_ticker(args.remove_ticker)
        print(f"Watchlist now: {', '.join(tickers)}")
        return
    if args.list_tickers:
        print(", ".join(load_tickers()))
        return

    try:
        if args.once:
            asyncio.run(run_once())
        else:
            asyncio.run(run_service())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
