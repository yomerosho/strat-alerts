"""
main.py
-------
Long-running asyncio service that:
  1. Re-reads the watchlist (tickers.txt) every cycle -> dynamic custom tickers.
  2. Scans every symbol x timeframe concurrently via asyncio.TaskGroup.
  3. Debounces alerts against SQLite-persisted last-known-state.
  4. Sends new/changed setups to Telegram and/or WhatsApp.
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
from alerting import AlertManager, StateStore, format_alert

logger = logging.getLogger("strat_scanner.main")
SNAPSHOT_FILE = BASE_DIR / "latest_scan.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def evaluate_symbol_timeframe(
    scanner: StratScanner,
    alert_manager: AlertManager,
    store: StateStore,
    symbol: str,
    timeframe: str,
) -> Optional[dict]:
    """Returns a JSON-serializable snapshot dict for the dashboard feed
    (or None if the fetch failed / there wasn't enough data)."""
    try:
        # alpaca-py's client is sync; offload to a thread so it doesn't block the loop.
        state: StratState | None = await asyncio.to_thread(scanner.get_state, symbol, timeframe)
    except Exception:
        logger.exception("Failed to fetch/evaluate %s [%s]", symbol, timeframe)
        return None

    if state is None:
        return None

    last_key = store.get_last_setup_key(symbol, timeframe)
    if state.setup_key != last_key:
        # Only alert on something actionable: a live trigger, or a fresh 3-bar
        # sequence change (e.g. a new inside bar forming a setup).
        if state.trigger is not None or last_key is None:
            message = format_alert(state)
            logger.info("New setup for %s [%s]: %s", symbol, timeframe, state.setup_key)
            await alert_manager.send(message)
        store.set_last_setup_key(symbol, timeframe, state.setup_key)

    return state.to_dict()


async def scan_cycle(scanner: StratScanner, alert_manager: AlertManager, store: StateStore) -> None:
    tickers = load_tickers()
    logger.info("Starting scan cycle for %d tickers x %d timeframes", len(tickers), len(CONFIG.timeframes))

    tasks: list[asyncio.Task] = []
    try:
        async with asyncio.TaskGroup() as tg:  # Python 3.11+
            for symbol in tickers:
                for timeframe in CONFIG.timeframes:
                    tasks.append(
                        tg.create_task(
                            evaluate_symbol_timeframe(scanner, alert_manager, store, symbol, timeframe)
                        )
                    )
    except* Exception as eg:  # pragma: no cover -- safety net, individual tasks already self-handle
        for exc in eg.exceptions:
            logger.exception("Unhandled error in scan task", exc_info=exc)

    snapshots = [t.result() for t in tasks if not t.cancelled() and t.result() is not None]
    write_snapshot_file(tickers, snapshots)


def write_snapshot_file(tickers: list[str], snapshots: list[dict]) -> None:
    """Writes the dashboard's data feed: every symbol/timeframe's current
    Strat state, regardless of whether it triggered an alert this cycle."""
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "timeframes": list(CONFIG.timeframes),
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
