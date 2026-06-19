"""
main.py
-------
Long-running asyncio service that:
  1. Re-reads the watchlist (tickers.txt) every cycle -> dynamic custom tickers.
  2. For each symbol, fetches every configured timeframe concurrently.
  3. Computes FTFC (Full Timeframe Continuity) across the FTFC timeframes
     (15Min/30Min/1H/4H/1D by default) and checks the entry timeframes
     (5Min/15Min by default) for a trigger matching that direction.
  4. Sends a confluence alert to Telegram only when FTFC agrees
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
from alerting import AlertManager, StateStore, format_watch_alert, format_entry_alert

logger = logging.getLogger("strat_scanner.main")
SNAPSHOT_FILE = BASE_DIR / "latest_scan.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


TIMEFRAME_ORDER = ["5Min", "15Min", "30Min", "1H", "4H", "1D"]


def next_timeframe_up(timeframe: str) -> Optional[str]:
    """The next entry in TIMEFRAME_ORDER, or None if already at the top.
    Used for the classic Strat "broadening" target convention: a signal on
    one timeframe targets the opposing extreme of the timeframe above it."""
    try:
        idx = TIMEFRAME_ORDER.index(timeframe)
    except ValueError:
        return None
    return TIMEFRAME_ORDER[idx + 1] if idx + 1 < len(TIMEFRAME_ORDER) else None


def compute_continuity_score(states: dict[str, StratState], direction: str) -> str:
    """e.g. '4/5' -- how many FTFC timeframes currently agree with `direction`.
    Reported as context alongside an alert, not used to gate it -- matches
    the traditional use of FTFC as a bias filter rather than a strict
    all-or-nothing requirement."""
    relevant = [tf for tf in CONFIG.ftfc_timeframes if tf in states]
    if not relevant:
        return "0/0"
    agree = sum(1 for tf in relevant if states[tf].direction == direction)
    return f"{agree}/{len(relevant)}"


def compute_target(states: dict[str, StratState], timeframe: str, direction: str) -> Optional[float]:
    """Target = the next-timeframe-up's opposing extreme. Walks further up
    if that timeframe's data wasn't available this cycle (e.g. not enough
    bar history yet)."""
    nxt = next_timeframe_up(timeframe)
    while nxt:
        state = states.get(nxt)
        if state:
            return state.last_completed_high if direction == "bull" else state.last_completed_low
        nxt = next_timeframe_up(nxt)
    return None


def find_pmg_note(states: dict[str, StratState]) -> str:
    """If any timeframe currently shows an active PMG streak, surface it as
    supporting context on whichever alert is firing -- PMG is a warning
    flag, never a standalone alert by itself."""
    for tf, state in states.items():
        for pattern in state.patterns:
            if pattern.name == "PMG":
                return f"{pattern.note} ({tf})"
    return ""


async def evaluate_symbol(
    scanner: StratScanner,
    alert_manager: AlertManager,
    store: StateStore,
    symbol: str,
) -> list[dict]:
    """Fetches every configured timeframe for one symbol, scans each for
    named Strat setups (Failed-2, 2-1-2, 3-1-2, 3-2-2, 1-2-2 Rev Strat),
    and sends:
      - a WATCH alert when an actionable pattern forms on a higher
        timeframe (CONFIG.pattern_watch_timeframes) -- anticipatory, before
        any entry-timeframe trigger exists.
      - an ENTRY alert when an actionable pattern prints directly on an
        entry timeframe (CONFIG.entry_timeframes) -- the "go" signal, with
        a stop from the pattern itself and a target borrowed from the next
        timeframe up.
    Each (timeframe, pattern name) combination is debounced and
    cooldown-gated independently, so multiple distinct patterns firing in
    the same cycle don't clobber each other's tracking.
    """
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

    pmg_note = find_pmg_note(states)

    async def maybe_send(tf: str, pattern, alert_kind: str, message: str) -> None:
        # Pattern name included in the store key -- distinct patterns on the
        # same timeframe must never share debounce/cooldown tracking.
        store_key = f"{alert_kind}:{tf}:{pattern.name}"
        setup_key = f"{pattern.direction}"
        last_key = store.get_last_setup_key(symbol, store_key)
        if setup_key == last_key:
            return

        minutes_since = store.minutes_since_last_alert(symbol, store_key)
        in_cooldown = minutes_since is not None and minutes_since < CONFIG.alert_cooldown_minutes
        if in_cooldown:
            logger.info(
                "Suppressing %s alert for %s [%s/%s] (cooldown %.1f/%d min)",
                alert_kind, symbol, tf, pattern.name, minutes_since, CONFIG.alert_cooldown_minutes,
            )
        else:
            logger.info("%s alert for %s [%s/%s]: %s", alert_kind, symbol, tf, pattern.name, setup_key)
            await alert_manager.send(message)
            store.record_alert_sent(symbol, store_key)
        store.set_last_setup_key(symbol, store_key, setup_key)

    # WATCH -- anticipatory, on the higher timeframes
    for tf in CONFIG.pattern_watch_timeframes:
        state = states.get(tf)
        if not state:
            continue
        for pattern in state.patterns:
            if not pattern.actionable:
                continue
            score = compute_continuity_score(states, pattern.direction)
            message = format_watch_alert(symbol, tf, pattern, score, pmg_note)
            await maybe_send(tf, pattern, "WATCH", message)

    # ENTRY -- the actual "go" signal, on the entry timeframes
    for tf in CONFIG.entry_timeframes:
        state = states.get(tf)
        if not state:
            continue
        for pattern in state.patterns:
            if not pattern.actionable:
                continue
            target = compute_target(states, tf, pattern.direction)
            score = compute_continuity_score(states, pattern.direction)
            message = format_entry_alert(symbol, tf, pattern, target, score, pmg_note)
            await maybe_send(tf, pattern, "ENTRY", message)

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
        "pattern_watch_timeframes": list(CONFIG.pattern_watch_timeframes),
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
