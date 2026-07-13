"""
main.py
-------
Orchestrator.

    python main.py --once      # single cycle (GitHub Actions / cron-job.org)
    python main.py             # continuous loop (VPS)
    python main.py --dry-run   # scan, print the board, send nothing

--dry-run is how you should start. Watch the armed levels print for a couple
of sessions and check them against your charts before you let it touch
Telegram. If the 4H levels it prints don't match what you see on TradingView,
something is still wrong and you want to know that before it's shouting at you.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from alerting import AlertManager, AlertStore, format_for_tier
from config import CONFIG, SNAPSHOT_FILE, STATE_DB_PATH, add_ticker, load_tickers, remove_ticker
from levels import ARMED, FAMILY_F2, TIER1, TIER2, ArmedLevel
from scanner import StratScanner

logger = logging.getLogger("strat_scanner.main")

TIER_RANK = {ARMED: 0, TIER1: 1, TIER2: 2}


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def continuity_ok(lv: ArmedLevel, tier: str) -> bool:
    threshold = CONFIG.min_continuity_tier2 if tier == TIER2 else CONFIG.min_continuity_tier1
    if threshold <= 0:
        return True
    try:
        agree = int(lv.continuity.split("/")[0])
    except (ValueError, IndexError):
        return True
    return agree >= threshold


def tiers_to_announce(lv: ArmedLevel) -> list[str]:
    """
    Which alerts does this level currently warrant?

    Every tier the level has REACHED is announced, not just its current one.
    If a scan lands after both the 5m and 15m confirmed (which happens all
    the time on a 5-minute cron), you still get the Tier 1 message. Otherwise
    a fast move would silently skip a tier and the Tier 2 alert would arrive
    with no context for how it got there.

    The ARM alert is proximity-gated: a level 3% away shouldn't ping you at
    9:31. Tier 1 and Tier 2 are never proximity-gated -- they already
    triggered, distance is moot.
    """
    out: list[str] = []
    rank = TIER_RANK[lv.tier]

    if rank == 0:
        if lv.invalidated_by_opposite:
            return []
        near = (
            CONFIG.proximity_alert_pct <= 0
            or abs(lv.distance_pct) <= CONFIG.proximity_alert_pct
        )
        if near:
            out.append(ARMED)
        return out

    if rank >= 1:
        out.append(TIER1)
    if rank >= 2:
        out.append(TIER2)
    return out


async def process_symbol(
    scanner: StratScanner,
    alerts: AlertManager,
    store: AlertStore,
    symbol: str,
    dry_run: bool,
) -> tuple[list[dict], dict]:
    try:
        armed, snapshot = await asyncio.to_thread(scanner.scan_symbol, symbol)
    except Exception:
        logger.exception("Scan failed for %s", symbol)
        return [], {}

    for lv in armed:
        for tier in tiers_to_announce(lv):
            if not continuity_ok(lv, tier):
                logger.info(
                    "%s %s %s: continuity %s below threshold; skipping %s",
                    symbol, lv.setup_tf, lv.pattern, lv.continuity, tier,
                )
                continue

            if store.already_sent(lv.key, tier):
                continue

            message = format_for_tier(lv, tier, CONFIG.failed_two_tier1_actionable)
            if dry_run:
                print("\n" + "-" * 60)
                print(f"[DRY RUN] would send {tier}:")
                print(message)
            else:
                logger.info("ALERT %s %s %s %s", tier, symbol, lv.setup_tf, lv.pattern)
                await alerts.send(message)
                store.mark_sent(lv.key, tier)

    return [lv.to_dict() for lv in armed], snapshot


async def scan_cycle(
    scanner: StratScanner,
    alerts: AlertManager,
    store: AlertStore,
    dry_run: bool = False,
) -> None:
    tickers = load_tickers()
    logger.info("Scanning %d tickers | setup TFs: %s", len(tickers), ", ".join(CONFIG.setup_timeframes))

    results = await asyncio.gather(
        *(process_symbol(scanner, alerts, store, s, dry_run) for s in tickers)
    )

    all_levels: list[dict] = []
    snapshots: list[dict] = []
    for levels, snap in results:
        all_levels.extend(levels)
        if snap:
            snapshots.append(snap)

    # Nearest-to-trigger first. That ordering IS the watchlist -- the top of
    # the board is what's about to happen.
    all_levels.sort(key=lambda d: (-TIER_RANK[d["tier"]], abs(d["distance_pct"])))

    write_snapshot(tickers, all_levels, snapshots)
    store.prune()

    if dry_run:
        print_board(all_levels)


def print_board(levels: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ARMED LEVELS")
    print("=" * 78)
    if not levels:
        print("  (nothing armed)")
        return
    print(f"{'SYM':<6} {'TF':<4} {'PATTERN':<8} {'DIR':<5} {'TIER':<6} "
          f"{'LEVEL':>9} {'PRICE':>9} {'DIST%':>8}  CONT")
    print("-" * 78)
    for d in levels:
        print(
            f"{d['symbol']:<6} {d['setup_tf']:<4} {d['pattern']:<8} "
            f"{d['direction']:<5} {d['tier']:<6} "
            f"{d['level']:>9.2f} {d['current_price']:>9.2f} "
            f"{d['distance_pct']:>+8.2f}  {d['continuity']}"
        )


def write_snapshot(tickers: list[str], levels: list[dict], snapshots: list[dict]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generated_at_et": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "version": 4,
        "tickers": tickers,
        "setup_timeframes": list(CONFIG.setup_timeframes),
        "tier1_timeframe": CONFIG.tier1_timeframe,
        "tier2_timeframe": CONFIG.tier2_timeframe,
        "proximity_alert_pct": CONFIG.proximity_alert_pct,
        "armed_levels": levels,
        "symbols": snapshots,
    }
    SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %d armed levels to %s", len(levels), SNAPSHOT_FILE)


def build() -> tuple[StratScanner, AlertManager, AlertStore]:
    for p in CONFIG.validate():
        logger.warning("Config: %s", p)
    return (
        StratScanner(CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key, CONFIG.alpaca_data_feed),
        AlertManager(CONFIG.telegram_bot_token, CONFIG.telegram_chat_id),
        AlertStore(STATE_DB_PATH),
    )


async def run_once(dry_run: bool) -> None:
    scanner, alerts, store = build()
    await scan_cycle(scanner, alerts, store, dry_run)


async def run_service(dry_run: bool) -> None:
    scanner, alerts, store = build()
    logger.info("Service starting. Interval=%ss", CONFIG.scan_interval_seconds)
    while True:
        await scan_cycle(scanner, alerts, store, dry_run)
        await asyncio.sleep(CONFIG.scan_interval_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strat Scanner v4 — armed levels")
    p.add_argument("--once", action="store_true", help="Single scan cycle, then exit")
    p.add_argument("--dry-run", action="store_true", help="Scan and print; send nothing")
    p.add_argument("--add-ticker", metavar="SYM")
    p.add_argument("--remove-ticker", metavar="SYM")
    p.add_argument("--list-tickers", action="store_true")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.add_ticker:
        print(", ".join(add_ticker(args.add_ticker)))
        return
    if args.remove_ticker:
        print(", ".join(remove_ticker(args.remove_ticker)))
        return
    if args.list_tickers:
        print(", ".join(load_tickers()))
        return

    try:
        if args.once or args.dry_run:
            asyncio.run(run_once(args.dry_run))
        else:
            asyncio.run(run_service(args.dry_run))
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
