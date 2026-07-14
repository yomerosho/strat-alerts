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

import magnitude
from alerting import AlertManager, AlertStore, format_for_tier
from config import CONFIG, SNAPSHOT_FILE, STATE_DB_PATH, add_ticker, load_tickers, remove_ticker
from levels import ARMED, FAMILY_F2, FAMILY_INSIDE, TIER1, TIER2, ArmedLevel
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
    symbol: str,
) -> tuple[list[ArmedLevel], dict]:
    """Scan one symbol. Returns its passing levels and snapshot -- sends
    nothing. Alerting is deferred to scan_cycle so Gate 5 (the watchlist-wide
    budget) can rank every symbol's levels against each other before any go
    out."""
    try:
        armed, snapshot = await asyncio.to_thread(scanner.scan_symbol, symbol)
    except Exception:
        logger.exception("Scan failed for %s", symbol)
        return [], {}
    return armed, snapshot


async def send_level(
    alerts: AlertManager,
    store: AlertStore,
    lv: ArmedLevel,
    dry_run: bool,
) -> None:
    """Emit whatever tier alerts this level currently warrants, subject to the
    continuity gate and the dedup store."""
    for tier in tiers_to_announce(lv):
        if not continuity_ok(lv, tier):
            logger.info(
                "%s %s %s: continuity %s below threshold; skipping %s",
                lv.symbol, lv.setup_tf, lv.pattern, lv.continuity, tier,
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
            logger.info("ALERT %s %s %s %s", tier, lv.symbol, lv.setup_tf, lv.pattern)
            await alerts.send(message)
            store.mark_sent(lv.key, tier)


async def scan_cycle(
    scanner: StratScanner,
    alerts: AlertManager,
    store: AlertStore,
    dry_run: bool = False,
) -> None:
    tickers = load_tickers()
    logger.info("Scanning %d tickers | setup TFs: %s", len(tickers), ", ".join(CONFIG.setup_timeframes))

    results = await asyncio.gather(
        *(process_symbol(scanner, s) for s in tickers)
    )

    all_levels: list[ArmedLevel] = []
    snapshots: list[dict] = []
    for levels, snap in results:
        all_levels.extend(levels)
        if snap:
            snapshots.append(snap)

    # --- Gate 5: hard cap across the whole watchlist, ranked by score ---
    # This is why sending is deferred out of process_symbol: the budget is a
    # property of the ENTIRE scan, not of any one symbol. A brilliant TSLA
    # setup should crowd out a mediocre SPY one, and it can only do that if
    # they are ranked together.
    budgeted = magnitude.rank_and_budget(all_levels, budget=CONFIG.alert_budget)
    for lv in budgeted:
        await send_level(alerts, store, lv, dry_run)

    # The board and snapshot show EVERYTHING that passed the gates, not just
    # the budgeted few -- you still want to see what was in contention.
    level_dicts = [lv.to_dict() for lv in all_levels]
    # Nearest-to-trigger first. That ordering IS the watchlist -- the top of
    # the board is what's about to happen.
    level_dicts.sort(key=lambda d: (-TIER_RANK[d["tier"]], abs(d["distance_pct"])))

    write_snapshot(tickers, level_dicts, snapshots)
    store.prune()

    if dry_run:
        print_board(level_dicts)


def print_board(levels: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("ARMED LEVELS")
    print("=" * 78)
    if not levels:
        print("  (nothing armed)")
        return
    print(f"{'SYM':<6} {'TF':<4} {'PATTERN':<8} {'DIR':<5} {'TIER':<6} "
          f"{'LEVEL':>9} {'PRICE':>9} {'DIST%':>8} {'R:R':>6}  CONT")
    print("-" * 78)
    for d in levels:
        rr = d.get("risk_reward")
        rr_txt = f"{rr:>6.2f}" if rr is not None else "     -"
        flag = "  <-- R:R too low" if (rr is not None and rr < CONFIG.min_risk_reward > 0) else ""
        print(
            f"{d['symbol']:<6} {d['setup_tf']:<4} {d['pattern']:<8} "
            f"{d['direction']:<5} {d['tier']:<6} "
            f"{d['level']:>9.2f} {d['current_price']:>9.2f} "
            f"{d['distance_pct']:>+8.2f} {rr_txt}  {d['continuity']}{flag}"
        )


def write_snapshot(tickers: list[str], levels: list[dict], snapshots: list[dict]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generated_at_et": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "version": 5,
        "tickers": tickers,
        "setup_timeframes": list(CONFIG.setup_timeframes),
        "tier2_timeframe": CONFIG.tier2_timeframe,
        "proximity_alert_pct": CONFIG.proximity_alert_pct,
        "min_risk_reward": CONFIG.min_risk_reward,
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


SAMPLE_TIERS = [
    ("ARM", ARMED, FAMILY_INSIDE, "2-1-2", "bull", "above", 412.50, 408.10, 418.20, 411.05),
    ("TIER 1", TIER1, FAMILY_INSIDE, "2-1-2", "bull", "above", 412.50, 408.10, 418.20, 413.80),
    ("TIER 1 (F2)", TIER1, FAMILY_F2, "F2D", "bear", "below", 184.20, 185.90, 178.40, 183.55),
    ("TIER 2", TIER2, FAMILY_INSIDE, "3-1-2", "bear", "below", 295.75, 296.47, 293.62, 294.90),
]


async def test_telegram(mode: str) -> None:
    """
    Prove the pipe actually works.

    `sample` builds one of each alert type from fake data and sends it. This
    is not just a "hello world" -- the alert text uses *bold* and _italic_,
    and Telegram's legacy Markdown parser silently REJECTS the entire message
    if a marker is unbalanced. A dry run can't catch that; only a real send
    can. If a message doesn't arrive, that's the bug.

    `live` takes whatever is armed right now and force-sends its alert,
    bypassing the proximity, R:R, continuity, and dedup gates. Nothing is
    recorded to the store, so your real alerts are unaffected.
    """
    import pandas as pd

    alerts = AlertManager(CONFIG.telegram_bot_token, CONFIG.telegram_chat_id)

    if not CONFIG.telegram_bot_token or not CONFIG.telegram_chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set. Nothing to test.")
        return

    print(f"Sending to {len(CONFIG.telegram_chat_id)} chat id(s): "
          f"{', '.join(CONFIG.telegram_chat_id)}")
    failed: list[str] = []

    await alerts.send(
        "🧪 *Strat Scanner v4 — alert test*\n"
        f"Mode: `{mode}`. The next few messages are tests, not signals."
    )

    if mode == "sample":
        now = pd.Timestamp.now(tz="America/New_York")
        close = now.normalize() + pd.Timedelta(hours=16)
        for name, tier, family, pattern, direction, side, lvl, inval, tgt, px in SAMPLE_TIERS:
            sample = ArmedLevel(
                symbol="TEST", setup_tf="4H", family=family, pattern=pattern,
                direction=direction, trigger_side=side, level=lvl,
                invalidation=inval, target=tgt,
                setup_bar_time=now, arm_time=now, current_price=px,
                setup_bar_closes_at=close, continuity="4/5", tier=tier,
                tier1_time=now if tier in (TIER1, TIER2) else None,
                tier2_time=now if tier == TIER2 else None,
                minutes_to_next_15m=10,
            )
            msg = format_for_tier(sample, tier, CONFIG.failed_two_tier1_actionable)
            ok = await alerts.send(msg)
            print(f"\n--- {name}  [{'SENT' if ok else 'FAILED'}] ---\n{msg}")
            if not ok:
                failed.append(name)

        if failed:
            print(f"\n{len(failed)} alert type(s) FAILED to send: {failed}")
            print("Almost certainly unbalanced Markdown (* or _) in that format.")
            sys.exit(1)
        await alerts.send("🧪 Test complete — all 4 alert types delivered.")
        print("\nAll 4 alert types delivered.")
        return

    # --- live mode ---
    scanner = StratScanner(CONFIG.alpaca_api_key, CONFIG.alpaca_secret_key,
                           CONFIG.alpaca_data_feed)
    tickers = load_tickers()
    sent = 0
    for symbol in tickers:
        armed, _ = await asyncio.to_thread(scanner.scan_symbol, symbol)
        for lv in armed:
            msg = format_for_tier(lv, lv.tier, CONFIG.failed_two_tier1_actionable)
            ok = await alerts.send(msg)
            print(f"\n--- {lv.symbol} {lv.setup_tf} {lv.pattern} {lv.tier} "
                  f"[{'SENT' if ok else 'FAILED'}] ---\n{msg}")
            if ok:
                sent += 1
            else:
                failed.append(f"{lv.symbol} {lv.pattern}")

    await alerts.send(
        f"🧪 Test complete. {sent} live level(s) sent, all gates bypassed "
        f"(proximity, R:R, continuity, dedup). Nothing was recorded — your real "
        f"alerts are unaffected."
    )
    print(f"\nSent {sent} live levels.")
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strat Scanner v4 — armed levels")
    p.add_argument("--once", action="store_true", help="Single scan cycle, then exit")
    p.add_argument("--dry-run", action="store_true", help="Scan and print; send nothing")
    p.add_argument("--add-ticker", metavar="SYM")
    p.add_argument("--remove-ticker", metavar="SYM")
    p.add_argument("--list-tickers", action="store_true")
    p.add_argument(
        "--test-telegram", choices=["sample", "live"], metavar="MODE",
        help="sample = one of each alert type from fake data. "
             "live = every currently-armed level, all gates bypassed. "
             "Neither records anything to the dedup store.",
    )
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
    if args.test_telegram:
        asyncio.run(test_telegram(args.test_telegram))
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
