# Strat Scanner v4 — Armed Levels

## What changed and why

v3 alerted on **completed** patterns. A confirmed 4H 2-1-2 means the third candle already closed — the move is over. You were being notified about trades you'd already missed.

v4 detects setups while they're still **armed**, publishes the trigger level, and stages alerts as lower timeframes confirm.

```
ARMED ──5m close through──▶ TIER 1 ──15m close through──▶ TIER 2
 👀                          ⚡                             🎯
```

Setups arm on **2H and 4H only**. Nothing arms on 15m or below anymore. That's the noise reduction.

---

## 🔴 The bug you were trading on

Your v3 `resample_to_4h` built 4H bars by bucketing Alpaca's **1-Hour** bars from a 09:30 anchor. Two things broke it:

1. Alpaca stamps hourly bars **on the hour** (09:00, 10:00, …). The 09:00 bar is the one that *contains* the 09:30 open — and it was filtered out because `09:00 < 09:30`.
2. Alpaca returns **extended-hours** bars by default (04:00–20:00 ET). Nothing "naturally truncated at 16:00" the way the docstring assumed.

Actual output:

```
"09:30" 4H bar  →  contained 10:00, 11:00, 12:00, 13:00   ← no market open
"13:30" 4H bar  →  contained 14:00, 15:00, 16:00, 17:00   ← 2h of after-hours
"17:30" 4H bar  →  phantom bar of pure evening tape
```

**Every 4H signal the scanner ever produced was computed on the wrong candles.** Not slightly off — the opening range was absent entirely.

**The fix:** build everything from **5-minute** bars, explicitly filter to RTH, and bucket from the 09:30 session open. 5m bars nest exactly into 15m/30m/1H/2H/4H boundaries, so no bar straddles an edge. Session-aligned buckets:

```
4H:  09:30–13:30 │ 13:30–16:00               (2nd is a 2.5h stub)
2H:  09:30–11:30 │ 11:30–13:30 │ 13:30–15:30 │ 15:30–16:00
```

The trailing stub is real and is kept — it's what TradingView shows, and a lot of afternoon resolution happens there.

**Verify this before trusting anything.** Run `python main.py --dry-run`, take the 4H levels it prints, and put them next to your TradingView chart. They must match. If they don't, stop and tell me.

---

## The core abstraction

Every Strat setup collapses to one object:

> *"If a bar closes on THIS SIDE of THIS LEVEL, the trade is on."*

Six named patterns, one machine. The pattern name is a **label on the alert**, not a branch in the logic.

### Family A — inside-bar setups (2-1-2, 3-1-2, 1-1-2)
Arms off the **last CLOSED** setup bar. An inside bar has no direction, so **both** edges arm: high → bullish breakout, low → bearish breakdown. Trigger = close **beyond** the level.

### Family B — Failed 2 (F2U / F2D)
Arms off the **currently FORMING** setup bar. Price pokes through the prior bar's high, fails, and traps the breakout buyers. Trigger = close **back inside** the level.

That's why `trigger_side` is an explicit field: Family A closes *away* from the level, Family B closes *back through* it.

F2 also tracks the **breach time** — only 5m closes *after* the poke count as a failure. Price was below the prior high before the poke too; that doesn't mean anything.

---

## State is a pure function of bar data

Armed levels and tier status are re-derived from scratch every scan. Nothing is carried forward.

This matters because the scanner runs on a **fresh GitHub Actions VM every cycle** — persisted state was always the fragile part. The SQLite DB now tracks exactly one thing: *"have I already sent this alert?"*

Expiry falls out for free. If the last closed setup bar is no longer an inside bar, no level arms. Nothing to expire, nothing to leak.

Dedup is by **level identity** — `symbol | setup_tf | setup_bar_timestamp | pattern | direction`. That identity doesn't flicker, so price can chop across the trigger all afternoon and you will not be spammed. The v3 cooldown hack is gone.

---

## Two things I'd flag as *yours* to decide, not mine

**1. F2 at Tier 1.** `F2_TIER1_ACTIONABLE=true` marks Failed-2 Tier 1 alerts as 🔥 act-now, on the theory that trapped-trader unwinds move fast enough that waiting for the 15m hands back most of the move. **That's a plausible mechanism, not a proven edge.** Watch it live for a few weeks. If F2 Tier 1s keep failing, set it `false` and F2 waits for the 15m like everything else.

**2. DTE selection.** Nothing in this code picks your expiry. A 4H setup targeting the prior 4H high can take a full session to resolve, so +1/+2 DTE is directionally right — but log actual time-to-target on every trade for a month and set the rule from your own data, not a rule of thumb.

---

## Files

| File | What it does |
|---|---|
| `bars.py` | Fetching + session-aligned resampling. **The bug fix lives here.** |
| `strat.py` | Bar labeling (1 / 2U / 2D / 3), continuity score. Pure functions. |
| `levels.py` | `ArmedLevel`, the two arming families, tier promotion. The core. |
| `scanner.py` | Per-symbol orchestration. |
| `alerting.py` | Telegram + dedup store + the three alert formats. |
| `main.py` | Entry point. |
| `test_v4.py` | 30 checks, no network. Includes a regression test for the 4H bug. |
| `dashboard_app.py` | Streamlit board → rename to `dashboard/app.py` |
| `strat-scanner.yml` | → `.github/workflows/strat-scanner.yml` |

Also: one Alpaca call per symbol instead of six. v3 made a request per (symbol, timeframe) — 84 per scan on a 14-symbol list. Now everything intraday is derived from one 5-minute pull, which also guarantees a 15m close and a 4H high can never disagree with each other.

---

## Rollout

```bash
# 1. Tests first — they should all pass before anything touches the market
python test_v4.py

# 2. Dry run: prints the board and every alert it WOULD send. Sends nothing.
python main.py --dry-run

# 3. Check the 4H levels against TradingView. Do not skip this.

# 4. Wipe v3's alert state (schema changed) and go live
rm scanner_state.db
python main.py --once
```

**cron-job.org:** keep the 5-minute cadence — it matches the Tier 1 (5m close) rhythm. Fire ~60s *after* each 5-minute boundary so the bar has settled at Alpaca. Scope it to **09:30–16:00 ET, Mon–Fri** — it's currently firing at all hours.

---

## Config knobs

| Env var | Default | Notes |
|---|---|---|
| `PROXIMITY_ALERT_PCT` | `0.75` | Only send 👀 ARM alerts when price is within this % of the level. Tier 1/2 are never proximity-gated — they already triggered. |
| `MIN_CONTINUITY_TIER1` | `0` (off) | v3's 4/5 gate was silently eating setups. With 2H/4H-only you have far fewer, better candidates — start with the gate off and only turn it on if the board gets busy. |
| `MIN_CONTINUITY_TIER2` | `0` (off) | Same. |
| `F2_TIER1_ACTIONABLE` | `true` | See above. |
| `ENABLE_FAILED_TWO` | `true` | Set `false` to ship Family A first and add F2 once you trust it. |
