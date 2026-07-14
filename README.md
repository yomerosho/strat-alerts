# Strat Scanner v5 — Armed Levels

A gated confluence detector for "The Strat." It watches a watchlist during
market hours, arms levels off **4H and Daily** setups, runs each through a hard
gate stack, and sends at most a handful of ranked alerts to Telegram — each one
carrying a complete **scale-out** trade plan. It does not execute, size, or pick
your strike. The trade is yours.

```
NOMINATE ─▶ RUNWAY ─▶ CONTINUITY ─▶ 15m CONFIRM ─▶ BUDGET ─▶ 📱
  4H/1D      ≥ 2R       ≥ 4/5 FTFC     TIER 2        top-N
```

---

## What v5 inverts

v4 armed on 2H/4H, computed a same-timeframe target, and alerted. It never asked
the only question that separates a scalp from a swing: *once price reaches the
target, is there anything left above it?* v5 flips the whole flow:

> A level is not something the scanner **finds**. It is something a higher
> timeframe **nominates**, and everything below it confirms.

- **4H / Daily** nominate the level *(Gate 1)*
- **the ladder** says how far it can run *(Gate 2 — runway)*
- **continuity** says whether to hold it *(Gate 3 — a hard gate now, not a printed score)*
- **the 15m close** says when to enter *(Tier 2)*
- **the budget** says how many you get *(Gate 5)*

Expect it to go quiet. A scanner that sends nothing on a chop day is working
correctly. That is the entire point.

---

## The gate stack

A nominated setup must clear **every** gate, in order. Most die at Gate 2.

| Gate | Condition |
|---|---|
| **1 — Nominate** | Only a **4H or Daily** inside-bar / Failed-2 may originate a level. 2H/1H/15m/5m never nominate. |
| **2a — Min risk** | Stop must be **≥ 0.15% of price** away — kills near-zero-stop setups that fake an absurd R. |
| **2 — Runway** | The **nearest untouched higher-TF level** (the magnitude ladder: 4H/1D/1W) must be **≥ `MIN_RUNWAY_R` (2R)** beyond the trigger. No room ⇒ no trade. |
| **3 — Continuity** | The forming **4H and Daily** bars must **both** agree with the trade direction. Hard gate. |
| **3b — FTFC** | At least **`MIN_FTFC` (4)** of **1H/2H/4H/1D/1W** aligned with the trade. |
| **5 — Budget** | Across the whole watchlist, only the **top `ALERT_BUDGET` (10)** by score alert per scan. |

**Score** (the budget's sort key) = 4H+1D continuity + FTFC aligned + 2×nested
timeframes + untouched gating rungs + compression bonus, ×2 if runway > 4R (×1.5
if ≥ 2R). It's an ordering heuristic, not a validated probability.

---

## What it nominates — two families, on 4H/Daily only

**Family A — inside-bar setups (2-1-2 / 3-1-2 / 1-1-2).** The last closed 4H/Daily
bar is an inside bar; **both edges arm** (high → long, low → short). Trigger =
a close beyond the edge; stop = the opposite edge (usually a tight range, which
is what makes the R math work). The name is just the bar before the `1`.

**Family B — Failed-2 (F2U / F2D).** The only setup that arms mid-bar. The
forming bar pokes through the prior bar's extreme, then fails back through it,
trapping the breakout crowd. Trigger = a close back inside; stop = the failed
extreme. F2s move fast — they treat **Tier 1 as actionable**, not a heads-up.

*(v4's 3-2-2, 1-2-2 Rev Strat, and PMG are gone — they aren't in the v5 model.)*

---

## The tiers (and the 5m tier is gone)

```
ARMED ──price through intrabar──▶ TIER 1 ──15m closes through──▶ TIER 2
 👀 loaded, proximity-gated        ⚡ live heads-up               🎯 the entry
```

- **ARMED** — cleared every gate, waiting. Pings only if price is within
  `PROXIMITY_ALERT_PCT` (0.75%) of the trigger.
- **Tier 1** — price is *currently* through the trigger, 15m not yet closed. A
  heads-up (except for F2, where it's actionable). Carries the minutes to the
  next 15m close.
- **Tier 2** — a 15-minute bar **closed** through the trigger. This is the entry,
  and nothing else is. The old 5-minute Tier 1 flipped back inside too often to
  trust.

**Freshness gate:** a Tier-1/Tier-2 is only sent if its confirming bar closed
within `MAX_ALERT_AGE_MIN` (30). A level that confirmed an hour ago and already
ran is **dropped, not delivered late** — no more "10:00 signal at 11:29."

---

## The scale-out exit (this is the edge)

Every Tier-2 alert carries the same plan, and it is not optional flavor — on the
backtest it turned a 43%-win entry into a **~71%** one:

> **Take half off at +1R, move the stop to breakeven, run the rest to the first
> untouched higher-TF level.**

Once +1R is banked and the stop is at breakeven, the runner can only win or
scratch — never lose a full R. `ArmedLevel.scale_level` / `runner_r` expose it;
`alerting._scale_plan` prints it; `replay.resolve()` measures it.

---

## What to expect (honest backtest)

Validated over **12 months, 14 symbols, ~1,300 trades**, resolved from the
**actual fill** (no measuring tricks), scale-out exit:

| | value |
|---|---|
| Win rate | **71.2%** |
| Expectancy | **+0.42R / trade** |
| Per-quarter win rate | **70.6% – 71.5%** (stable — not curve-fit) |

Read the two columns separately: **win rate is regime-stable, expectancy is
trend-dependent** (you still win ~71% in chop by banking the +1R scale; the
runners just don't run). Calibrate to these annual numbers — a short recent
window looked rosier (76% / +0.65R) and would mislead you.

**All R is measured on the underlying (shares).** Options add theta/non-linearity
the plan doesn't account for. Median hold ~1 day but the tail runs to ~2 weeks,
so **1–3 DTE fights this strategy** — prefer shares/ETFs or **10–21 DTE,
higher-delta** contracts.

---

## Why the candles are RTH-only

Everything is built from **5-minute RTH bars**, bucketed from the 09:30 session
open (`bars.py`). v3 bucketed Alpaca's hourly bars from a 09:30 anchor and
silently dropped the market open + folded in after-hours tape — every 4H signal
it ever produced was on the wrong candles. 5m bars nest exactly into
15m/30m/1H/2H/4H boundaries, so nothing straddles an edge.

Premarket tape is read for **current price only** — never fed into the buckets.
That's a deliberate, load-bearing decision, and it's why a premarket 4H F2
cannot exist (there is no forming 4H bar before the bell). The **pre-open brief**
(`premarket.py`) maps what *is* set up at the open instead.

> **Verify before trusting anything.** Run `python main.py --dry-run`, take the
> 4H/Daily levels it prints, and put them next to your TradingView chart. They
> must match.

---

## State is a pure function of bar data

Armed levels and tier status are re-derived from scratch every scan — nothing is
carried forward. The scanner runs on a fresh GitHub Actions VM each cycle, so
persisted state was always the fragile part. The SQLite store tracks exactly one
thing: *"have I already sent this alert?"* Dedup is by level identity
(`symbol | setup_tf | setup_bar_time | pattern | direction`), which doesn't
flicker — price can chop across the trigger all day without spamming you.

---

## Files

| File | What it does |
|---|---|
| `magnitude.py` | **The gate stack.** Open magnitude, runway, continuity, FTFC, scoring, budget. |
| `bars.py` | Fetching + session-aligned RTH resampling. |
| `strat.py` | Bar labeling (1 / 2U / 2D / 3), continuity. Pure functions. |
| `levels.py` | `ArmedLevel`, the two arming families, tiers, scale-out geometry. |
| `scanner.py` | Per-symbol orchestration: fetch → label → arm → gate → tier. |
| `alerting.py` | Telegram + dedup store + the three alert formats + scale-out plan. |
| `main.py` | Orchestrator: budget, freshness gate, snapshot, alert dispatch. |
| `config.py` | All config + the watchlist loader. |
| `premarket.py` | Pre-open brief: 4H/Daily F2 map + FTFC stack + overnight magnets. |
| `replay.py` | Forward-walk backtester — *calls* the live scanner, structural anti-lookahead. |
| `exit_sweep.py` / `ftfc_sweep.py` | Exit-policy and FTFC tradeoff experiments. |
| `dashboard/app.py` | Streamlit board (runway / FTFC / score / scale-out, with filters). |
| `pine/strat_v5_companion.pine` | TradingView companion indicator. |
| `tickers.txt` | The watchlist, one symbol per line. |
| `test_v4.py` | v4-era tests — **expected to fail under v5, left as-is.** |

One Alpaca call per symbol for intraday (everything derives from a single 5m
pull) plus one daily pull — so a 15m close and a 4H high can never disagree.

---

## Running it

```bash
python main.py --dry-run          # scan, print the board + would-be alerts, send nothing
python main.py --once             # one live cycle (GitHub Actions / cron)
python main.py --test-telegram sample   # prove the Telegram pipe end to end
python replay.py SPY QQQ NVDA TSLA --days 365   # backtest (needs Alpaca keys in .env)
python premarket.py --dry-run     # print the pre-open brief
```

**Deployment.** The scanner and pre-open brief are `workflow_dispatch`-only
GitHub Actions (GitHub's native cron disables itself on quiet repos). Trigger
them from **cron-job.org**, scoped to **09:30–16:00 ET, Mon–Fri** for the
scanner (~60s after each boundary so bars settle) and **~08:30 ET** for the
brief. The watchlist is read from `tickers.txt` **on `main`** — edit it there.

---

## Config knobs

| Env var | Default | Notes |
|---|---|---|
| `ALPACA_DATA_FEED` | `sip` | SIP (consolidated tape) matches TradingView. |
| `MIN_RUNWAY_R` | `2.0` | Gate 2 — nearest gating rung must be ≥ this many R. |
| `MIN_FTFC` | `4` | Gate 3b — frames of 1H/2H/4H/1D/1W that must align. |
| `ALERT_BUDGET` | `10` | Gate 5 — max alerts per scan, ranked by score. Raise for a big watchlist. |
| `MAX_ALERT_AGE_MIN` | `30` | Freshness gate — drop tier alerts older than this. 0 disables. |
| `PROXIMITY_ALERT_PCT` | `0.75` | ARM alerts only within this % of the trigger. |
| `F2_TIER1_ACTIONABLE` | `true` | Mark Failed-2 Tier 1 as act-now (a judgment call, not a proven edge). |
| `ENABLE_FAILED_TWO` | `true` | Set `false` to run inside-bar setups only. |

*(`MIN_CONTINUITY_TIER1/2` still exist but default to 0 — the real continuity
gate is Gates 3/3b now, upstream of the alert.)*

---

## Yours to decide, not the tool's

- **The tool flags mechanically-valid setups; it does not read trend structure.**
  FTFC continuity is current-bar alignment, not trend — a strong bounce can print
  5/5 inside a downtrend. When the tool says long but the structure says down,
  that's a counter-trend bounce: your discretionary overlay, not a bug.
- **DTE and size are yours.** The scanner gives direction, stop, +1R, and runner.
  Log your own time-to-target before trusting any expiry rule.
