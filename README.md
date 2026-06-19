# Strat Scanner — Built for 0DTE Confluence Trading

Detects "The Strat" (Rob Smith) patterns and alerts you only when:
1. **FTFC (Full Timeframe Continuity)** agrees across **15Min / 30Min / 1H / 4H / 1D**, and
2. A live trigger fires on an **entry timeframe** (**5Min / 15Min**) in that same direction.

That's the actual confluence signal you'd act on — not a generic single-timeframe blip. The engine (GitHub Actions) checks this every 5 minutes and pushes a Telegram/WhatsApp alert when it lines up. The dashboard (Streamlit) shows the same logic live for ad-hoc checks, with a "Live Confluence Signals" section right at the top showing exactly what would have alerted you.

## Why no 1W/1M

Removed entirely. Weekly/monthly bias doesn't matter when you're flat by end of day on 0DTE — only the higher *intraday* timeframes (up through Daily) matter for bias, and the fast timeframes matter for the actual entry trigger.

## Architecture

```
config.py     Watchlist + settings. TIMEFRAMES = 5Min/15Min/30Min/1H/4H/1D.
              FTFC_TIMEFRAMES = 15Min/30Min/1H/4H/1D. ENTRY_TIMEFRAMES = 5Min/15Min.
scanner.py    Strat labeling (1/2U/2D/3), trigger detection, Alpaca fetch.
              bar_period_has_closed() correctly handles intraday/daily bar
              boundaries so "last bar" timestamps are never stale.
main.py       Per-symbol: fetches all 6 timeframes, computes FTFC, checks
              entry timeframes for a matching trigger, alerts only on that
              confluence (debounced so it won't re-fire on the same state).
alerting.py   Telegram + WhatsApp (Twilio) delivery, SQLite debounce store.
.github/workflows/strat-scanner.yml   Cron every 5 minutes (fastest
              practical free-tier interval)
dashboard/app.py   Password-gated Streamlit viewer: live confluence
              signals up top, per-timeframe tabs with filters below.
```

## Default watchlist

`SPY QQQ IWM DIA AAPL AMD AMZN GOOG GOOGL META MSFT NVDA PLTR TSLA` — mostly index ETFs and mega-caps, which is where 0DTE liquidity actually lives. Edit `tickers.txt` to change it.

---

## Alerting setup
See `.env.example` for Telegram and Twilio WhatsApp setup instructions (same as before — nothing changed there).

---

## Deploying

### GitHub Actions (engine)
Same process as before — push to GitHub, add secrets (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `TELEGRAM_*` and/or `TWILIO_*`), enable Actions, manually trigger once to confirm, then it runs itself every 5 minutes.

Optional: `ALPACA_DATA_FEED=sip` if you have a subscription that includes the full consolidated tape (e.g. Algo Trader Plus) — otherwise it defaults to IEX-only data.

### Streamlit dashboard (viewer)
Same as before — deploy from `dashboard/app.py`, set `APP_PASSWORD` / `GITHUB_OWNER` / `GITHUB_REPO` / `GITHUB_BRANCH` as Streamlit secrets.

---

## Cost / limits reality check

- **GitHub Actions:** free/unlimited on a public repo. On a private repo, 2,000 free minutes/month. At every 5 minutes (~288 runs/day) with 14 tickers × 6 timeframes per run, each run still finishes in well under a minute, so you'll stay inside the free tier for personal use — but it's worth keeping an eye on usage if you significantly expand the watchlist.
- **GitHub Actions cron isn't millisecond-precise.** Expect occasional multi-minute slippage during platform load. For a strategy where you're in and out within 30 minutes, treat alerts as "go check your chart now," not a guaranteed instant signal. If you need tighter timing than free Actions can offer, the VPS continuous-loop mode (`python main.py`, no `--once` flag) can poll every 30-60 seconds instead — ask if you want that walkthrough.
- Alpaca's IEX feed can miss brief high/low prints from other exchanges, which can occasionally cause a Strat label to differ from what a full-tape platform (like TradingView) shows. Switch to `ALPACA_DATA_FEED=sip` if you have that entitlement.
- This is detection/alerting only — no order/trade execution logic anywhere.
- 4H bars are synthetic (built from 1H bars), anchored to the 9:30 ET open.

## If you ever want to scale up to a VPS instead

`python main.py --once` → single scan, used by GitHub Actions. `python main.py` (no flag) → continuous service for a VPS with systemd (`strat-scanner.service` included). Same dual-mode design as before.
