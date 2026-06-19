# OBI Strat Scanner — GitHub Actions Engine + Streamlit Dashboard

A scanner that detects "The Strat" (Rob Smith) patterns on the 4-Hour,
Daily, Weekly, and Monthly timeframes for your watchlist. Two pieces work
together:

1. **The engine** (`main.py` + GitHub Actions) runs every 15 minutes for
   free, fetches Alpaca data, detects setups, and pushes alerts to
   **Telegram** and/or **WhatsApp** — only when something's new or a
   trigger actually fires.
2. **The dashboard** (`dashboard/app.py`, a password-protected Streamlit
   app) reads the engine's latest results and shows them as a live table,
   like your original screenshot. It does no scanning itself — it just
   displays what the engine already found.

This split exists because Streamlit apps aren't persistent background
workers (they only run while a page is open, and sleep when idle) — they
can't autonomously wake up and fire alerts on a timer the way GitHub
Actions' cron trigger can. So GitHub Actions stays the "always-on" piece,
and Streamlit is purely the viewer, matching the pattern of your other
Streamlit apps with password protection.

## Architecture

```
config.py     Env-based settings + dynamic watchlist (tickers.txt)
scanner.py    Strat labeling (1 / 2U / 2D / 3), trigger detection,
              Alpaca data fetch, 4H resampling anchored to 9:30 ET open
alerting.py   AlertManager (Telegram + WhatsApp via Twilio), SQLite
              StateStore for debounce
main.py       --once mode (one scan cycle, for GitHub Actions) and a
              continuous loop mode (for a VPS, if you ever scale up).
              Writes latest_scan.json after every run.
.github/workflows/strat-scanner.yml   The 15-minute cron job
dashboard/app.py             Password-gated Streamlit viewer
dashboard/requirements.txt   Streamlit's own deps (kept separate from the
                              engine's requirements.txt so Streamlit Cloud
                              doesn't need to install alpaca-py)
```

Each engine run:
1. Reads `tickers.txt` for the current watchlist.
2. Scans every (symbol, timeframe) pair concurrently.
3. Compares each new Strat state to the last-alerted state in
   `scanner_state.db` (SQLite) and alerts only on a live trigger or a
   freshly-changed 3-bar sequence.
4. Writes `latest_scan.json` — a full snapshot of every symbol/timeframe's
   current state, used by the dashboard.
5. Commits `scanner_state.db`, `tickers.txt`, and `latest_scan.json` back to
   the repo (GitHub Actions runners are stateless, so without this step
   you'd get duplicate alerts and the dashboard would never update).

The dashboard fetches `latest_scan.json` directly from
`raw.githubusercontent.com` on every page load — always current, with no
dependency on Streamlit Cloud's own redeploy timing.

## Default watchlist

Seeded from your screenshots: `SPY QQQ IWM DIA AAPL AMD AMZN GOOG GOOGL META
MSFT NVDA PLTR TSLA` (see `tickers.txt`).

### Adding/removing tickers
Edit `tickers.txt` directly on GitHub's web UI (click the file → pencil icon
→ edit → commit), or locally via:
```bash
python main.py --add-ticker COIN
python main.py --remove-ticker PLTR
```
The next scheduled run picks it up automatically.

---

## Alerting setup

### Telegram (free, no approval needed)
1. Message **@BotFather** on Telegram → `/newbot` → follow the prompts → you get a `TELEGRAM_BOT_TOKEN`.
2. Message **@userinfobot** to get your numeric `TELEGRAM_CHAT_ID`.
3. Send your new bot at least one message first (required before it can message you back).

### WhatsApp (via Twilio)
1. Free account at https://www.twilio.com.
2. **Messaging → Try it out → Send a WhatsApp message** → join the sandbox (text the join code from your phone).
3. Copy `Account SID` and `Auth Token` from the Twilio console.
4. `TWILIO_WHATSAPP_FROM=whatsapp:+14155238886` (Twilio's sandbox number), `TWILIO_WHATSAPP_TO=whatsapp:+1XXXXXXXXXX` (your phone).
5. Sandbox needs re-joining every ~72 hours — fine for personal use.

Enable one or both — leave the unused values blank.

---

## Part 1: Deploy the engine on GitHub Actions

### Push the code to GitHub
Using the web UI: create a new repo, upload all the root-level files, then
use **Add file → Create new file** and type the path
`.github/workflows/strat-scanner.yml` to create that nested file (paste in
its contents). Do the same for `dashboard/app.py` and
`dashboard/requirements.txt` if you didn't upload them as a folder.

This repo intentionally does **not** gitignore `scanner_state.db`,
`tickers.txt`, or `latest_scan.json` — the workflow commits changes to all
three so state, watchlist edits, and dashboard data persist between runs.

### Add secrets
Repo **Settings → Secrets and variables → Actions → New repository secret**:

| Secret name | Value |
|---|---|
| `ALPACA_API_KEY` | your Alpaca key |
| `ALPACA_SECRET_KEY` | your Alpaca secret |
| `TELEGRAM_BOT_TOKEN` | from @BotFather |
| `TELEGRAM_CHAT_ID` | from @userinfobot |
| `TWILIO_ACCOUNT_SID` | from Twilio console (WhatsApp only) |
| `TWILIO_AUTH_TOKEN` | from Twilio console (WhatsApp only) |
| `TWILIO_WHATSAPP_FROM` | e.g. `whatsapp:+14155238886` (WhatsApp only) |
| `TWILIO_WHATSAPP_TO` | e.g. `whatsapp:+1XXXXXXXXXX` (WhatsApp only) |

### Enable & test
**Actions** tab → enable workflows if prompted → **Strat Scanner** →
**Run workflow** → watch it go green → check Telegram/WhatsApp for a test
alert (if any symbol currently has a live setup) → confirm `latest_scan.json`
now exists in the repo's file list.

From here it runs itself every 15 minutes.

---

## Part 2: Deploy the dashboard on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** → pick your `strat-scanner` repo and branch.
3. Set **Main file path** to `dashboard/app.py` (important — this is what
   tells Streamlit Cloud to use `dashboard/requirements.txt` instead of the
   root one).
4. Click **Deploy**.
5. Once it's up, go to the app's **Settings → Secrets** and paste in:
   ```toml
   APP_PASSWORD = "choose-a-password"
   GITHUB_OWNER = "your-github-username"
   GITHUB_REPO = "strat-scanner"
   GITHUB_BRANCH = "main"
   ```
6. Save — the app restarts automatically. Visit the app URL, enter your
   password, and you should see the live scan table (once the Actions
   workflow has run at least once).

The dashboard re-fetches `latest_scan.json` from GitHub every 60 seconds
(cached) or immediately if you click **Refresh now**.

---

## Cost / limits reality check

- **GitHub Actions:** unlimited and free on a public repo. On a private
  repo you get 2,000 free minutes/month — a run here takes well under a
  minute, so 15-minute scheduling stays comfortably inside that.
- **Streamlit Community Cloud:** free tier, sleeps after a period of no
  visits and wakes back up when you open it — fine for a personal viewer.
- GitHub may pause scheduled workflows after **60 days with no commits** to
  the repo — push anything or click "Run workflow" manually to wake it
  back up if that happens.
- Actions cron timing isn't exact — expect occasional multi-minute delays
  during platform load. Not an issue for casual monitoring.

---

## If you ever want to scale up to a VPS instead

The engine already supports both modes:
- `python main.py --once` → single scan, used by the GitHub Actions cron.
- `python main.py` (no flag) → continuous asyncio service, for a VPS with systemd.

A `strat-scanner.service` systemd unit template is included for that path.
Ask if you want the full VPS walkthrough later.

## Notes / known limits

- **Data feed matters for matching other charting tools.** Alpaca defaults to
  IEX-only data (one exchange) unless told otherwise, even on paid plans.
  TradingView and most other charting platforms default to the full
  consolidated SIP tape. This can cause your Strat label sequence to
  genuinely differ from what you see elsewhere, because IEX alone can miss
  brief high/low prints that happened on other exchanges. If you have a
  subscription that includes SIP (e.g. Algo Trader Plus), set
  `ALPACA_DATA_FEED=sip` as a secret/env var to match. Leave it unset
  (defaults to `iex`) if you're not sure — requesting `sip` without
  entitlement causes 403 errors.
- Alpaca's free/IEX feed is end-of-day-ish for some symbols and doesn't
  cover futures — stocks & ETFs (SPY/QQQ as proxies for ES/NQ) work fine.
- This is detection only — no order/trade execution logic anywhere.
- 4H bars are synthetic (built from 1H bars via `resample_to_4h`), anchored
  to the 9:30 ET open, not a native Alpaca timeframe.
- The dashboard is read-only by design — it never talks to Alpaca directly,
  so there's no risk of it accidentally burning API calls or needing your
  Alpaca keys at all.
