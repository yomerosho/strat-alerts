"""Build REPORT.md from trades.parquet per the user's reporting spec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
ET = "America/New_York"

t = pd.read_parquet(OUT / "trades.parquet")
t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True).dt.tz_convert(ET)
filled = t[t.status == "filled"].copy()
filled["win"] = filled.r > 0

P15 = ["P1", "P2", "P3", "P4", "P5"]
F2 = ["P6-F2D", "P7-F2U"]


def metrics(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series(dict(n=0, win=np.nan, avg_r=np.nan, pf=np.nan, maxdd=np.nan))
    wins = df.r[df.r > 0].sum()
    losses = -df.r[df.r < 0].sum()
    eq = df.sort_values("entry_time").r.cumsum()
    dd = (eq - eq.cummax()).min()
    return pd.Series(dict(
        n=n, win=df.win.mean() * 100, avg_r=df.r.mean(),
        pf=(wins / losses) if losses > 0 else np.inf, maxdd=dd,
    ))


def table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by).apply(metrics, include_groups=False).round(2)
    g["n"] = g["n"].astype(int)
    return g


def md(df: pd.DataFrame) -> str:
    return df.reset_index().to_markdown(index=False)


lines = []
A = lines.append

A("# Strat Pattern Backtest -- SPY / QQQ / IWM")
A("")
A("Data: Alpaca SIP, RTH only. 4H = session-anchored 09:30-13:30 + 13:30-16:00.")
A("In-sample (IS): 2022-01-01 .. 2024-12-31. Out-of-sample (OOS): 2025-01-01 .. 2026-07-24.")
A("Entries/exits sequenced on 5m tape; any 5m bar touching both stop and target = STOP.")
A("Win = R > 0. Stops fill at stop price (R = -1). No commissions/slippage beyond fill rules.")
A("")

# ---------------- 1. per pattern / tf / direction (E1, unfiltered) ----------
A("## 1. Core results -- E1, no filter, per pattern / timeframe / direction")
for smp in ("IS", "OOS"):
    A(f"\n### {smp}")
    sub = filled[(filled.variant == "E1") & (filled["sample"] == smp) & filled.pattern.isin(P15)]
    A(md(table(sub, ["pattern", "tf", "side"])))

# ---------------- 2. E1 vs E2 ------------------------------------------------
A("\n## 2. Entry variant comparison (patterns aggregated across direction+symbol, IS)")
rows = []
for smp in ("IS", "OOS"):
    for (pat, tf), g in t[t.pattern.isin(P15) & (t["sample"] == smp)].groupby(["pattern", "tf"]):
        for var in ("E1", "E2-5m", "E2-15m"):
            gv = g[g.variant == var]
            f = gv[gv.status == "filled"]
            m = metrics(f.assign(win=f.r > 0))
            rows.append(dict(
                sample=smp, pattern=pat, tf=tf, variant=var, n=int(m.n),
                win=round(m.win, 1) if m.n else np.nan,
                avg_r=round(m.avg_r, 2) if m.n else np.nan,
                pf=round(m.pf, 2) if m.n else np.nan,
                stop_rate=round((f.outcome == "stop").mean() * 100, 1) if len(f) else np.nan,
                slip_r=round(f.slippage_r.mean(), 2) if len(f) else np.nan,
                missed=int((gv.status == "missed").sum() + (gv.status == "missed_winner").sum()),
                missed_winners=int((gv.status == "missed_winner").sum()),
            ))
E2 = pd.DataFrame(rows)
A("\n### IS")
A(E2[E2["sample"] == "IS"].drop(columns="sample").to_markdown(index=False))
A("\n### OOS")
A(E2[E2["sample"] == "OOS"].drop(columns="sample").to_markdown(index=False))

# ---------------- 3. loser MFE ----------------------------------------------
A("\n## 3. Target-hit distance on losers (median % of magnitude reached, E1)")
losers = filled[(filled.variant == "E1") & (filled.r < 0) & filled.pattern.isin(P15)]
g = losers.groupby(["pattern", "tf"]).agg(
    n_losers=("mfe_pct", "size"),
    median_mfe_pct=("mfe_pct", lambda x: round(np.median(x) * 100, 1)),
    pct_reaching_half=("mfe_pct", lambda x: round((x >= 0.5).mean() * 100, 1)),
).reset_index()
A(g.to_markdown(index=False))

# ---------------- 4. FTFC ----------------------------------------------------
A("\n## 4. FTFC filter effect (E1, IS) -- with vs without")
rows = []
for (pat, tf), g in filled[(filled.variant == "E1") & (filled["sample"] == "IS")
                           & filled.pattern.isin(P15)].groupby(["pattern", "tf"]):
    for label, gg in (("all", g), ("FTFC", g[g.ftfc == True])):  # noqa: E712
        m = metrics(gg)
        rows.append(dict(pattern=pat, tf=tf, subset=label, n=int(m.n),
                         win=round(m.win, 1), avg_r=round(m.avg_r, 2),
                         pf=round(m.pf, 2)))
A(pd.DataFrame(rows).to_markdown(index=False))
A("\nSame, OOS:")
rows = []
for (pat, tf), g in filled[(filled.variant == "E1") & (filled["sample"] == "OOS")
                           & filled.pattern.isin(P15)].groupby(["pattern", "tf"]):
    for label, gg in (("all", g), ("FTFC", g[g.ftfc == True])):  # noqa: E712
        m = metrics(gg)
        rows.append(dict(pattern=pat, tf=tf, subset=label, n=int(m.n),
                         win=round(m.win, 1), avg_r=round(m.avg_r, 2),
                         pf=round(m.pf, 2)))
A(pd.DataFrame(rows).to_markdown(index=False))

# ---------------- 5. ranking -------------------------------------------------
A("\n## 5. Expectancy ranking (E1, no filter): IS vs OOS")
r_is = table(filled[(filled.variant == "E1") & (filled["sample"] == "IS")
                    & filled.pattern.isin(P15)], ["pattern", "tf", "side"])
r_oos = table(filled[(filled.variant == "E1") & (filled["sample"] == "OOS")
                     & filled.pattern.isin(P15)], ["pattern", "tf", "side"])
rank = r_is.join(r_oos, lsuffix="_IS", rsuffix="_OOS").sort_values("avg_r_IS", ascending=False)
A(md(rank.round(2)))

# ---------------- 6. P6/P7 --------------------------------------------------
A("\n## 6. Failed 2s (P6 F2D long / P7 F2U short, Daily)")
f2 = t[t.pattern.isin(F2)]
f2f = filled[filled.pattern.isin(F2)]
for smp in ("IS", "OOS"):
    A(f"\n### {smp}: T1 vs T2 vs T3")
    A(md(table(f2f[f2f["sample"] == smp], ["pattern", "variant"])))

A("\n### T1 false-trigger rate (entry, then new same-day extreme beyond the failure level)")
ft = f2f[f2f.variant == "T1"].groupby(["pattern", "sample"]).agg(
    n=("false_trigger", "size"),
    false_trigger_pct=("false_trigger", lambda x: round(x.mean() * 100, 1)),
).reset_index()
A(ft.to_markdown(index=False))

A("\n### Undercut depth (ATR14 units) vs outcome -- T1, IS+OOS combined")
d = f2f[f2f.variant == "T1"].copy()
d["depth"] = pd.cut(d.undercut_atr, [0, 0.1, 0.25, 0.5, 10],
                    labels=["0-0.10", "0.10-0.25", "0.25-0.50", "0.50+"])
g = d.groupby(["pattern", "depth"], observed=True).agg(
    n=("r", "size"), win=("win", lambda x: round(x.mean() * 100, 1)),
    avg_r=("r", lambda x: round(x.mean(), 2))).reset_index()
A(g.to_markdown(index=False))
A("\nSame, T3:")
d3 = f2f[f2f.variant == "T3"].copy()
d3["depth"] = pd.cut(d3.undercut_atr, [0, 0.1, 0.25, 0.5, 10],
                     labels=["0-0.10", "0.10-0.25", "0.25-0.50", "0.50+"])
g3 = d3.groupby(["pattern", "depth"], observed=True).agg(
    n=("r", "size"), win=("win", lambda x: round(x.mean() * 100, 1)),
    avg_r=("r", lambda x: round(x.mean(), 2))).reset_index()
A(g3.to_markdown(index=False))

# ---------------- 7. trades/day ---------------------------------------------
A("\n## 7. Trades/day if alerting only positive-expectancy combos")
daily_counts = {}
enabled = rank[(rank.avg_r_IS > 0) & (rank.n_IS >= 30)].index
A(f"\nEnabled combos (IS avg R > 0, n >= 30): {[f'{p}/{tf}/{s}' for p, tf, s in enabled]}")
en = filled[(filled.variant == "E1") & filled.pattern.isin(P15)]
en = en[en.set_index(["pattern", "tf", "side"]).index.isin(enabled)]
cal = pd.read_parquet(Path(__file__).parents[1] / "orb" / "data" / "SPY_daily.parquet")
cal.index = cal.index.tz_convert(ET).normalize()
spans = {
    "IS": ((cal.index >= "2022-01-01") & (cal.index < "2025-01-01")).sum(),
    "OOS": (cal.index >= "2025-01-01").sum(),
}
for smp in ("IS", "OOS"):
    sub = en[en["sample"] == smp]
    days = sub.entry_time.dt.normalize().nunique()
    span = spans[smp]
    A(f"- {smp}: {len(sub)} trades / {span} trading days = "
      f"**{len(sub)/span:.2f} trades/day** (fired on {days} distinct days)")

text = "\n".join(lines)
(OUT / "REPORT.md").write_text(text, encoding="utf-8")
print(text[:3000])
print("\n... written to REPORT.md, length", len(text))
