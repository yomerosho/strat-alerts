"""
C2 (2 closes beyond prior extreme) + vol_expand entry, gated by
DECISION-TIME trend-day proxies.

Prior findings: the confirmation entry only makes money on trend days
(ex-post: 71% win, PF 1.7), and volume can't identify fakeouts. This tests
whether anything knowable at/before entry can stand in for "trend day":

  gates   Breakout Gate Score (research/smc/four_gate.csv): g1 gap-size,
          g2 premarket vol, g3 opens above 10-day high, g4 OR-15m volume.
          Long-directional by construction -> tested on all trades AND
          longs-only.
  wideOR  today's OR range >= its own 14-day average (known 09:45,
          direction-neutral).
  biggap  |RTH open / prev close - 1| >= rolling 60-day 67th percentile
          for the symbol (known 09:30, direction-neutral).

Trades: out/confirm2_trades_C2.csv (V1 exits). vol_expand = entry bar
volume > mean of prior 6 bars. trend_day column is ex-post, shown only to
report each proxy's hit rate against it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
IS_START, IS_END, OOS_START = "2022-01-01", "2024-12-31", "2025-01-01"
SYMBOLS = ["SPY", "QQQ", "IWM"]


def day_context(sym):
    """Per-day decision-time proxies + ex-post trend flag."""
    d = pd.read_parquet(HERE / "data" / f"{sym}_daily.parquet")
    d.index = d.index.normalize()
    rng = d["high"] - d["low"]
    trend = rng > rng.rolling(14).mean().shift(1)

    gap = (d["open"] / d["close"].shift(1) - 1).abs() * 100
    gap_thr = gap.rolling(60, min_periods=20).quantile(0.667).shift(1)
    biggap = gap >= gap_thr

    df5 = pd.read_parquet(HERE / "data" / f"{sym}_5min.parquet")
    df5 = df5.between_time("09:30", "09:40")
    day_key = df5.index.normalize()
    orh = df5.groupby(day_key)["high"].max()
    orl = df5.groupby(day_key)["low"].min()
    orr = (orh - orl)
    wide = orr >= orr.rolling(14, min_periods=14).mean().shift(1)

    out = pd.DataFrame({"trend_day": trend, "biggap": biggap})
    out["wideOR"] = wide.reindex(out.index)
    out["symbol"] = sym
    out["date"] = [x.date() for x in out.index]
    return out.reset_index(drop=True)


def main():
    ctx = pd.concat([day_context(s) for s in SYMBOLS], ignore_index=True)

    gates = pd.read_csv("../smc/four_gate.csv") if (HERE / "../smc/four_gate.csv").exists() \
        else pd.read_csv(HERE.parent / "smc" / "four_gate.csv")
    gates = gates[gates["sym"].isin(SYMBOLS)].rename(columns={"sym": "symbol"})
    gates["date"] = pd.to_datetime(gates["day"]).dt.date
    gates = gates[["symbol", "date", "score", "g1", "g2", "g3", "g4"]]

    t = pd.read_csv(OUT / "confirm2_trades_C2.csv")
    t["date"] = pd.to_datetime(t["date"]).dt.date
    t = t.merge(ctx, on=["symbol", "date"], how="left")
    t = t.merge(gates, on=["symbol", "date"], how="left")
    t["vol_expand"] = t["entry_vol"] > t["prior6_vol"].fillna(np.inf)
    # gate data starts 2022-03-30; restrict everything to the overlap
    t = t[~t["score"].isna()].reset_index(drop=True)

    CONFIGS = {
        "C2 base":                lambda x: pd.Series(True, index=x.index),
        "C2+vx":                  lambda x: x["vol_expand"],
        "C2+vx wideOR":           lambda x: x["vol_expand"] & x["wideOR"].fillna(False),
        "C2+vx biggap":           lambda x: x["vol_expand"] & x["biggap"].fillna(False),
        "C2+vx score>=1":         lambda x: x["vol_expand"] & (x["score"] >= 1),
        "C2+vx score>=2":         lambda x: x["vol_expand"] & (x["score"] >= 2),
        "C2+vx score>=1 longs":   lambda x: x["vol_expand"] & ((x["direction"] == "short") | (x["score"] >= 1)),
        "C2+vx wideOR+biggap":    lambda x: x["vol_expand"] & x["wideOR"].fillna(False) & x["biggap"].fillna(False),
        "C2+vx wideOR|biggap":    lambda x: x["vol_expand"] & (x["wideOR"].fillna(False) | x["biggap"].fillna(False)),
        "wideOR only (no vx)":    lambda x: x["wideOR"].fillna(False),
    }

    def stats(g):
        n = len(g)
        if n == 0:
            return None
        pnl = g["s1_pnl"]
        w, l = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
        ndays = g["date"].nunique()
        return dict(n=n, win=(pnl > 0).mean() * 100,
                    pf=(w / abs(l)) if l != 0 else np.inf,
                    avg_r=g["r_mult"].mean(), pnl=pnl.sum(),
                    tpw=n / max(ndays, 1),
                    trend_pct=g["trend_day"].mean() * 100)

    lines = []
    W = lines.append
    W("# C2 + vol_expand gated by decision-time trend proxies\n")
    W(f"Trades restricted to gate-data overlap ({t['date'].min()} .. {t['date'].max()}). "
      "V1 exits, $10k, 1c/side. trend% = share of trades landing on ex-post trend days.\n")

    dt = pd.to_datetime(t["date"])
    spans = (("IS 2022-2024", (dt >= IS_START) & (dt <= IS_END)),
             ("OOS 2025+", dt >= OOS_START),
             ("FULL", pd.Series(True, index=t.index)))
    for span_name, m in spans:
        W(f"\n## {span_name}\n")
        W("| config | trades | win% | PF | avg R | net P&L | trend% |")
        W("|---|---|---|---|---|---|---|")
        sub = t[m]
        for cname, cmask in CONFIGS.items():
            s = stats(sub[cmask(sub)])
            if s is None:
                W(f"| {cname} | 0 | - | - | - | - | - |")
                continue
            pf = "inf" if np.isinf(s["pf"]) else f"{s['pf']:.2f}"
            W(f"| {cname} | {s['n']} | {s['win']:.1f}% | {pf} | {s['avg_r']:+.3f} | "
              f"${s['pnl']:+,.0f} | {s['trend_pct']:.0f}% |")

    # how good is each proxy at actually finding trend days?
    W("\n## Proxy vs ex-post trend day (trade-level, full period)\n")
    W("| proxy | trades flagged | trend-day hit rate | base rate |")
    W("|---|---|---|---|")
    base = t["trend_day"].mean() * 100
    for pname, pmask in (("wideOR", t["wideOR"].fillna(False)),
                         ("biggap", t["biggap"].fillna(False)),
                         ("score>=1", t["score"] >= 1),
                         ("score>=2", t["score"] >= 2)):
        g = t[pmask]
        W(f"| {pname} | {len(g)} | {g['trend_day'].mean()*100:.1f}% | {base:.1f}% |")

    # yearly consistency of the best direction-neutral config
    W("\n## C2+vx wideOR: by year\n")
    W("| year | trades | win% | avg R | net P&L |")
    W("|---|---|---|---|---|")
    best = t[CONFIGS["C2+vx wideOR"](t)].copy()
    best["year"] = pd.to_datetime(best["date"]).dt.year
    for y, g in best.groupby("year"):
        W(f"| {y} | {len(g)} | {(g['s1_pnl'] > 0).mean()*100:.1f}% | "
          f"{g['r_mult'].mean():+.3f} | ${g['s1_pnl'].sum():+,.0f} |")

    (OUT / "c2_gated_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written -> out/c2_gated_report.md")


if __name__ == "__main__":
    main()
