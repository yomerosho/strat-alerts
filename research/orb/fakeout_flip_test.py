"""
Fakeout-then-opposite-break: does a failed ORB break on one side improve
the confirmed C2 entry on the OTHER side?

Motivated by QQQ 2026-08-26: fake break of ORH at 09:55, close back
inside at 10:00, then a hard push down. The fade itself tested negative
(fade_report.md); this asks the different question -- when the OPPOSITE
side later breaks with the full C2 confirmation, is that entry stronger
than an ordinary one?

Method: take the existing C2 trade log (confirm2_trades_C2.csv, V1
exits). For each trade, scan the day's 5-min bars BEFORE the trade's
break bar for a failed break of the opposite level: some bar closed
beyond it, then a later bar (still before our entry) closed back inside.
Split trade stats by that flag. Also condition on vol_expand and on the
full period IS/OOS.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
IS_END, OOS_START = "2024-12-31", "2025-01-01"
SYMBOLS = ["SPY", "QQQ", "IWM"]
T945 = pd.Timestamp("09:45").time()


def flag_opposite_fakeouts(sym, trades):
    df5 = pd.read_parquet(HERE / "data" / f"{sym}_5min.parquet")
    df5 = df5.between_time("09:30", "15:55")
    df5["tod"] = df5.index.time
    flags = []
    grouped = {d: g for d, g in df5.groupby(df5.index.normalize())}
    for _, tr in trades.iterrows():
        ts = pd.Timestamp(tr["signal_ts"])
        day = grouped.get(ts.normalize())
        if day is None:
            flags.append(False)
            continue
        sign = 1 if tr["direction"] == "long" else -1
        opp_level = tr["orl"] if sign == 1 else tr["orh"]
        pre = day[(day["tod"] >= T945) & (day.index < ts)]
        broke = sign * (opp_level - pre["close"]) > 0  # closed beyond opp level
        if not broke.any():
            flags.append(False)
            continue
        first = broke.idxmax()
        after = pre[pre.index > first]
        failed = (sign * (opp_level - after["close"]) < 0).any()  # closed back inside
        flags.append(bool(failed))
    return flags


def stats(g):
    n = len(g)
    if n == 0:
        return "| 0 | - | - | - | - |"
    pnl = g["s1_pnl"]
    w, l = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    pf = w / abs(l) if l != 0 else np.inf
    pf = "inf" if np.isinf(pf) else f"{pf:.2f}"
    return (f"| {n} | {(pnl > 0).mean()*100:.1f}% | {pf} | "
            f"{g['r_mult'].mean():+.3f} | ${pnl.sum():+,.0f} |")


def main():
    t = pd.read_csv(OUT / "confirm2_trades_C2.csv")
    t["vol_expand"] = t["entry_vol"] > t["prior6_vol"].fillna(np.inf)
    parts = []
    for sym in SYMBOLS:
        sub = t[t["symbol"] == sym].copy()
        sub["opp_fakeout"] = flag_opposite_fakeouts(sym, sub)
        parts.append(sub)
        print(f"{sym}: {sub['opp_fakeout'].sum()} of {len(sub)} entries follow an opposite-side fakeout")
    t = pd.concat(parts, ignore_index=True)

    lines = ["# Fakeout-then-opposite-break (C2 entries, V1 exits)\n"]
    W = lines.append
    dt = pd.to_datetime(t["date"])
    for span, m in (("IS 2022-2024", dt <= IS_END),
                    ("OOS 2025+", dt >= OOS_START),
                    ("FULL", pd.Series(True, index=t.index))):
        sub = t[m]
        W(f"\n## {span}\n")
        W("| config | trades | win% | PF | avg R | net P&L |")
        W("|---|---|---|---|---|---|")
        fo = sub["opp_fakeout"]
        vx = sub["vol_expand"]
        W(f"| after opp fakeout {stats(sub[fo])}")
        W(f"| no opp fakeout {stats(sub[~fo])}")
        W(f"| after opp fakeout +vx {stats(sub[fo & vx])}")
        W(f"| no opp fakeout +vx {stats(sub[~fo & vx])}")
        for dr in ("long", "short"):
            dm = sub["direction"] == dr
            W(f"| {dr} after opp fakeout {stats(sub[fo & dm])}")
            W(f"| {dr} no opp fakeout {stats(sub[~fo & dm])}")

    (OUT / "fakeout_flip_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report -> out/fakeout_flip_report.md")


if __name__ == "__main__":
    main()
