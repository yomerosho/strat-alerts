"""
Stage 1: is trend/range day type predictable before 09:45 ET?

Label (end of day, RTH):
  trend = day_range > trailing-14-session MEDIAN of day_range (prior sessions)
          AND |close - open| / day_range >= 0.5

Predictors X1..X10 as specified; all computable by 09:45 same day.
IS = 2022-01-01..2024-12-31.  OOS untouched here except label base rates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
SYMBOLS = ["SPY", "QQQ", "IWM"]
IS_START, IS_END = "2022-01-01", "2024-12-31"
OOS_START = "2025-01-01"

# Actual release dates (BLS year schedules incl. 2025 shutdown shifts;
# FOMC = decision day, federalreserve.gov calendar).
FOMC = """2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17""".split()
CPI = """2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14""".split()
NFP = """2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03 2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02""".split()
MACRO_DATES = {pd.Timestamp(d).date() for d in FOMC + CPI + NFP}


def wilder_atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_ticker(sym, vix):
    df5 = pd.read_parquet(DATA / f"{sym}_5min.parquet").between_time("09:30", "15:55")
    daily = pd.read_parquet(DATA / f"{sym}_daily.parquet")
    daily.index = daily.index.normalize().tz_localize(None)

    t930, t940, t945 = (pd.Timestamp(x).time() for x in ("09:30", "09:40", "09:45"))
    recs = []
    for date, day in df5.groupby(df5.index.normalize()):
        times = day.index.time
        if t930 not in times or t945 not in times:
            continue
        orb = day[day.index.time <= t940]
        recs.append(dict(
            date=date.tz_localize(None),
            open_rth=day.iloc[0]["open"], close_rth=day.iloc[-1]["close"],
            high_rth=day["high"].max(), low_rth=day["low"].min(),
            or_range=orb["high"].max() - orb["low"].min(),
            first15_vol=orb["volume"].sum(),
        ))
    d = pd.DataFrame(recs).set_index("date").sort_index()

    # ---- label ----
    d["day_range"] = d["high_rth"] - d["low_rth"]
    med14 = d["day_range"].rolling(14).median().shift(1)
    directional = (d["close_rth"] - d["open_rth"]).abs() / d["day_range"] >= 0.5
    d["trend"] = ((d["day_range"] > med14) & directional).astype(float)
    d.loc[med14.isna(), "trend"] = np.nan

    # ---- daily context (aligned to session dates, shifted to prior close) ----
    dd = daily.reindex(d.index)
    atr = wilder_atr(daily["high"], daily["low"], daily["close"]).reindex(d.index)
    d["atr_d"] = atr.shift(1)
    d["prev_close_d"] = dd["close"].shift(1)
    prev_candle_dir = np.sign(dd["close"] - dd["open"]).shift(1)

    # ---- predictors ----
    gap = d["open_rth"] - d["prev_close_d"]
    d["X1"] = gap.abs() / d["atr_d"]
    gap_dir = np.sign(gap).where(gap.abs() > 0.1 * d["atr_d"], 0.0)
    d["X2"] = gap_dir * prev_candle_dir
    d["X3"] = d["or_range"] / d["atr_d"]
    d["X4"] = d["first15_vol"] / d["first15_vol"].rolling(20).mean().shift(1)
    vix_c = vix["close"].reindex(d.index, method="ffill")
    v1, v2 = vix_c.shift(1), vix_c.shift(2)
    d["vix_prev"] = v1
    d["X5"] = pd.cut(v1, [0, 15, 20, 28, np.inf],
                     labels=["<15", "15-20", "20-28", ">28"])
    chg = v1 / v2 - 1
    d["X6"] = np.select([chg > 0.02, chg < -0.02], ["up", "down"], "flat")
    d["X7"] = d["trend"].shift(1)
    d["X8"] = d["day_range"].shift(1) / d["atr_d"]
    d["X9"] = pd.Series([dt.date() in MACRO_DATES for dt in d.index], index=d.index).astype(float)
    d["X10"] = pd.Series(d.index.dayofweek, index=d.index).map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"})
    d["symbol"] = sym
    return d


def main():
    vix = pd.read_parquet(DATA / "VIX.parquet").set_index("date").sort_index()
    frames = [build_ticker(s, vix) for s in SYMBOLS]
    df = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    need = ["trend", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]
    df = df.dropna(subset=need)
    df = df[df["date"] >= IS_START].reset_index(drop=True)
    df.to_csv(OUT / "stage1_dataset.csv", index=False)

    ins = df[df["date"] <= IS_END].copy()
    print(f"dataset: {len(df)} ticker-days total, {len(ins)} in-sample")

    L = []
    W = L.append
    W("# Stage 1: day-type classification\n")

    # ---- base rates ----
    W("## Trend-day base rate by ticker/year (label distribution; "
      "2025-26 rows are OOS labels, no predictor evaluated on them here)\n")
    br = df.copy()
    br["year"] = br["date"].dt.year
    piv = br.pivot_table(values="trend", index="year", columns="symbol", aggfunc="mean")
    cnt = br.pivot_table(values="trend", index="year", columns="symbol", aggfunc="count")
    W("| year | " + " | ".join(f"{s} (n)" for s in SYMBOLS) + " |")
    W("|---|---|---|---|")
    for y in piv.index:
        W(f"| {y} | " + " | ".join(
            f"{piv.loc[y, s]*100:.1f}% ({cnt.loc[y, s]:.0f})" for s in SYMBOLS) + " |")
    base_is = ins["trend"].mean()
    W(f"\nIn-sample pooled base rate: **{base_is*100:.1f}%** ({len(ins)} ticker-days)\n")

    # ---- univariate ----
    W("## Univariate: in-sample trend rate by bucket (pooled across tickers)\n")
    cont = {"X1": "gap/ATR", "X3": "OR/ATR", "X4": "f15 vol ratio", "X8": "prior range/ATR"}
    cat = {"X2": "gap align", "X5": "VIX bucket", "X6": "VIX chg", "X7": "prior day type",
           "X9": "macro day", "X10": "weekday"}
    for x, nm in cont.items():
        q = pd.qcut(ins[x], 5, duplicates="drop")
        g = ins.groupby(q, observed=True)["trend"].agg(["mean", "count"])
        W(f"\n**{x} ({nm})** quintiles:\n")
        W("| bucket | trend rate | n |")
        W("|---|---|---|")
        for b, r in g.iterrows():
            W(f"| {b} | {r['mean']*100:.1f}% | {r['count']:.0f} |")
    for x, nm in cat.items():
        g = ins.groupby(ins[x].astype(str), observed=True)["trend"].agg(["mean", "count"])
        W(f"\n**{x} ({nm})**:\n")
        W("| value | trend rate | n |")
        W("|---|---|---|")
        for b, r in g.iterrows():
            W(f"| {b} | {r['mean']*100:.1f}% | {r['count']:.0f} |")

    # ---- univariate p-values (LR test vs null) + Bonferroni ----
    import statsmodels.api as sm
    from scipy import stats as sps

    def design(dfx, xs):
        parts = []
        for x in xs:
            if x in cont or x in ("X7", "X9"):
                parts.append(dfx[[x]].astype(float))
            elif x == "X2":
                parts.append(pd.get_dummies(dfx[x].map({-1: "opp", 0: "none", 1: "align"}),
                                            prefix="X2", drop_first=True).astype(float))
            else:
                parts.append(pd.get_dummies(dfx[x].astype(str), prefix=x,
                                            drop_first=True).astype(float))
        X = pd.concat(parts, axis=1)
        X.insert(0, "const", 1.0)
        return X

    y = ins["trend"].values
    ll0 = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf
    W("\n## Univariate significance (logistic LR test) and multiple-testing view\n")
    W("| predictor | LR chi2 | df | p | naive p<0.05 | Bonferroni p<0.005 |")
    W("|---|---|---|---|---|---|")
    pvals = {}
    for x in list(cont) + list(cat):
        X = design(ins, [x])
        m = sm.Logit(y, X).fit(disp=0)
        lr = 2 * (m.llf - ll0)
        dfree = X.shape[1] - 1
        p = sps.chi2.sf(lr, dfree)
        pvals[x] = p
        W(f"| {x} | {lr:.1f} | {dfree} | {p:.2g} | {'YES' if p < 0.05 else 'no'} | "
          f"{'YES' if p < 0.005 else 'no'} |")
    W("\nTesting 10 predictors at alpha=0.05, ~0.5 would clear by chance alone; "
      "Bonferroni-adjusted threshold is 0.005.\n")

    # ---- full logistic + blocked 5-fold CV AUC ----
    from sklearn.metrics import roc_auc_score

    Xfull = design(ins, list(cont) + list(cat))
    mfull = sm.Logit(y, Xfull).fit(disp=0)
    W("## Full logistic regression (in-sample fit)\n")
    W("| term | coef | z | p |")
    W("|---|---|---|---|")
    for t, c, z, p in zip(mfull.params.index, mfull.params, mfull.tvalues, mfull.pvalues):
        W(f"| {t} | {c:+.3f} | {z:+.2f} | {p:.3f} |")

    # contiguous time-block folds (split by unique date, all tickers together)
    dates = np.sort(ins["date"].unique())
    folds = np.array_split(dates, 5)
    W("\n## Blocked 5-fold CV (contiguous date blocks)\n")

    def cv_auc(xs):
        aucs = []
        for f in folds:
            te = ins["date"].isin(f)
            Xtr, Xte = design(ins[~te], xs), design(ins[te], xs)
            Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
            m = sm.Logit(ins.loc[~te, "trend"].values, Xtr).fit(disp=0, maxiter=200)
            aucs.append(roc_auc_score(ins.loc[te, "trend"], m.predict(Xte)))
        return np.mean(aucs), aucs

    auc_full, aucs = cv_auc(list(cont) + list(cat))
    W(f"Full model CV AUC: **{auc_full:.3f}** (folds: "
      + ", ".join(f"{a:.3f}" for a in aucs) + ")\n")
    W("| predictor (alone) | CV AUC |")
    W("|---|---|")
    rank = {}
    for x in list(cont) + list(cat):
        rank[x], _ = cv_auc([x])
        W(f"| {x} | {rank[x]:.3f} |")

    # ---- simple gate search: best 1-2 predictors, max 2 thresholds ----
    # Coverage floor: gate must fire on >=10% of train ticker-days (a
    # precision number computed on a handful of days is noise).
    W("\n## Simple gate search (complexity budget: <=2 predictors, <=2 thresholds, "
      "coverage >= 10% of days)\n")

    def gate_candidates(dfx):
        """Returns dict name -> boolean series, threshold rules only."""
        out = {}
        for x in cont:
            for q in (0.5, 0.6, 0.7, 0.8, 0.9):
                thr = dfx[x].quantile(q)
                out[f"{x}>q{int(q*100)}"] = (dfx[x] > thr, f"{x} > {thr:.3f}")
                out[f"{x}<q{int(100-q*100)}"] = (dfx[x] < dfx[x].quantile(1 - q),
                                                 f"{x} < {dfx[x].quantile(1-q):.3f}")
        out["X9=1"] = (dfx["X9"] == 1, "X9 = 1 (macro day)")
        out["X7=0"] = (dfx["X7"] == 0, "X7 = 0 (prior day range)")
        out["X7=1"] = (dfx["X7"] == 1, "X7 = 1 (prior day trend)")
        return out

    # honest CV: rule family is fixed above; for each fold, pick the best rule
    # on train (precision, coverage>=10%), apply to test; report test precision
    def cv_gate():
        per_fold = []
        for f in folds:
            te = ins["date"].isin(f)
            tr_df, te_df = ins[~te], ins[te]
            best_rule, best_prec = None, -1
            cands = gate_candidates(tr_df)
            singles = list(cands.items())
            for nm, (mask, desc) in singles:
                if mask.mean() < 0.10:
                    continue
                prec = tr_df.loc[mask, "trend"].mean()
                if prec > best_prec:
                    best_prec, best_rule = prec, ("single", nm)
            for i in range(len(singles)):        # OR / AND combos of two
                for j in range(i + 1, len(singles)):
                    n1, (m1, _) = singles[i]
                    n2, (m2, _) = singles[j]
                    for op, mm in (("OR", m1 | m2), ("AND", m1 & m2)):
                        if mm.mean() < 0.10:
                            continue
                        prec = tr_df.loc[mm, "trend"].mean()
                        if prec > best_prec:
                            best_prec, best_rule = prec, (op, n1, n2)
            # apply chosen rule to test fold
            cte = gate_candidates(tr_df)  # thresholds from TRAIN quantiles
            def apply(rule, dfx):
                c = {k: v[0] for k, v in gate_candidates(tr_df).items()}
                # rebuild masks on dfx using train thresholds
                def m_of(nm):
                    if nm.startswith("X9"): return dfx["X9"] == 1
                    if nm == "X7=0": return dfx["X7"] == 0
                    if nm == "X7=1": return dfx["X7"] == 1
                    x = nm.split(">")[0].split("<")[0]
                    if ">" in nm:
                        q = int(nm.split("q")[1]) / 100
                        return dfx[x] > tr_df[x].quantile(q)
                    q = int(nm.split("q")[1]) / 100
                    return dfx[x] < tr_df[x].quantile(q)
                if rule[0] == "single": return m_of(rule[1])
                if rule[0] == "OR": return m_of(rule[1]) | m_of(rule[2])
                return m_of(rule[1]) & m_of(rule[2])
            mte = apply(best_rule, te_df)
            prec_te = te_df.loc[mte, "trend"].mean() if mte.sum() > 0 else np.nan
            per_fold.append((best_rule, best_prec, prec_te, mte.mean()))
        return per_fold

    per_fold = cv_gate()
    W("| fold | rule chosen on train | train prec | TEST prec | test coverage |")
    W("|---|---|---|---|---|")
    test_precs = []
    for i, (rule, ptr, pte, cov) in enumerate(per_fold):
        rs = rule[1] if rule[0] == "single" else f"{rule[1]} {rule[0]} {rule[2]}"
        W(f"| {i+1} | {rs} | {ptr*100:.1f}% | {pte*100:.1f}% | {cov*100:.1f}% |")
        test_precs.append(pte)
    cv_hit = np.nanmean(test_precs)
    W(f"\nCross-validated gate hit rate (mean test-fold precision): "
      f"**{cv_hit*100:.1f}%** vs base rate {base_is*100:.1f}%")
    W(f"\n**KILL CRITERION (>=55% CV hit rate): "
      f"{'PASS' if cv_hit >= 0.55 else 'FAIL -- STOP, no Stage 2'}**\n")

    # the single pre-registered gate fit on ALL in-sample data (only used if pass)
    cands = gate_candidates(ins)
    best = None
    singles = list(cands.items())
    for nm, (mask, desc) in singles:
        if mask.mean() < 0.10:
            continue
        prec = ins.loc[mask, "trend"].mean()
        if best is None or prec > best[1]:
            best = ((("single", nm), prec, desc))
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            n1, (m1, d1) = singles[i]
            n2, (m2, d2) = singles[j]
            for op, mm in (("OR", m1 | m2), ("AND", m1 & m2)):
                if mm.mean() < 0.10:
                    continue
                prec = ins.loc[mm, "trend"].mean()
                if prec > best[1]:
                    best = ((op, n1, n2), prec, f"({d1}) {op} ({d2})")
    W(f"Pre-registered full-IS gate (for Stage 2 if passing): **{best[2]}** "
      f"(IS precision {best[1]*100:.1f}%)\n")

    (OUT / "stage1_report.md").write_text("\n".join(L), encoding="utf-8")
    print("stage1 report written")
    print(f"base rate {base_is*100:.1f}%  full-model CV AUC {auc_full:.3f}  "
          f"CV gate hit {cv_hit*100:.1f}%  kill={'PASS' if cv_hit >= 0.55 else 'FAIL'}")


if __name__ == "__main__":
    main()
