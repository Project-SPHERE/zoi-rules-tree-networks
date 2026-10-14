import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import zoi_core as z

NAME = re.compile(r"synth_([a-z_]+?)_s(\d+(?:\.\d+)?)_w(\d+(?:\.\d+)?)_f(\d+(?:\.\d+)?)")
SHORT = {"stability": "stab", "stability_terms": "terms", "multi_agg": "agg", "multi_union": "union"}
TERMS = {t: z.Form.make([g for g in s if len(g) == 3], [g for g in s if len(g) == 2]).terms() for t, s in z.TRUTHS.items()}
BASELINES = ("logistic", "distance")
ORDER = ["stab200", "stab400", "stab800", "terms400", "agg400", "union400", "single2000", "single4000", "logistic",
         "distance"]
TESTS = [
    ("stab200", "single2000", "noisy", ("species_dbh", "sqrt_z")),
    ("stab200", "single2000", "noisy", ("pair",)),
    ("stab400", "single4000", "noisy", None),
    ("agg400", "single4000", "noisy", None),
    ("agg400", "stab400", "noisy", None),
    ("stab400", "terms400", "noisy", None),
    ("agg400", "stab400", "clean", ("species_dbh",)),
    ("agg400", "stab800", "clean", ("species_dbh",)),
]


def variant(method, M, runs):
    if method in SHORT:
        return f"{SHORT[method]}{M}"
    if method == "single":
        return f"single{2 * runs * M}"
    return method


def load(sources, runs):
    frames = []
    for label, M, patterns in sources:
        files = sorted({f for p in patterns for f in glob.glob(p)})
        files = [f for f in files if not f.endswith("_summary.csv") and NAME.search(os.path.basename(f))]
        print(f"{label}: {len(files)} files")
        for f in files:
            m = NAME.search(os.path.basename(f))
            d = pd.read_csv(f)
            d["truth"], d["sigma"], d["width"] = m.group(1), float(m.group(2)), float(m.group(3))
            d["variant"] = [variant(x, M, runs) for x in d["method"]]
            frames.append(d)
    if not frames:
        raise SystemExit("no synthetic results found")
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["width", "truth", "sigma", "rep", "variant"])
    has = ~d["variant"].isin(BASELINES)
    d["ok"] = np.where(has, (d["exact_form"].astype(str) == "True").astype(float), np.nan)
    found = [z.parse_form(s).terms() if isinstance(s, str) and s.strip() else set() for s in d["form"]]
    truth = [TERMS[t] for t in d["truth"]]
    d["false_terms"] = np.where(has, [len(f - t) for f, t in zip(found, truth)], np.nan)
    d["missed_terms"] = np.where(has, [len(t - f) for f, t in zip(found, truth)], np.nan)
    return d.reset_index(drop=True)


def share_true(v):
    return float((v.astype(str) == "True").mean()) if v.notna().any() else np.nan


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def tests(d, noisy):
    ok = d[d["ok"].notna()].pivot_table(index=["width", "truth", "sigma", "rep"], columns="variant", values="ok",
                                        aggfunc="max")
    rows = []
    for A, B, kind, truths in TESTS:
        part = ok[ok.index.get_level_values("width") == (noisy if kind == "noisy" else 0.0)]
        if truths:
            part = part[part.index.get_level_values("truth").isin(truths)]
        row = {"A": A, "B": B, "setting": kind, "truths": "+".join(truths) if truths else "all"}
        if A not in part.columns or B not in part.columns or part[[A, B]].dropna().empty:
            rows.append({**row, "note": "no data"})
            continue
        x = part[[A, B]].dropna()
        n1, n2 = int(((x[A] == 1) & (x[B] == 0)).sum()), int(((x[A] == 0) & (x[B] == 1)).sum())
        rows.append({**row, "n": len(x), "recovered_A": int(x[A].sum()), "recovered_B": int(x[B].sum()),
                     "only_A": n1, "only_B": n2, "p_exact": binomtest(n1, n1 + n2).pvalue if n1 + n2 else 1.0,
                     "note": ""})
    df = pd.DataFrame(rows)
    valid = df["p_exact"].notna() if "p_exact" in df else pd.Series(False, index=df.index)
    df["p_holm"] = np.nan
    if valid.any():
        df.loc[valid, "p_holm"] = holm(df.loc[valid, "p_exact"])
    return df[[c for c in df.columns if c != "note"] + ["note"]]


def wide(table, value, keys):
    cols = [v for v in ORDER if v in set(table["variant"])]
    return table.pivot_table(index=keys, columns="variant", values=value, aggfunc="first").reindex(columns=cols)


def main():
    ap = argparse.ArgumentParser(prog="zoi_synth_stats")
    ap.add_argument("--m200", nargs="*", default=[os.path.join("results", "synth_m200", "synth_*.csv")])
    ap.add_argument("--m400", nargs="*", default=[os.path.join("results", "synth_m400", "synth_*.csv")])
    ap.add_argument("--single4000", nargs="*", default=[os.path.join("results", "synth_single4000", "synth_*.csv")])
    ap.add_argument("--m800", nargs="*", default=[os.path.join("results", "synth_m800", "synth_*.csv")])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--noisy", type=float, default=0.2)
    ap.add_argument("--out", default=os.path.join("results", "synth_stats"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load([("m200", 200, a.m200), ("m400", 400, a.m400), ("single4000", 400, a.single4000),
              ("m800", 800, a.m800)], a.runs)
    keys = ["width", "truth", "sigma"]
    f1 = d.groupby(keys + ["variant"])["F1_vs_truth"].agg(n="count", mean="mean", sd="std", min="min",
                                                          max="max").round(4).reset_index()
    f1["text"] = [f"{m:.4f} [{lo:.4f}, {hi:.4f}]" if n else "" for n, m, lo, hi in
                  zip(f1["n"], f1["mean"], f1["min"], f1["max"])]
    s = d[d["ok"].notna()]
    rec = s.groupby(keys + ["variant"]).agg(n=("ok", "size"), recovered=("ok", "sum"), rate=("ok", "mean"),
                                            replaced=("safeguard", share_true)).round(3).reset_index()
    terms = s.groupby(["width", "variant"]).agg(n=("ok", "size"), false_mean=("false_terms", "mean"),
                                                any_false=("false_terms", lambda v: float((v > 0).mean())),
                                                missed_mean=("missed_terms", "mean")).round(3).reset_index()
    tst = tests(d, a.noisy)
    f1.to_csv(os.path.join(a.out, "synth_f1.csv"), index=False)
    rec.to_csv(os.path.join(a.out, "synth_recovery.csv"), index=False)
    terms.to_csv(os.path.join(a.out, "synth_terms.csv"), index=False)
    tst.to_csv(os.path.join(a.out, "synth_tests.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print("\nF1 against the true edges, mean [min, max] over replicates")
    print(wide(f1, "text", keys).to_string())
    print("\nStructural recovery, recovered replicates")
    print(wide(rec, "recovered", keys).astype("Int64").to_string())
    print("\nShare of replicates that used the replacement rule")
    print(wide(rec[rec["replaced"].notna()], "replaced", keys).to_string())
    print("\nFalsely selected and omitted generating terms per replicate")
    main_terms = terms[terms["variant"].isin(ORDER)].copy()
    main_terms["k"] = [ORDER.index(v) for v in main_terms["variant"]]
    print(main_terms.sort_values(["width", "k"]).drop(columns="k").to_string(index=False))
    print("\nExact McNemar tests (two-sided binomial test on discordant replicates) with Holm correction")
    show = tst.copy()
    for c in ("n", "recovered_A", "recovered_B", "only_A", "only_B"):
        if c in show:
            show[c] = show[c].astype("Int64")
    for c in ("p_exact", "p_holm"):
        show[c] = [f"{v:.3g}" if pd.notna(v) else "" for v in show[c]]
    print(show.to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
