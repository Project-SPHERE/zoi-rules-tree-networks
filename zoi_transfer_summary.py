import argparse
import glob
import os
import re

import pandas as pd

import zoi_core as z

COLS = ["F1_frozen", "MCC_frozen", "F1_form_refit"]


def load(root):
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "*_seed*", "transfer_*.csv"))):
        m = re.search(r"(plot|type)_(stability|single)_seed(\d+)", f.replace("\\", "/"))
        if m:
            d = pd.read_csv(f, dtype={"held_out": str})
            d["by"], d["mode"], d["seed"] = m.group(1), m.group(2), int(m.group(3))
            frames.append(d)
    if not frames:
        raise SystemExit(f"no transfer results in {root}")
    return pd.concat(frames, ignore_index=True)


def spread(df, keys):
    agg = {"runs": ("seed", "nunique")}
    for c in COLS:
        agg.update({f"{c}_mean": (c, "mean"), f"{c}_min": (c, "min"), f"{c}_max": (c, "max")})
    return df.groupby(keys).agg(**agg).round(4).reset_index()


def term_shares(d):
    rows = []
    for (by, mode), part in d.groupby(["by", "mode"]):
        counts = {}
        for rule in part["rule"]:
            for t in (z.parse_form(rule).terms() if isinstance(rule, str) else set()):
                counts[t] = counts.get(t, 0) + 1
        rows += [{"by": by, "mode": mode, "term": t, "runs": k, "of": len(part), "share": round(k / len(part), 3)}
                 for t, k in sorted(counts.items(), key=lambda kv: -kv[1])]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(prog="zoi_transfer_summary")
    ap.add_argument("--root", default=os.path.join("results", "transfer_seeds"))
    ap.add_argument("--blackbox", nargs="+", default=[os.path.join("results", "blackbox_tuned", "tuned_transfer_sym.csv")])
    ap.add_argument("--out", default=os.path.join("results", "transfer_summary"))
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    d = load(a.root)
    keys = ["by", "mode", "held_out"]
    per = spread(d, keys)
    rules = d.groupby(keys)["rule"].agg(top_rule=lambda s: s.value_counts().index[0],
                                        top_rule_runs=lambda s: int(s.value_counts().iloc[0]),
                                        distinct_rules="nunique").reset_index()
    per = per.merge(rules, on=keys)
    overall = spread(d.groupby(["by", "mode", "seed"])[COLS].mean().reset_index(), ["by", "mode"])
    terms = term_shares(d)
    per.to_csv(os.path.join(out, "transfer_seeds_per_heldout.csv"), index=False)
    overall.to_csv(os.path.join(out, "transfer_seeds_overall.csv"), index=False)
    terms.to_csv(os.path.join(out, "transfer_seeds_terms.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("Per held-out plot or type, mean, min and max over seeds")
    print(per.to_string(index=False))
    print("\nMean over held-out plots or types, then mean, min and max over seeds")
    print(overall.to_string(index=False))
    print("\nShare of runs whose rule contains a term")
    print(terms.to_string(index=False))
    files = [f for f in a.blackbox if os.path.exists(f)]
    if files:
        tuned = pd.concat([pd.read_csv(f, dtype={"held_out": str}) for f in files], ignore_index=True)
        tuned = tuned.sort_values("inner_F1").drop_duplicates(["by", "held_out", "model"], keep="last")
        bbx = tuned.pivot_table(index=["by", "held_out"], columns="model", values="F1")
        bbx.columns = [f"F1_{c}_tuned" for c in bbx.columns]
        stab = per[per["mode"] == "stability"].set_index(["by", "held_out"])
        cmp = stab[["F1_frozen_mean", "F1_frozen_min", "F1_frozen_max"]].join(bbx, how="left")
        for c in bbx.columns:
            cmp[f"gap_min_{c[3:-6]}"] = (cmp["F1_frozen_min"] - cmp[c]).round(4)
        cmp = cmp.reset_index()
        cmp.to_csv(os.path.join(out, "transfer_seeds_vs_blackbox.csv"), index=False)
        print(f"\nFixed rule against tuned black-box models ({', '.join(files)}), gap_min = worst seed minus black box")
        print(cmp.to_string(index=False))
        print(cmp.drop(columns="held_out").groupby("by").mean().round(4).to_string())


if __name__ == "__main__":
    main()
