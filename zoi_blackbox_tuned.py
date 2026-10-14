import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

import zoi_blackbox as bb
import zoi_core as z

GRIDS = {
    "hgb": {"learning_rate": [0.1, 0.2, 0.3], "max_leaf_nodes": [2, 3, 5], "min_samples_leaf": [20, 100]},
    "rf": {"max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2, 5, 20]},
}


def grid(name):
    keys = sorted(GRIDS[name])
    return [dict(zip(keys, v)) for v in itertools.product(*(GRIDS[name][k] for k in keys))]


def make(name, params, a):
    if name == "rf":
        return RandomForestClassifier(n_estimators=a.trees, n_jobs=a.jobs, random_state=a.seed, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=2000, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=20, random_state=a.seed, **params)
    raise ValueError(name)


def tune(X, y, g, name, a):
    splits = list(GroupKFold(n_splits=min(a.inner, len(np.unique(g)))).split(X, y, g))
    rows, best = [], None
    for params in grid(name):
        t0 = time.time()
        oof = np.zeros(len(y))
        for tr, va in splits:
            oof[va] = make(name, params, a).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        thr = bb.best_threshold(oof, y)
        f1 = z.metrics(y, oof >= thr)["F1"]
        rows.append({"params": json.dumps(params, sort_keys=True), "inner_F1": round(f1, 5),
                     "threshold": round(thr, 5), "seconds": round(time.time() - t0, 1)})
        if best is None or f1 > best[0] + 1e-12:
            best = (f1, thr, params)
    return best, rows


def read_rows(path):
    return pd.read_csv(path, dtype={"held_out": str}).to_dict("records") if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser(prog="zoi_blackbox_tuned")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("results", "blackbox_tuned"))
    ap.add_argument("--by", choices=["plot", "type", "both"], default="both")
    ap.add_argument("--models", default="hgb,rf")
    ap.add_argument("--features", choices=["raw", "sym"], default="sym")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f"tuned_transfer_{a.features}.csv")
    grid_path = os.path.join(a.out, f"tuned_grid_{a.features}.csv")
    done, inner = read_rows(res_path), read_rows(grid_path)
    finished = {(r["by"], r["held_out"], r["model"]) for r in done}
    plots, labels = z.load_folder(a.data)
    G = len(labels)
    sym = a.features == "sym"
    for by in (["plot", "type"] if a.by == "both" else [a.by]):
        for held, train, test in bb.folds(plots, by):
            for name in a.models.split(","):
                if (by, held, name) in finished:
                    print(f"skip {by} {held} {name}", flush=True)
                    continue
                t0 = time.time()
                X, y, g = bb.stack(train, G, sym)
                (f1, thr, params), rows = tune(X, y, g, name, a)
                model = make(name, params, a).fit(X, y)
                Xt, yt, _ = bb.stack(test, G, sym)
                met = z.metrics(yt, model.predict_proba(Xt)[:, 1] >= thr)
                done.append({"by": by, "held_out": held, "model": name, "F1": round(met["F1"], 4),
                             "MCC": round(met["MCC"], 4), "errors": met["errors"],
                             "params": json.dumps(params, sort_keys=True), "inner_F1": round(f1, 4),
                             "threshold": round(thr, 4), "nodes": bb.size(model),
                             "seconds": round(time.time() - t0, 1)})
                inner += [{"by": by, "held_out": held, "model": name, **r} for r in rows]
                pd.DataFrame(done).to_csv(res_path, index=False)
                pd.DataFrame(inner).to_csv(grid_path, index=False)
                r = done[-1]
                print(f"{by:5s} {held:10s} {name:4s} F1={r['F1']:.4f} MCC={r['MCC']:.4f} errors={r['errors']} "
                      f"inner_F1={r['inner_F1']:.4f} {r['params']} time={r['seconds']}s", flush=True)
    df = pd.DataFrame(done)
    summary = df.groupby(["by", "model"]).agg(F1_mean=("F1", "mean"), F1_min=("F1", "min"), F1_max=("F1", "max"),
                                             MCC_mean=("MCC", "mean"), errors=("errors", "sum")).round(4).reset_index()
    summary.to_csv(os.path.join(a.out, f"tuned_summary_{a.features}.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print()
    print(df.drop(columns=["threshold", "nodes"]).to_string(index=False))
    print()
    print(summary.to_string(index=False))
    compare = a.compare or os.path.join("results", "blackbox", f"blackbox_transfer_{a.features}.csv")
    if os.path.exists(compare):
        base = pd.read_csv(compare, dtype={"held_out": str})[["by", "held_out", "model", "F1", "MCC"]]
        cmp = df[["by", "held_out", "model", "F1", "MCC"]].merge(base, on=["by", "held_out", "model"],
                                                                 suffixes=("_tuned", "_default"))
        cmp["F1_gain"] = (cmp["F1_tuned"] - cmp["F1_default"]).round(4)
        cmp.to_csv(os.path.join(a.out, f"tuned_vs_default_{a.features}.csv"), index=False)
        print()
        print(f"Tuned against default settings ({compare})")
        print(cmp.to_string(index=False))
        print()
        print(cmp.groupby(["by", "model"])[["F1_tuned", "F1_default", "F1_gain"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
