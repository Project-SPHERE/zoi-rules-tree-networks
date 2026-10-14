import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

import zoi_core as z


def features(plot, G, sym=False):
    a = plot.attrs
    i, j = plot.i, plot.j
    swap = (a["dbh"][i] > a["dbh"][j]) | ((a["dbh"][i] == a["dbh"][j]) & (i > j))
    s = np.where(swap, j, i)
    l = np.where(swap, i, j)
    cols = [plot.d]
    for k in ("dbh", "soc", "z"):
        cols += [a[k][s], a[k][l]]
        if sym:
            cols += [a[k][s] + a[k][l], np.abs(a[k][s] - a[k][l])]
    eye = np.eye(G)
    return np.column_stack(cols + [eye[plot.group[s]], eye[plot.group[l]]])


def stack(plots, G, sym=False):
    X = np.vstack([features(p, G, sym) for p in plots])
    y = np.concatenate([p.y for p in plots]).astype(bool)
    g = np.concatenate([np.full(p.m, k) for k, p in enumerate(plots)])
    return X, y, g


def make(name, jobs, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=jobs, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.1, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    raise ValueError(name)


def size(model):
    if isinstance(model, RandomForestClassifier):
        return int(sum(t.tree_.node_count for t in model.estimators_))
    try:
        return int(sum(p.nodes.shape[0] for it in model._predictors for p in it))
    except Exception:
        return -1


def best_threshold(p, y):
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    f1 = 2 * tp / (tp + fp + ys.sum())
    return float(p[o][int(np.argmax(f1))])


def run(train, test, name, G, jobs, seed, sym=False):
    X, y, g = stack(train, G, sym)
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=min(5, len(train))).split(X, y, g):
        oof[va] = make(name, jobs, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    thr = best_threshold(oof, y)
    model = make(name, jobs, seed).fit(X, y)
    Xt, yt, _ = stack(test, G, sym)
    met = z.metrics(yt, model.predict_proba(Xt)[:, 1] >= thr)
    return met, thr, size(model)


def folds(plots, by):
    if by == "type":
        keys = sorted({z.plot_type(p) for p in plots})
        return [(k, [p for p in plots if z.plot_type(p) != k], [p for p in plots if z.plot_type(p) == k]) for k in keys]
    return [(p.name, [q for q in plots if q is not p], [p]) for p in plots]


def main():
    ap = argparse.ArgumentParser(prog="zoi_blackbox")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("results", "blackbox"))
    ap.add_argument("--by", choices=["plot", "type", "both"], default="both")
    ap.add_argument("--models", default="rf,hgb")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--features", choices=["raw", "sym"], default="raw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plots, labels = z.load_folder(a.data)
    G = len(labels)
    rows = []
    for by in (["plot", "type"] if a.by == "both" else [a.by]):
        for held, train, test in folds(plots, by):
            for name in a.models.split(","):
                t0 = time.time()
                met, thr, nodes = run(train, test, name, G, a.jobs, a.seed, a.features == "sym")
                rows.append({"by": by, "held_out": held, "model": name, "F1": round(met["F1"], 4),
                             "MCC": round(met["MCC"], 4), "errors": met["errors"], "threshold": round(thr, 4),
                             "nodes": nodes, "seconds": round(time.time() - t0, 1)})
                print(f"{by:5s} {held:10s} {name:4s} F1={met['F1']:.4f} MCC={met['MCC']:.4f} "
                      f"errors={met['errors']} nodes={nodes} time={rows[-1]['seconds']}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"blackbox_transfer_{a.features}.csv"), index=False)
    summary = df.groupby(["by", "model"]).agg(F1_mean=("F1", "mean"), F1_min=("F1", "min"), F1_max=("F1", "max"),
                                             MCC_mean=("MCC", "mean"), nodes_mean=("nodes", "mean")).reset_index()
    summary.to_csv(os.path.join(a.out, f"blackbox_summary_{a.features}.csv"), index=False)
    print()
    print(df.to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
