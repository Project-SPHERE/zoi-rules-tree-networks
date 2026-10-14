import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

import zoi_core as z


def add_common(ap):
    ap.add_argument("--data", default="data")
    ap.add_argument("--plots", default=None)
    ap.add_argument("--attrs", default="dbh,h,v,soc,z")
    ap.add_argument("--transforms", default="id,sq,sqrt,log")
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--no-species", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pos-errors", default="0.25,0.5,1.0")
    ap.add_argument("--attr-errors", default="dbh=1,h=1")


def add_search(ap):
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--max-terms", type=int, default=4)
    ap.add_argument("--max-coef", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--penalty", type=float, default=0.002)
    ap.add_argument("--tau", default="auto")
    ap.add_argument("--pi", type=float, default=0.6)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--single-evals", type=int, default=0)
    ap.add_argument("--final", choices=["terms", "class"], default="terms")
    ap.add_argument("--calib-complete", action="store_true")


def tau_of(a):
    return None if a.tau == "auto" else float(a.tau)


def setup(a):
    plots, labels = z.load_folder(a.data, select=a.plots)
    attrs = [s for s in a.attrs.split(",") if s]
    lib = z.make_library(plots, attrs=attrs, transforms=a.transforms.split(","),
                         pair_ops=() if a.no_pair else ("absdiff", "prod"), same_species=not a.no_pair,
                         species=not a.no_species)
    os.makedirs(a.out, exist_ok=True)
    return plots, labels, lib, z.make_scales(plots, lib), len(labels)


def config(a):
    return z.SearchConfig(pop=a.pop, evals=a.evals, max_terms=a.max_terms, max_coef=a.max_coef,
                          delta=a.delta, penalty=a.penalty)


def show(title, rows):
    print(f"\n{title}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def select_rule(train, labels, lib, sc, G, a, single, seed):
    if single:
        cfg = config(a)
        cfg.evals = a.single_evals or 2 * a.runs * a.evals
        res = z.single_search(train, lib, sc, G, labels, cfg, seed=seed)
        model, _ = z.fit(res["selected"], train, sc, G, labels, margin=True)
        return model, {"mode": "single", "selected": res["selected"].key(), "runs": [res]}
    rep = z.stability(train, lib, sc, G, labels, config(a), B=a.runs, jobs=a.jobs, seed=seed, tau=tau_of(a), pi=a.pi,
                      final=a.final, calib_complete=a.calib_complete)
    w = rep["winner"]
    return rep["model"], {"mode": "stability", "selected": rep["model"].form.key(), "freq": w["freq"],
                          "lcb": w["lcb"], "stable": rep["stable"], "report": rep}


def stability_rows(rep):
    return [{"freq": c["freq"], "lcb95": c["lcb"], "coef": c["coef"], "F1_full": c["F1"], "jac_top": c["jac_top"],
             "members": len(c["forms"]), "representative": c["representative"].key()} for c in rep["classes"]]


def cmd_diag(a):
    plots, labels, lib, sc, G = setup(a)
    attrs = [s for s in a.attrs.split(",") if s]
    rows = []
    for p in plots:
        thr = z.distance_threshold([p])
        _, s, viol = z.free_radii(p)
        lg = z.fit_logistic([p], attrs, G)
        rows.append({"plot": p.name, "trees": p.n, "edges": int(p.y.sum()), "pairs": p.m,
                     "density": p.y.mean(), "F1_distance": z.metrics(p.y, p.d <= thr)["F1"],
                     "errors_free_radii": int(((s > -0.5e-3) != p.y).sum()),
                     "F1_logistic": z.evaluate(lg.predict, [p])["F1"]})
    show("In-sample diagnostics per plot", rows)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "diag.csv"), index=False)


def parse_errors(a):
    pos = tuple(float(v) for v in a.pos_errors.split(",") if v)
    attr = {k: float(v) for k, v in (kv.split("=") for kv in a.attr_errors.split(",") if kv)}
    return pos, attr


def report_model(model, plots, out, tag, a, conflicts=True):
    pos, attr = parse_errors(a)
    m = z.evaluate(model.predict, plots)
    print(f"\nRule: {model.form.key()}   coefficients={model.form.n_coef(model.G)}   margin_t={model.t:.4f} m")
    print(model.describe())
    print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    radii = np.concatenate([z.node_radii(model, p) for p in plots])
    print(f"Node radius over observed trees: min={radii.min():.3f} m  max={radii.max():.3f} m"
          + ("  (node part only, the form has pair terms)" if model.form.pair else ""))
    cert = z.certificate_table(model, plots, pos)
    cert_err = z.certificate_table(model, plots, pos, errors=attr)
    show("Certified share (position error only)", cert)
    show(f"Certified share (position error and attribute errors {attr})", cert_err)
    pd.DataFrame(model.coefficients()).to_csv(os.path.join(out, f"{tag}_coefficients.csv"), index=False)
    result = {"form": model.form.key(), "metrics": m, "certificates": cert, "certificates_attr": cert_err,
              "coefficients": model.coefficients(), "describe": model.describe(),
              "radius_min": float(radii.min()), "radius_max": float(radii.max())}
    if conflicts:
        sets = []
        for p in plots:
            cs = z.conflict_set(model.form, [p], model.scales, model.G)
            sets.append({"plot": p.name, "size": 0 if cs is None else cs["size"],
                         "minimal": None if cs is None else cs["minimal"], "pairs": [] if cs is None else cs["pairs"],
                         "delta_set": None if cs is None else cs["delta_set"],
                         "delta_all": None if cs is None else cs["delta_all"]})
        show("Conflict sets per plot at resolution eps, delta = largest exact separation (0 = no exact reproduction)",
             [{"plot": s["plot"], "size": s["size"], "minimal": s["minimal"],
               "helly_bound": model.form.n_coef(model.G) + 1,
               "delta_set": "" if s["delta_set"] is None else f"{s['delta_set']:.2e}",
               "delta_all": "" if s["delta_all"] is None else f"{s['delta_all']:.2e}"} for s in sets])
        pd.DataFrame([dict(r, plot=s["plot"]) for s in sets for r in s["pairs"]]).to_csv(
            os.path.join(out, f"{tag}_conflicts.csv"), index=False)
        result["conflicts"] = sets
    return result


def cmd_search(a):
    plots, labels, lib, sc, G = setup(a)
    summary = []
    for r in range(a.repeat):
        t0 = time.time()
        rep = z.stability(plots, lib, sc, G, labels, config(a), B=a.runs, jobs=a.jobs, seed=a.seed + 1000 * r,
                          tau=tau_of(a), pi=a.pi, final=a.final)
        print(f"\nRepeat {r + 1}/{a.repeat}   searches={rep['R']}   tau={rep['tau']:.4f}   time={time.time() - t0:.0f} s")
        show("Predictive equivalence classes", stability_rows(rep))
        show("Term selection frequency", [{"term": k, "freq": v} for k, v in rep["term_freq"].items()])
        print(f"\nMB/CPSS bound on expected falsely selected terms at pi={a.pi}: "
              f"q^2/((2pi-1)p) = {rep['q']:.2f}^2/((2*{a.pi}-1)*{rep['p']}) = {rep['bound']:.3f}")
        print("Stable class found" if rep["stable"] else "No class reached the stability threshold")
        show(f"Stable terms (freq >= {a.pi})", [{"term": t["term"], "freq": t["freq"], "scope_share": t["scope_share"]}
                                                for t in rep["stable_terms"]] or [{"term": "none"}])
        print(f"Final rule ({'stable terms + intercept' if rep['final_mode'] == 'terms' else 'most frequent class'}): "
              f"{rep['model'].form.key()}")
        if rep["terms_form"] is not None:
            print(f"Stable-term structure before the safeguard: {rep['terms_form'].key()}   F1={rep['terms_F1']:.4f}   "
                  f"most frequent class F1={rep['class_F1']:.4f}   accepted={rep['final_mode'] == 'terms'}")
        tag = f"search_r{r + 1}"
        res = report_model(rep["model"], plots, a.out, tag, a)
        fronts = [{"search": k, "coef": c, "F1": f1, "form": f.key()} for k, run in enumerate(rep["runs"])
                  for f, f1, c in run["front"]]
        pd.DataFrame(fronts).to_csv(os.path.join(a.out, f"{tag}_fronts.csv"), index=False)
        pd.DataFrame(stability_rows(rep)).to_csv(os.path.join(a.out, f"{tag}_classes.csv"), index=False)
        dump(os.path.join(a.out, f"{tag}.json"), {"classes": stability_rows(rep), "term_freq": rep["term_freq"],
                                                    "structural": rep["structural"], "bound": rep["bound"], "tau": rep["tau"],
                                                    "selected": rep["model"].form.key(), "final_mode": rep["final_mode"],
                                                    "terms_form": None if rep["terms_form"] is None else rep["terms_form"].key(),
                                                    "terms_F1": rep["terms_F1"], "class_F1": rep["class_F1"],
                                                    "stable_terms": [{k: v for k, v in t.items() if k != "gene"}
                                                                     for t in rep["stable_terms"]], **res})
        top_term = max(rep["term_freq"], key=rep["term_freq"].get)
        summary.append({"repeat": r + 1, "final_rule": rep["model"].form.key(), "mode": rep["final_mode"],
                        "top_class": rep["winner"]["representative"].key(), "class_freq": rep["winner"]["freq"],
                        "class_F1": rep["class_F1"], "top_term": top_term, "term_freq": rep["term_freq"][top_term],
                        "terms_form": "" if rep["terms_form"] is None else rep["terms_form"].key(),
                        "terms_F1": float("nan") if rep["terms_F1"] is None else rep["terms_F1"]})
    if a.repeat > 1:
        show("Selected rule per independent repeat", summary)
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(a.out, "repeats_summary.csv"), index=False)
        for col, val in (("top_class", "class_freq"), ("top_term", "term_freq")):
            top = df[col].mode()[0]
            f = df.loc[df[col] == top, val]
            print(f"{col}: {top}   most frequent in {len(f)} of {len(df)} repeats   "
                  f"{val} median={f.median():.2f} min={f.min():.2f} max={f.max():.2f}")
        print(f"final rules: {df['final_rule'].value_counts().to_dict()}")


def cmd_explain(a):
    plots, labels, lib, sc, G = setup(a)
    form = z.parse_form(a.form)
    extra = z.Library([(g[0], g[1]) for g in form.node if g[0] != "1"], list(form.pair))
    for key, v in z.make_scales(plots, extra).items():
        sc.setdefault(key, v)
    model, _ = z.fit(form, plots, sc, G, labels, margin=True)
    dump(os.path.join(a.out, "explain.json"), report_model(model, plots, a.out, "explain", a))


def folds(plots, by):
    if by == "type":
        keys = sorted({z.plot_type(p) for p in plots})
        return [(k, [p for p in plots if z.plot_type(p) != k], [p for p in plots if z.plot_type(p) == k]) for k in keys]
    return [(p.name, [q for q in plots if q is not p], [p]) for p in plots]


def cmd_transfer(a):
    plots, labels, lib, sc, G = setup(a)
    attrs = [s for s in a.attrs.split(",") if s]
    rows = []
    for k, (name, train, test) in enumerate(folds(plots, a.by)):
        t0 = time.time()
        model, info = select_rule(train, labels, lib, sc, G, a, a.single, a.seed + k)
        refit, _ = z.fit(model.form, test, sc, G, labels, margin=True)
        thr = z.distance_threshold(train)
        lg = z.fit_logistic(train, attrs, G)
        free = [z.free_radii(p) for p in test]
        frozen = z.evaluate(model.predict, test)
        row = {"held_out": name, "rule": info["selected"],
               "F1_frozen": frozen["F1"], "MCC_frozen": frozen["MCC"],
               "F1_form_refit": z.evaluate(refit.predict, test)["F1"],
               "F1_distance": z.evaluate(lambda p: p.d <= thr, test)["F1"],
               "F1_logistic": z.evaluate(lg.predict, test)["F1"],
               "errors_free_radii_insample": int(sum(((s > -0.5e-3) != p.y).sum() for p, (_, s, _) in zip(test, free))),
               "seconds": round(time.time() - t0, 1)}
        if not a.single:
            row["freq"] = info["freq"]
        rows.append(row)
        print(f"{name}: {row['rule']}  F1_frozen={row['F1_frozen']:.4f}  ({row['seconds']} s)")
    show(f"Transfer by {a.by} ({'single search' if a.single else 'stability selection'})", rows)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"transfer_{a.by}_{'single' if a.single else 'stability'}.csv"),
                              index=False)


def cmd_synth(a):
    plots, labels, lib, sc, G = setup(a)
    attrs = [s for s in a.attrs.split(",") if s]
    truth = z.truth_model(a.truth, labels)
    missing = [g for g in truth.form.node if g[0] != "1" and (g[0], g[1]) not in lib.node] + \
              [g for g in truth.form.pair if g not in lib.pair]
    if missing:
        print(f"warning: library cannot express the planted terms {missing}, exact recovery is impossible")
    print(f"Planted rule: {truth.form.key()}")
    print(truth.describe())
    test_mask = [bool(re.search(a.test, p.name)) for p in plots]
    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(a.seed + 7919 * rep)
        obs, clean = z.plant(plots, truth, rng, sigma=a.sigma, width=a.width, flip=a.flip)
        train = [o for o, t in zip(obs, test_mask) if not t]
        test_clean = [c for c, t in zip(clean, test_mask) if t]
        truth_pred = np.concatenate([truth.predict(p) for p in test_clean])
        dens = float(np.mean(np.concatenate([p.y for p in clean])))
        key = truth.form.key()
        seen = lambda runs: any(r["key"] == key for run in runs for r in run["archive"])
        methods = {}
        for single in [x for x in ([False, True] if not a.skip_stability else [True]) if not (x and a.skip_single)]:
            model, info = select_rule(train, labels, lib, sc, G, a, single, a.seed + rep)
            if single:
                methods["single"] = (model.predict, model.form, None, seen(info["runs"]))
            else:
                r = info["report"]
                methods["stability"] = (model.predict, model.form, r["final_mode"] == "class", seen(r["runs"]))
                methods["stability_terms"] = (None if r["terms_model"] is None else r["terms_model"].predict,
                                              r["terms_form"], None, None)
        if a.multistart:
            r = z.stability(train, lib, sc, G, labels, config(a), B=a.runs, jobs=a.jobs, seed=a.seed + rep,
                            tau=tau_of(a), pi=a.pi, final=a.final, subsample=False)
            methods["multi_agg"] = (r["model"].predict, r["model"].form, r["final_mode"] == "class", seen(r["runs"]))
            methods["multi_union"] = (r["union_model"].predict, r["union_form"], None, seen(r["runs"]))
        lg = z.fit_logistic(train, attrs, G)
        methods["logistic"] = (lg.predict, None, None, None)
        thr = z.distance_threshold(train)
        methods["distance"] = (lambda p, thr=thr: p.d <= thr, None, None, None)
        for name, (predict, form, guard, evaluated) in methods.items():
            if predict is None:
                f1 = jac = np.nan
            else:
                pred = np.concatenate([predict(p) for p in test_clean])
                f1, jac = z.metrics(truth_pred, pred)["F1"], z.jaccard(pred, truth_pred)
            rows.append({"rep": rep + 1, "method": name, "density": dens, "F1_vs_truth": f1, "jaccard_vs_truth": jac,
                         "exact_form": None if form is None else form == truth.form,
                         "safeguard": guard, "truth_evaluated": evaluated,
                         "coef": None if form is None else form.n_coef(G),
                         "form": "" if form is None else form.key()})
        print(f"rep {rep + 1}: " + "  ".join(f"{r['method']}={r['F1_vs_truth']:.4f}" for r in rows[-len(methods):]))
    df = pd.DataFrame(rows)
    show("Per replicate", rows)
    agg = df.groupby("method").agg(F1_mean=("F1_vs_truth", "mean"), F1_sd=("F1_vs_truth", "std"),
                                   jaccard_mean=("jaccard_vs_truth", "mean"),
                                   equivalent=("jaccard_vs_truth", lambda v: float(np.mean(v >= (tau_of(a) or 0.99)))),
                                   exact=("exact_form", lambda v: float(np.mean([x is True for x in v]))
                                          if v.notna().any() else np.nan),
                                   safeguard=("safeguard", lambda v: float(np.mean([x is True for x in v]))
                                              if v.notna().any() else np.nan)).reset_index()
    missed = df[df["exact_form"].eq(False) & df["truth_evaluated"].eq(True)].groupby("method").size()
    agg["failed_but_evaluated"] = agg["method"].map(missed).fillna(0).astype(int)
    show("Summary", agg.to_dict("records"))
    tag = f"synth_{a.truth}_s{a.sigma}_w{a.width}_f{a.flip}_m{a.evals}" + ("_ms" if a.multistart else "")
    df.to_csv(os.path.join(a.out, f"{tag}.csv"), index=False)
    agg.to_csv(os.path.join(a.out, f"{tag}_summary.csv"), index=False)


def cmd_power(a):
    plots, labels, lib, sc, G = setup(a)
    _, g, prof = z.fit_power(plots, G, labels)
    print(f"In-sample on all plots: best gamma={g:.2f}  errors={prof[g]}")
    print("  ".join(f"{k:.2f}:{v}" for k, v in prof.items()))
    for by in (["plot", "type"] if a.by == "both" else [a.by]):
        rows = []
        for name, train, test in folds(plots, by):
            model, g, _ = z.fit_power(train, G, labels)
            m = z.evaluate(model.predict, test)
            rows.append({"held_out": name, "gamma": g, "F1_power": m["F1"], "MCC_power": m["MCC"], "errors": m["errors"]})
            print(f"{name}: gamma={g:.2f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}", flush=True)
        show(f"Transfer of the power rule c_g + b_g dbh^gamma by {by}", rows)
        df = pd.DataFrame(rows)
        print(f"mean F1={df.F1_power.mean():.4f}  min F1={df.F1_power.min():.4f}  max F1={df.F1_power.max():.4f}  "
              f"mean MCC={df.MCC_power.mean():.4f}  gamma {df.gamma.min():.2f} to {df.gamma.max():.2f}")
        df.to_csv(os.path.join(a.out, f"power_{by}.csv"), index=False)


def cmd_export(a):
    plots, labels, lib, sc, G = setup(a)
    path = os.path.join(a.out, a.file)
    z.export_pairs(plots, path, [s for s in a.attrs.split(",") if s])
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(prog="zoi_run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diag")
    add_common(p)
    p = sub.add_parser("search")
    add_common(p)
    add_search(p)
    p.add_argument("--repeat", type=int, default=1)
    p = sub.add_parser("explain")
    add_common(p)
    p.add_argument("--form", required=True)
    p = sub.add_parser("transfer")
    add_common(p)
    add_search(p)
    p.add_argument("--by", choices=["plot", "type"], default="plot")
    p.add_argument("--single", action="store_true")
    p = sub.add_parser("synth")
    add_common(p)
    add_search(p)
    p.add_argument("--truth", choices=sorted(z.TRUTHS), default="species_dbh")
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--width", type=float, default=0.0)
    p.add_argument("--flip", type=float, default=0.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--test", default="_100$")
    p.add_argument("--skip-stability", action="store_true")
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--multistart", action="store_true")
    p = sub.add_parser("power")
    add_common(p)
    p.add_argument("--by", choices=["plot", "type", "both"], default="both")
    p = sub.add_parser("export")
    add_common(p)
    p.add_argument("--file", default="pairs.csv")
    a = ap.parse_args()
    {"diag": cmd_diag, "search": cmd_search, "explain": cmd_explain, "transfer": cmd_transfer,
     "synth": cmd_synth, "power": cmd_power, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
