import glob
import io
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack, identity, issparse
from scipy.special import expit
from scipy.stats import beta as beta_dist

NUMERIC = ("dbh", "h", "v", "soc", "z")

TRANSFORMS = {
    "id": lambda x: x,
    "sq": lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.maximum(x, 1e-12)),
    "log": lambda x: np.log(np.maximum(x, 1e-12)),
}
POSITIVE_ONLY = {"sqrt", "log"}
for _p in (25, 33, 67, 75, 125, 150):
    TRANSFORMS[f"p{_p:03d}"] = (lambda q: lambda x: np.power(np.maximum(x, 1e-12), q))(_p / 100)
    POSITIVE_ONLY.add(f"p{_p:03d}")
for _g in range(20, 151):
    TRANSFORMS[f"g{_g:03d}"] = (lambda q: lambda x: np.power(np.maximum(x, 1e-12), q))(_g / 100)
    POSITIVE_ONLY.add(f"g{_g:03d}")


@dataclass
class Plot:
    name: str
    names: np.ndarray
    xy: np.ndarray
    attrs: dict
    species: np.ndarray
    group: np.ndarray
    i: np.ndarray
    j: np.ndarray
    d: np.ndarray
    y: np.ndarray

    @property
    def n(self):
        return len(self.xy)

    @property
    def m(self):
        return len(self.d)


def is_network_file(path):
    with open(path, encoding="latin-1") as f:
        return f.readline().strip().startswith("$vertices")


def read_raw(path):
    with open(path, encoding="latin-1") as f:
        txt = f.read().replace("\r", "")
    v, e = txt.split("$edges")
    V = pd.read_csv(io.StringIO(v.replace("$vertices", "").strip()), sep=r"\s+")
    E = pd.read_csv(io.StringIO(e.strip()), sep=r"\s+")
    return V, E


def distances(xy, i, j):
    return np.sqrt(((xy[i] - xy[j]) ** 2).sum(1))


def build_plot(name, V, E, group_of, other):
    n = len(V)
    names = V["name"].to_numpy()
    idx = {nm: k for k, nm in enumerate(names)}
    A = np.zeros((n, n), bool)
    for a, b in zip(E["from"], E["to"]):
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = True
    xy = V[["x", "y"]].to_numpy(float)
    i, j = np.triu_indices(n, 1)
    attrs = {a: V[a].to_numpy(float) for a in NUMERIC if a in V.columns}
    sp = V["dv"].to_numpy()
    grp = np.array([group_of.get(s, other) for s in sp])
    return Plot(name, names, xy, attrs, sp, grp, i, j, distances(xy, i, j), A[i, j])


def load_folder(folder, pattern="*.txt", select=None, min_species=20):
    paths = sorted(p for p in glob.glob(os.path.join(folder, pattern)) if is_network_file(p))
    if select:
        paths = [p for p in paths if re.search(select, os.path.splitext(os.path.basename(p))[0])]
    if not paths:
        raise FileNotFoundError(f"no network files in {folder}")
    raw = {os.path.splitext(os.path.basename(p))[0]: read_raw(p) for p in paths}
    counts = pd.concat([V["dv"] for V, _ in raw.values()]).value_counts()
    major = sorted(counts[counts >= min_species].index)
    labels = [str(c) for c in major] + ["other"]
    group_of = {c: k for k, c in enumerate(major)}
    plots = [build_plot(k, V, E, group_of, len(major)) for k, (V, E) in raw.items()]
    return plots, labels


def induced(plot, keep):
    keep = np.asarray(keep, bool)
    remap = np.cumsum(keep) - 1
    sel = keep[plot.i] & keep[plot.j]
    return Plot(plot.name, plot.names[keep], plot.xy[keep], {a: v[keep] for a, v in plot.attrs.items()},
                plot.species[keep], plot.group[keep], remap[plot.i[sel]], remap[plot.j[sel]],
                plot.d[sel], plot.y[sel])


def moved(plot, xy):
    return replace(plot, xy=xy, d=distances(xy, plot.i, plot.j))


def plot_type(plot):
    return plot.name.split("_")[0]


@dataclass(frozen=True)
class Form:
    node: tuple = ()
    pair: tuple = ()

    @staticmethod
    def make(node=(), pair=()):
        return Form(tuple(sorted(set(node))), tuple(sorted(set(pair))))

    def key(self):
        a = " + ".join(node_label(g) for g in self.node)
        b = " + ".join(f"{op}({at})" for at, op in self.pair)
        return a + (" | " + b if b else "")

    def n_terms(self):
        return len(self.node) + len(self.pair)

    def n_coef(self, G):
        return sum(G if s == "sp" else 1 for _, _, s in self.node) + len(self.pair)

    def terms(self):
        return {f"{a}" if t == "id" else f"{t}({a})" for a, t, _ in self.node if a != "1"} | \
            {f"{op}({a})" for a, op in self.pair}

    def __str__(self):
        return self.key()


def node_label(g):
    a, t, s = g
    base = a if (t == "id" or a == "1") else f"{t}({a})"
    return f"{base}@{s}"


def parse_form(text):
    node, pair = [], []
    parts = text.split("|")
    for tok in parts[0].split("+"):
        tok = tok.strip()
        if not tok:
            continue
        base, scope = tok.split("@")
        m = re.fullmatch(r"(\w+)\((\w+)\)", base.strip())
        node.append((m.group(2), m.group(1), scope.strip()) if m else (base.strip(), "id", scope.strip()))
    for tok in (parts[1].split("+") if len(parts) > 1 else []):
        tok = tok.strip()
        if tok:
            m = re.fullmatch(r"(\w+)\((\w+)\)", tok)
            pair.append((m.group(2), m.group(1)))
    return Form.make(node, pair)


@dataclass
class Library:
    node: list
    pair: list
    species: bool = True


def make_library(plots, attrs=NUMERIC, transforms=("id", "sq", "sqrt", "log"), pair_ops=("absdiff", "prod"),
                 same_species=True, species=True):
    node = []
    for a in attrs:
        vals = np.concatenate([p.attrs[a] for p in plots])
        for t in transforms:
            if t in POSITIVE_ONLY and vals.min() <= 0:
                continue
            node.append((a, t))
    pair = [(a, op) for a in attrs for op in pair_ops]
    if same_species:
        pair.append(("species", "same"))
    return Library(node, pair, species)


def make_scales(plots, lib):
    sc = {("1", "id"): 1.0}
    for a, t in lib.node:
        v = np.concatenate([TRANSFORMS[t](p.attrs[a]) for p in plots])
        sc[(a, t)] = float(np.max(np.abs(v))) or 1.0
    for a, op in lib.pair:
        if a == "species":
            sc[(a, op)] = 1.0
            continue
        v = np.concatenate([p.attrs[a] for p in plots])
        sc[(a, op)] = (float(np.ptp(v)) if op == "absdiff" else float(np.max(np.abs(v))) ** 2) or 1.0
    return sc


def node_columns(plot, form, scales, G):
    cols = []
    for a, t, s in form.node:
        v = np.ones(plot.n) if a == "1" else TRANSFORMS[t](plot.attrs[a])
        if scales is not None:
            v = v / scales[(a, t)]
        if s == "sp":
            cols.extend(v * (plot.group == g) for g in range(G))
        else:
            cols.append(v)
    return np.column_stack(cols) if cols else np.zeros((plot.n, 0))


def pair_columns(plot, form, scales):
    cols = []
    for a, op in form.pair:
        if a == "species":
            v = (plot.species[plot.i] == plot.species[plot.j]).astype(float)
        else:
            xi, xj = plot.attrs[a][plot.i], plot.attrs[a][plot.j]
            v = np.abs(xi - xj) if op == "absdiff" else xi * xj
        if scales is not None:
            v = v / scales[(a, op)]
        cols.append(v)
    return np.column_stack(cols) if cols else np.zeros((plot.m, 0))


def design(plot, form, scales, G):
    P = node_columns(plot, form, scales, G)
    return np.hstack([P[plot.i] + P[plot.j], pair_columns(plot, form, scales)])


def stack(plots, form, scales, G):
    X = np.vstack([design(p, form, scales, G) for p in plots])
    d = np.concatenate([p.d for p in plots])
    y = np.concatenate([p.y for p in plots])
    return X, d, y


def owners(plots):
    return (np.concatenate([np.full(p.m, k) for k, p in enumerate(plots)]),
            np.concatenate([np.arange(p.m) for p in plots]))


def _lp(X, d, y, rows, margin, C, eps, t_max, theta_lb, method="highs"):
    k, p = len(rows), X.shape[1]
    sign = np.where(y[rows], -1.0, 1.0)
    Xw = X[rows]
    A_theta = csr_matrix(Xw.multiply(sign[:, None])) if issparse(Xw) else csr_matrix(Xw * sign[:, None])
    A = hstack([A_theta, csr_matrix(np.ones((k, 1))), -identity(k, format="csr")], format="csr")
    b = np.where(y[rows], -d[rows], d[rows] - eps)
    c = np.r_[np.zeros(p), -1.0 if margin else 0.0, np.full(k, C)]
    bounds = [(theta_lb, None)] * p + [(0.0, t_max if margin else 0.0)] + [(0.0, None)] * k
    return linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=method)


def _initial_rows(d, y, size):
    o = np.argsort(d)
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    thr = d[o][int(np.argmax(2 * tp / (tp + fp + y.sum())))]
    return np.sort(np.argsort(np.abs(d - thr))[:size])


def solve(X, d, y, margin=True, C=1000.0, eps=1e-3, t_max=5.0, theta_lb=None, init=1500, add=1500, max_iter=100):
    m, p = len(d), X.shape[1]
    rows = np.arange(m) if m <= init else _initial_rows(d, y, init)
    for _ in range(max_iter):
        for method in ("highs", "highs-ds", "highs-ipm"):
            res = _lp(X, d, y, rows, margin, C, eps, t_max, theta_lb, method)
            if res.status == 0:
                break
        if res.status != 0:
            raise RuntimeError(res.message)
        theta, t = res.x[:p], res.x[p]
        s = X @ theta - d
        need = np.where(y, t - s, s + eps + t)
        need[rows] = -np.inf
        out = np.flatnonzero(need > 1e-9)
        if out.size == 0:
            duals = np.zeros(m)
            duals[rows] = -res.ineqlin.marginals
            violation = float(np.maximum(np.where(y, t - s, s + eps + t), 0).sum())
            return theta, float(t), s, duals, violation
        rows = np.union1d(rows, out[np.argsort(-need[out])[:add]])
    raise RuntimeError("constraint generation did not converge")


@dataclass
class Model:
    form: Form
    theta: np.ndarray
    t: float
    violation: float
    scales: dict
    G: int
    labels: list
    eps: float = 1e-3

    def score(self, plot):
        return design(plot, self.form, self.scales, self.G) @ self.theta - plot.d

    def predict(self, plot):
        return self.score(plot) > -0.5 * self.eps

    def coefficients(self):
        rows, k = [], 0
        for a, t, s in self.form.node:
            sc = self.scales[(a, t)] if self.scales is not None else 1.0
            term = "1" if a == "1" else (a if t == "id" else f"{t}({a})")
            for g in (range(self.G) if s == "sp" else [None]):
                rows.append({"term": term, "group": "all" if g is None else self.labels[g],
                             "value": self.theta[k] / sc})
                k += 1
        for a, op in self.form.pair:
            sc = self.scales[(a, op)] if self.scales is not None else 1.0
            rows.append({"term": f"{op}({a})", "group": "pair", "value": self.theta[k] / sc})
            k += 1
        return rows

    def describe(self):
        coefs = self.coefficients()
        lines = []
        for g in self.labels:
            parts = []
            for c in coefs:
                if c["group"] in ("all", g):
                    parts.append(f"{c['value']:+.4g}" + ("" if c["term"] == "1" else f"*{c['term']}"))
            lines.append(f"r[{g}] = " + (" ".join(parts) if parts else "0"))
        pair = [f"{c['value']:+.4g}*{c['term']}" for c in coefs if c["group"] == "pair"]
        lines.append("s_ij = r(i) + r(j) - d_ij" + (" " + " ".join(pair) if pair else "") + "   edge iff s_ij >= 0")
        return "\n".join(lines)


def fit(form, plots, scales, G, labels, margin=True, C=1000.0, eps=1e-3):
    X, d, y = stack(plots, form, scales, G)
    theta, t, s, _, viol = solve(X, d, y, margin=margin, C=C, eps=eps)
    model = Model(form, theta, t, viol, scales, G, labels, eps)
    return model, metrics(y, s > -0.5 * eps)


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp = int(np.sum(y & p)), int(np.sum(~y & p))
    fn, tn = int(np.sum(y & ~p)), int(np.sum(~y & ~p))
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"F1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "MCC": (tp * tn - fp * fn) / den if den else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "errors": fp + fn}


def evaluate(predict, plots, truth=None):
    y = np.concatenate([(p.y if truth is None else truth[k].y) for k, p in enumerate(plots)])
    return metrics(y, np.concatenate([predict(p) for p in plots]))


def jaccard(a, b):
    u = np.sum(a | b)
    return float(np.sum(a & b) / u) if u else 1.0


@dataclass
class SearchConfig:
    pop: int = 24
    evals: int = 200
    max_terms: int = 4
    max_coef: int = 16
    pc: float = 0.5
    pm: float = 0.9
    penalty: float = 0.002
    delta: float = 0.005
    immigrants: float = 0.15
    p_pair: float = 0.15
    p_species: float = 0.5
    screen: float = 0.05
    C: float = 1000.0


class Searcher:
    def __init__(self, plots, lib, scales, G, labels, cfg, rng):
        self.plots, self.lib, self.scales, self.G, self.labels = plots, lib, scales, G, labels
        self.cfg, self.rng = cfg, rng
        self.archive = {}
        self.attempts = 0
        self.best = None
        self.d = np.concatenate([p.d for p in plots])
        self.y = np.concatenate([p.y for p in plots])
        self.rows0 = _initial_rows(self.d, self.y, 1500) if len(self.d) > 1500 else np.arange(len(self.d))

    def _record(self, form, s, approx):
        m = metrics(self.y, s > -0.5e-3)
        return {"form": form, "key": form.key(), "F1": m["F1"], "MCC": m["MCC"], "errors": m["errors"],
                "coef": form.n_coef(self.G), "terms": form.n_terms(), "approx": approx}

    def exact(self, form):
        X = np.vstack([design(p, form, self.scales, self.G) for p in self.plots])
        try:
            s = X @ solve(X, self.d, self.y, margin=False, C=self.cfg.C)[0] - self.d
        except RuntimeError:
            s = np.full(len(self.d), -1.0)
        rec = self._record(form, s, False)
        self.archive[rec["key"]] = rec
        self.best = rec["F1"] if self.best is None else max(self.best, rec["F1"])
        return rec

    def evaluate(self, form):
        self.attempts += 1
        k = form.key()
        if k in self.archive:
            return self.archive[k]
        if self.best is not None and self.cfg.screen > 0 and len(self.d) > len(self.rows0):
            X = np.vstack([design(p, form, self.scales, self.G) for p in self.plots])
            res = _lp(X, self.d, self.y, self.rows0, False, self.cfg.C, 1e-3, 5.0, None)
            if res.status == 0:
                rec = self._record(form, X @ res.x[:X.shape[1]] - self.d, True)
                if rec["F1"] < self.best - self.cfg.screen:
                    self.archive[k] = rec
                    return rec
        return self.exact(form)

    def finalise(self):
        while True:
            pending = [r for r in pareto(self.archive) if r["approx"]]
            if not pending:
                return self.archive
            for r in pending:
                self.exact(r["form"])

    def fitness(self, r):
        return r["F1"] - self.cfg.penalty * r["coef"]

    def better(self, a, b):
        fa, fb = self.fitness(a), self.fitness(b)
        return fa > fb + 1e-12 or (abs(fa - fb) <= 1e-12 and a["coef"] < b["coef"])

    def scope(self):
        return "sp" if self.lib.species and self.rng.random() < self.cfg.p_species else "all"

    def random_gene(self, node, pair):
        free_pair = [g for g in self.lib.pair if g not in pair]
        free_node = [g for g in [("1", "id")] + self.lib.node if g not in node]
        if free_pair and (not free_node or self.rng.random() < self.cfg.p_pair):
            return "pair", free_pair[self.rng.integers(len(free_pair))]
        if free_node:
            return "node", free_node[self.rng.integers(len(free_node))]
        return None, None

    def add_random(self, node, pair):
        kind, g = self.random_gene(node, pair)
        if kind == "node":
            node[g] = self.scope()
        elif kind == "pair":
            pair.add(g)

    def repair(self, node, pair):
        node, pair = dict(node), set(pair)
        while len(node) + len(pair) > self.cfg.max_terms:
            genes = [("n", g) for g in node] + [("p", g) for g in sorted(pair)]
            kind, g = genes[self.rng.integers(len(genes))]
            (node.pop(g) if kind == "n" else pair.discard(g))
        while sum(self.G if s == "sp" else 1 for s in node.values()) + len(pair) > self.cfg.max_coef:
            sp = [g for g, s in node.items() if s == "sp"]
            if not sp:
                break
            node[sp[self.rng.integers(len(sp))]] = "all"
        if not node and not pair:
            node[("1", "id")] = "all"
        return Form.make([(a, t, s) for (a, t), s in node.items()], pair)

    def random_form(self):
        node, pair = {}, set()
        if self.rng.random() < 0.9:
            node[("1", "id")] = self.scope()
        for _ in range(self.rng.integers(1, 3)):
            self.add_random(node, pair)
        return self.repair(node, pair)

    def mutate(self, form):
        node = {(a, t): s for a, t, s in form.node}
        pair = set(form.pair)
        op = self.rng.choice(["add", "remove", "scope", "swap", "transform"])
        genes = [("n", g) for g in node] + [("p", g) for g in sorted(pair)]
        if op == "add" or not genes:
            self.add_random(node, pair)
        elif op == "remove" and len(genes) > 1:
            kind, g = genes[self.rng.integers(len(genes))]
            (node.pop(g) if kind == "n" else pair.discard(g))
        elif op == "scope" and node and self.lib.species:
            g = list(node)[self.rng.integers(len(node))]
            node[g] = "all" if node[g] == "sp" else "sp"
        elif op == "transform":
            cand = [g for g in node if g[0] != "1"]
            if cand:
                a, t = cand[self.rng.integers(len(cand))]
                alts = [g for g in self.lib.node if (g[0] == a or g[1] == t) and g not in node]
                if alts:
                    node[alts[self.rng.integers(len(alts))]] = node.pop((a, t))
            else:
                self.add_random(node, pair)
        else:
            kind, g = genes[self.rng.integers(len(genes))]
            (node.pop(g) if kind == "n" else pair.discard(g))
            self.add_random(node, pair)
        return self.repair(node, pair)

    def crossover(self, f1, f2):
        node, pair = {}, set()
        for a, t, s in f1.node + f2.node:
            if self.rng.random() < 0.5:
                node[(a, t)] = s
        for g in f1.pair + f2.pair:
            if self.rng.random() < 0.5:
                pair.add(g)
        return self.repair(node, pair)

    def run(self):
        cfg = self.cfg
        seeds = [Form.make([("1", "id", "all")])]
        if self.lib.species:
            seeds.append(Form.make([("1", "id", "sp")]))
        pop = [self.evaluate(f) for f in seeds]
        while len(pop) < cfg.pop:
            pop.append(self.evaluate(self.random_form()))
        limit = 30 * cfg.evals
        while len(self.archive) < cfg.evals and self.attempts < limit:
            for idx in range(len(pop)):
                if len(self.archive) >= cfg.evals:
                    break
                if self.rng.random() < cfg.pc:
                    child = self.evaluate(self.crossover(pop[idx]["form"], pop[self.rng.integers(len(pop))]["form"]))
                    if self.better(child, pop[idx]):
                        pop[idx] = child
                if self.rng.random() < cfg.pm:
                    mutant = self.evaluate(self.mutate(pop[idx]["form"]))
                    if self.better(mutant, pop[idx]):
                        pop[idx] = mutant
            order = sorted(range(len(pop)), key=lambda k: self.fitness(pop[k]))
            for k in order[: int(cfg.immigrants * len(pop))]:
                pop[k] = self.evaluate(self.random_form())
        return self.finalise()


def select(archive, delta):
    best = max(r["F1"] for r in archive.values())
    cands = [r for r in archive.values() if r["F1"] >= best - delta]
    return min(cands, key=lambda r: (r["coef"], r["terms"], -r["F1"], r["key"]))


def pareto(archive):
    front, best = [], -1.0
    for r in sorted(archive.values(), key=lambda r: (r["coef"], -r["F1"], r["key"])):
        if r["F1"] > best + 1e-12:
            front.append(r)
            best = r["F1"]
    return front


def search_task(args):
    plots, lib, scales, G, labels, cfg, seed = args
    archive = Searcher(plots, lib, scales, G, labels, cfg, np.random.default_rng(seed)).run()
    sel = select(archive, cfg.delta)
    return {"selected": sel["form"], "F1": sel["F1"], "evals": len(archive),
            "front": [(r["form"], r["F1"], r["coef"]) for r in pareto(archive)],
            "archive": [{k: r[k] for k in ("form", "key", "F1", "coef", "terms", "approx")} for r in archive.values()]}


def union_select(runs, delta):
    arch = {}
    for run in runs:
        for r in run["archive"]:
            if r["key"] not in arch or (arch[r["key"]]["approx"] and not r["approx"]):
                arch[r["key"]] = r
    return select(arch, delta)


def run_tasks(fn, tasks, jobs):
    if jobs <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, tasks))


def halves(plots, rng):
    A, B = [], []
    for p in plots:
        perm, h = rng.permutation(p.n), p.n // 2
        first, second = np.zeros(p.n, bool), np.zeros(p.n, bool)
        first[perm[:h]] = True
        second[perm[h:2 * h]] = True
        A.append(induced(p, first))
        B.append(induced(p, second))
    return A, B


def single_search(plots, lib, scales, G, labels, cfg, seed=0):
    return search_task((plots, lib, scales, G, labels, cfg, seed))


def stability(plots, lib, scales, G, labels, cfg, B=10, jobs=1, seed=0, tau=None, pi=0.6, alpha=0.05, calib=20,
              final="terms", subsample=True, calib_complete=False):
    rng = np.random.default_rng(seed)
    tasks, calib_sets = [], []
    for _ in range(B):
        for half in halves(plots, rng):
            calib_sets.append(half)
            tasks.append((half if subsample else plots, lib, scales, G, labels, cfg, int(rng.integers(2 ** 31))))
    runs = run_tasks(search_task, tasks, jobs)
    chosen = [r["selected"] for r in runs]
    R = len(chosen)
    uniq = sorted(set(chosen), key=lambda f: (f.n_coef(G), f.n_terms(), f.key()))
    preds, full = {}, {}
    for f in uniq:
        model, m = fit(f, plots, scales, G, labels, margin=True, C=cfg.C)
        full[f] = (model, m)
        preds[f] = np.concatenate([model.predict(p) for p in plots])
    top = max(uniq, key=lambda f: (chosen.count(f), -f.n_coef(G)))
    if tau is None:
        if calib_complete:
            present = set(np.concatenate([p.group for p in plots]).tolist())
            full_sets = [h for h in calib_sets if set(np.concatenate([q.group for q in h]).tolist()) >= present]
            calib_sets = full_sets or calib_sets
        jac = [jaccard(np.concatenate([fit(top, h, scales, G, labels, margin=True, C=cfg.C)[0].predict(p)
                                       for p in plots]), preds[top]) for h in calib_sets[:calib]]
        tau = float(np.quantile(jac, 0.05))
    classes = []
    for f in uniq:
        for cl in classes:
            if jaccard(preds[f], preds[cl[0]]) >= tau:
                cl.append(f)
                break
        else:
            classes.append([f])
    table = []
    for cl in classes:
        k = sum(chosen.count(f) for f in cl)
        table.append({"forms": cl, "representative": cl[0], "count": k, "freq": k / R,
                      "lcb": float(beta_dist.ppf(alpha, k, R - k + 1)) if k else 0.0,
                      "coef": cl[0].n_coef(G), "F1": full[cl[0]][1]["F1"]})
    table.sort(key=lambda r: (-r["count"], r["coef"]))
    for r in table:
        r["jac_top"] = jaccard(preds[r["representative"]], preds[table[0]["representative"]])
    stable = [r for r in table if r["freq"] >= pi]
    winner = min(stable, key=lambda r: r["coef"]) if stable else table[0]
    terms = {}
    for f in chosen:
        for t in f.terms():
            terms[t] = terms.get(t, 0) + 1
    term_freq = {t: c / R for t, c in sorted(terms.items(), key=lambda kv: -kv[1])}
    p_lib = len(lib.node) + len(lib.pair)
    q = float(np.mean([len(f.terms()) for f in chosen]))
    bound = q * q / ((2 * pi - 1) * p_lib) if pi > 0.5 else float("nan")
    structural = {}
    for f in chosen:
        structural[f.key()] = structural.get(f.key(), 0) + 1
    node_scopes, pair_count, icpt = {}, {}, []
    for f in chosen:
        for a, t, s in f.node:
            (icpt.append(s) if a == "1" else node_scopes.setdefault((a, t), []).append(s))
        for g in f.pair:
            pair_count[g] = pair_count.get(g, 0) + 1
    stable_terms = []
    for (a, t), v in node_scopes.items():
        if len(v) / R >= pi:
            s = max(sorted(set(v)), key=v.count)
            stable_terms.append({"gene": (a, t, s), "term": node_label((a, t, s)), "freq": len(v) / R,
                                 "scope_share": v.count(s) / len(v)})
    for g, k in pair_count.items():
        if k / R >= pi:
            stable_terms.append({"gene": g, "term": f"{g[1]}({g[0]})", "freq": k / R, "scope_share": 1.0})
    stable_terms.sort(key=lambda r: -r["freq"])
    model, mode, final_form = full[winner["representative"]][0], "class", winner["representative"]
    terms_form, terms_model, terms_F1 = None, None, None
    if final == "terms" and stable_terms:
        scope = max(sorted(set(icpt)), key=icpt.count) if icpt else "all"
        final_form = Form.make([("1", "id", scope)] + [r["gene"] for r in stable_terms if len(r["gene"]) == 3],
                               [r["gene"] for r in stable_terms if len(r["gene"]) == 2])
        cand, cand_m = fit(final_form, plots, scales, G, labels, margin=True, C=cfg.C)
        terms_form, terms_model, terms_F1 = final_form, cand, cand_m["F1"]
        if cand_m["F1"] >= full[winner["representative"]][1]["F1"] - cfg.delta:
            model, mode = cand, "terms"
        else:
            final_form = winner["representative"]
    out = {"runs": runs, "R": R, "classes": table, "stable": bool(stable), "winner": winner,
           "model": model, "final_mode": mode, "final_form": final_form, "stable_terms": stable_terms,
           "term_freq": term_freq, "structural": structural, "q": q, "p": p_lib, "bound": bound, "tau": tau,
           "terms_form": terms_form, "terms_model": terms_model, "terms_F1": terms_F1,
           "class_F1": full[winner["representative"]][1]["F1"]}
    if not subsample:
        u = union_select(runs, cfg.delta)
        out["union_form"], out["union_model"] = u["form"], fit(u["form"], plots, scales, G, labels, margin=True,
                                                               C=cfg.C)[0]
    return out


def node_uncertainty(model, plot, errors):
    total, k = np.zeros(plot.n), 0
    for a, t, s in model.form.node:
        width = model.G if s == "sp" else 1
        sc = model.scales[(a, t)] if model.scales is not None else 1.0
        coefs = model.theta[k:k + width] / sc
        k += width
        if a == "1" or a not in errors:
            continue
        x, e, f = plot.attrs[a], errors[a], TRANSFORMS[t]
        lo, hi = x - e, x + e
        if t in POSITIVE_ONLY:
            lo = np.maximum(lo, 1e-9)
        cand = [np.abs(f(lo) - f(x)), np.abs(f(hi) - f(x))]
        if t == "sq":
            cand.append(np.where((lo < 0) & (hi > 0), x * x, 0.0))
        c = coefs[plot.group] if s == "sp" else np.full(plot.n, coefs[0])
        total += np.abs(c) * np.maximum.reduce(cand)
    return total


def pair_uncertainty(model, plot, errors):
    total = np.zeros(plot.m)
    k = sum(model.G if s == "sp" else 1 for _, _, s in model.form.node)
    for a, op in model.form.pair:
        sc = model.scales[(a, op)] if model.scales is not None else 1.0
        c = abs(model.theta[k] / sc)
        k += 1
        if a == "species" or a not in errors:
            continue
        e = errors[a]
        if op == "absdiff":
            total += c * 2 * e
        else:
            xi, xj = plot.attrs[a][plot.i], plot.attrs[a][plot.j]
            total += c * (np.abs(xi) * e + np.abs(xj) * e + e * e)
    return total


def certified(model, plot, pos_error, errors=None):
    s = model.score(plot)
    bound = np.full(plot.m, 2.0 * pos_error)
    if errors:
        u = node_uncertainty(model, plot, errors)
        bound = bound + u[plot.i] + u[plot.j] + pair_uncertainty(model, plot, errors)
    return np.abs(s + 0.5 * model.eps) > bound


def certificate_table(model, plots, pos_errors=(0.25, 0.5, 1.0), errors=None):
    rows = []
    for e in pos_errors:
        cert = np.concatenate([certified(model, p, e, errors) for p in plots])
        y = np.concatenate([p.y for p in plots])
        rows.append({"position_error_m": e, "certified_edges": float(cert[y].mean()),
                     "certified_pairs": float(cert.mean()), "uncertified_edges": int((~cert & y).sum())})
    return rows


def _infeasible(X, d, y, rows, eps):
    rows = np.asarray(rows)
    sign = np.where(y[rows], -1.0, 1.0)
    A = X[rows] * sign[:, None]
    b = np.where(y[rows], -d[rows], d[rows] - eps)
    res = linprog(np.zeros(X.shape[1]), A_ub=A, b_ub=b, bounds=[(None, None)] * X.shape[1], method="highs")
    return res.status == 2


def max_separation(X, d, y, rows=None):
    rows = np.arange(len(d)) if rows is None else np.asarray(rows)
    Xr, dr, yr = X[rows], d[rows], y[rows]
    A = np.vstack([np.c_[-Xr[yr], np.zeros(int(yr.sum()))], np.c_[Xr[~yr], np.ones(int((~yr).sum()))]])
    b = np.r_[-dr[yr], dr[~yr]]
    res = linprog(np.r_[np.zeros(X.shape[1]), -1.0], A_ub=A, b_ub=b,
                  bounds=[(None, None)] * X.shape[1] + [(0.0, 1.0)], method="highs")
    return float(res.x[-1]) if res.status == 0 else 0.0


def conflict_set(form, plots, scales, G, eps=1e-3, limit=400):
    X, d, y = stack(plots, form, scales, G)
    theta, _, s, duals, viol = solve(X, d, y, margin=False, C=1.0, eps=eps)
    if viol <= 1e-7:
        return None
    support = np.flatnonzero(duals > 1e-9)
    minimal = support.size <= limit
    rows = list(support[np.argsort(-duals[support])])
    if minimal:
        for k in list(rows):
            trial = [r for r in rows if r != k]
            if trial and _infeasible(X, d, y, trial, eps):
                rows = trial
    own, loc = owners(plots)
    out = []
    for r in rows[:limit]:
        p = plots[own[r]]
        a, b = p.i[loc[r]], p.j[loc[r]]
        out.append({"plot": p.name, "tree_i": str(p.names[a]), "tree_j": str(p.names[b]), "edge": bool(y[r]),
                    "d": float(d[r]), "s_at_min_violation": float(s[r])})
    return {"minimal": minimal, "size": len(out), "helly_bound": X.shape[1] + 1, "pairs": out,
            "delta_set": max_separation(X, d, y, rows), "delta_all": max_separation(X, d, y)}


def fit_power(plots, G, labels, attr="dbh", lo=0.25, hi=1.25, step=0.05, C=1000.0):
    res = {}

    def one(g):
        if g not in res:
            t = f"g{int(round(100 * g)):03d}"
            form = Form.make([("1", "id", "sp"), (attr, t, "sp")])
            res[g] = fit(form, plots, make_scales(plots, Library([(attr, t)], [])), G, labels, margin=True, C=C)
        return res[g]

    for g in np.round(np.arange(lo, hi + 1e-9, step), 2):
        one(float(g))
    best = max(res, key=lambda g: (res[g][1]["F1"], -g))
    for g in np.round(np.arange(max(lo, best - step + 0.01), min(hi, best + step - 0.01) + 1e-9, 0.01), 2):
        one(float(g))
    best = max(res, key=lambda g: (res[g][1]["F1"], -g))
    return res[best][0], best, {g: res[g][1]["errors"] for g in sorted(res)}


def node_radii(model, plot):
    k = sum(model.G if s == "sp" else 1 for _, _, s in model.form.node)
    return node_columns(plot, model.form, model.scales, model.G) @ model.theta[:k]


def distance_threshold(plots):
    d = np.concatenate([p.d for p in plots])
    y = np.concatenate([p.y for p in plots])
    o = np.argsort(d)
    ys = y[o]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    return float(d[o][int(np.argmax(2 * tp / (tp + fp + y.sum())))])


def free_radii(plot, margin=False):
    m, n = plot.m, plot.n
    X = csr_matrix((np.ones(2 * m), (np.r_[np.arange(m), np.arange(m)], np.r_[plot.i, plot.j])), shape=(m, n))
    theta, t, s, _, viol = solve(X, plot.d, plot.y, margin=margin, C=1000.0, theta_lb=0.0)
    return theta, s, viol


def pair_features(plot, attrs, G):
    i, j = plot.i, plot.j
    F = [plot.d, plot.d ** 2]
    for a in attrs:
        xi, xj = plot.attrs[a][i], plot.attrs[a][j]
        F += [xi + xj, np.abs(xi - xj), xi * xj]
    for g in range(G):
        F.append((plot.group[i] == g).astype(float) + (plot.group[j] == g))
    return np.column_stack(F)


@dataclass
class Logistic:
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    thr: float
    attrs: tuple
    G: int

    def predict(self, plot):
        z = ((pair_features(plot, self.attrs, self.G) - self.mu) / self.sd) @ self.w + self.b
        return z >= self.thr


def fit_logistic(plots, attrs, G, lam=1e-4):
    F = np.vstack([pair_features(p, attrs, G) for p in plots])
    y = np.concatenate([p.y for p in plots]).astype(float)
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = (F - mu) / sd

    def f(v):
        w, b = v[:-1], v[-1]
        z = Z @ w + b
        r = expit(z) - y
        return (np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * w @ w,
                np.r_[Z.T @ r / len(y) + lam * w, r.mean()])

    v = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B").x
    z = Z @ v[:-1] + v[-1]
    o = np.argsort(-z)
    ys = y[o] > 0.5
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    thr = float(z[o][int(np.argmax(2 * tp / (tp + fp + ys.sum())))])
    return Logistic(mu, sd, v[:-1], float(v[-1]), thr, tuple(attrs), G)


TRUTHS = {
    "const": {("1", "id", "all"): 7.0},
    "dbh": {("1", "id", "all"): 3.0, ("dbh", "id", "all"): 0.12},
    "species_dbh": {("1", "id", "sp"): {"110": 2.5, "210": 3.0, "410": 3.5, "other": 3.0},
                    ("dbh", "id", "sp"): {"110": 0.10, "210": 0.11, "410": 0.13, "other": 0.12}},
    "sqrt_z": {("1", "id", "all"): 0.5, ("dbh", "sqrt", "all"): 1.1, ("z", "id", "all"): 0.3},
    "pair": {("1", "id", "all"): 3.0, ("dbh", "id", "all"): 0.12, ("dbh", "absdiff"): -0.05},
}


def truth_model(name, labels):
    spec = TRUTHS[name]
    form = Form.make([g for g in spec if len(g) == 3], [g for g in spec if len(g) == 2])
    theta = []
    for g in form.node:
        v = spec[g]
        if g[2] == "sp":
            fallback = float(np.mean(list(v.values())))
            theta.extend(v.get(lab, fallback) for lab in labels)
        else:
            theta.append(v)
    theta.extend(spec[g] for g in form.pair)
    return Model(form, np.array(theta, float), 0.0, 0.0, None, len(labels), labels)


def plant(plots, truth, rng, sigma=0.0, width=0.0, flip=0.0):
    obs, clean = [], []
    for p in plots:
        s = truth.score(p)
        yc = s >= 0
        y = yc.copy()
        if width > 0:
            y ^= rng.random(p.m) < 0.5 * np.exp(-np.abs(s) / width)
        if flip > 0:
            y ^= rng.random(p.m) < flip
        o = replace(p, y=y)
        if sigma > 0:
            o = moved(o, p.xy + rng.normal(0.0, sigma, p.xy.shape))
        obs.append(o)
        clean.append(replace(p, y=yc))
    return obs, clean


def export_pairs(plots, path, attrs=NUMERIC):
    frames = []
    for p in plots:
        df = pd.DataFrame({"plot": p.name, "tree_i": p.names[p.i], "tree_j": p.names[p.j],
                           "y": p.y.astype(int), "d": p.d})
        for a in attrs:
            df[f"{a}_i"] = p.attrs[a][p.i]
            df[f"{a}_j"] = p.attrs[a][p.j]
        df["species_i"], df["species_j"] = p.species[p.i], p.species[p.j]
        df["group_i"], df["group_j"] = p.group[p.i], p.group[p.j]
        frames.append(df)
    pd.concat(frames).to_csv(path, index=False)
