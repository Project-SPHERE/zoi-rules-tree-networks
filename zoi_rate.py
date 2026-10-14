import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import pdist
from scipy.stats import gamma, kstest


def disc_points(rng, n, R):
    r = R * np.sqrt(rng.random(n))
    a = 2 * np.pi * rng.random(n)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def disc_density(u, R):
    q = u / (2 * R)
    return 4 * u / (np.pi * R ** 2) * (np.arccos(q) - q * np.sqrt(1 - q * q))


def constant_radius(rng, ns, reps, R, u0):
    h = disc_density(u0, R)
    limit = gamma(2, scale=1 / h)
    rows = []
    for n in ns:
        w = np.empty(reps)
        for k in range(reps):
            d = pdist(disc_points(rng, n, R))
            y = d <= u0
            w[k] = (d[~y].min() - d[y].max()) / 2
        z = n * n * w
        rows.append({"n": n, "mean_n2w": z.mean(), "theory_mean": limit.mean(),
                     "median_n2w": np.median(z), "theory_median": limit.median(),
                     "ks_pvalue": kstest(z, limit.cdf).pvalue,
                     "median_width_mm": 1000 * np.median(w), "theory_width_mm": 1000 * limit.median() / n ** 2})
    return pd.DataFrame(rows), h


def linear_radius(rng, ns, reps, R, a0, b0, lo, hi):
    rows = []
    for n in ns:
        wa, wb = np.full(reps, np.nan), np.full(reps, np.nan)
        i, j = np.triu_indices(n, 1)
        for k in range(reps):
            P = disc_points(rng, n, R)
            m = rng.uniform(lo, hi, n)
            d = np.sqrt(((P[i] - P[j]) ** 2).sum(1))
            S = m[i] + m[j]
            y = 2 * a0 + b0 * S >= d
            A = np.r_[np.c_[-2 * np.ones(y.sum()), -S[y]], np.c_[2 * np.ones((~y).sum()), S[~y]]]
            b = np.r_[-d[y], d[~y] - 1e-9]
            ext = []
            for c in ([1, 0], [-1, 0], [0, 1], [0, -1]):
                res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 2, method="highs")
                ext.append(res.x if res.status == 0 else None)
            if all(e is not None for e in ext):
                wa[k] = ext[1][0] - ext[0][0]
                wb[k] = ext[3][1] - ext[2][1]
        rows.append({"n": n, "median_width_a_m": np.nanmedian(wa), "median_width_b": np.nanmedian(wb),
                     "valid_reps": int(np.sum(~np.isnan(wa)))})
    return pd.DataFrame(rows)


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def coordinates(x, y):
    return "\\addplot coordinates {" + " ".join(f"({a},{b:.6g})" for a, b in zip(x, y)) + "};"


def main():
    ap = argparse.ArgumentParser(prog="zoi_rate")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--reps2", type=int, default=50)
    ap.add_argument("--R", type=float, default=35.7)
    ap.add_argument("--u0", type=float, default=14.0)
    ap.add_argument("--a0", type=float, default=2.0)
    ap.add_argument("--b0", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join("results", "rate"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fmt = lambda v: f"{v:.4g}"

    t0 = time.time()
    ns1 = [50, 100, 200, 400, 800]
    df1, h = constant_radius(rng, ns1, a.reps, a.R, a.u0)
    print(f"\nConstant radius   R={a.R} m   u0={a.u0} m   h(u0)={h:.5f} 1/m   reps={a.reps}   time={time.time() - t0:.0f} s")
    print(df1.to_string(index=False, float_format=fmt))
    print(f"log-log slope of the median width = {slope(df1.n, df1.median_width_mm):.3f}")

    t0 = time.time()
    ns2 = [50, 100, 200, 400]
    df2 = linear_radius(rng, ns2, a.reps2, a.R, a.a0, a.b0, 10.0, 60.0)
    print(f"\nLinear radius r = a + b*m   a={a.a0}   b={a.b0}   m ~ U(10, 60)   reps={a.reps2}   time={time.time() - t0:.0f} s")
    print(df2.to_string(index=False, float_format=fmt))
    print(f"log-log slopes   a = {slope(df2.n, df2.median_width_a_m):.3f}   b = {slope(df2.n, df2.median_width_b):.3f}")

    df1.to_csv(os.path.join(a.out, "constant_radius.csv"), index=False)
    df2.to_csv(os.path.join(a.out, "linear_radius.csv"), index=False)
    with open(os.path.join(a.out, "pgf_coordinates.tex"), "w", encoding="utf-8") as f:
        f.write(coordinates(df1.n, df1.median_width_mm) + "\n")
        f.write(coordinates(df1.n, df1.theory_width_mm) + "\n")
        f.write(coordinates(df2.n, df2.median_width_a_m) + "\n")
        f.write(coordinates(df2.n, df2.median_width_b) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
