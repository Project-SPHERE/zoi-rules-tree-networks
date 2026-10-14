import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def execute(cmd, out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(out, "log.txt"), "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t0


def report(out, code, seconds):
    print(f"{out}: {'ok' if code == 0 else f'ERROR {code}, see log.txt'}  {seconds:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="zoi_repeat")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("results", "transfer_seeds"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--what", default="plot-stability,type-stability,type-single")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--attrs", default="dbh,soc,z")
    ap.add_argument("--transforms", default="id,sq,sqrt,log,p033,p067,p075")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--evals", type=int, default=300)
    ap.add_argument("--delta", type=float, default=0.001)
    ap.add_argument("--pi", type=float, default=0.6)
    a = ap.parse_args()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zoi_run.py")
    stab, single = [], []
    for item in a.what.split(","):
        by, mode = item.strip().split("-")
        if by not in ("plot", "type") or mode not in ("stability", "single"):
            raise SystemExit(f"unknown item {item}")
        for r in range(a.seeds):
            seed = a.first_seed + a.step * r
            out = os.path.join(a.out, f"{by}_{mode}_seed{seed}")
            if os.path.exists(os.path.join(out, f"transfer_{by}_{mode}.csv")):
                print(f"{out}: already done", flush=True)
                continue
            cmd = [sys.executable, script, "transfer", "--data", a.data, "--by", by, "--seed", str(seed), "--out", out,
                   "--attrs", a.attrs, "--transforms", a.transforms, "--runs", str(a.runs), "--evals", str(a.evals),
                   "--delta", str(a.delta), "--pi", str(a.pi)]
            if mode == "single":
                single.append((cmd + ["--single"], out))
            else:
                stab.append((cmd + ["--jobs", str(a.jobs)], out))
    print(f"stability runs {len(stab)} in sequence, single runs {len(single)} with {a.parallel} in parallel", flush=True)
    t0 = time.time()
    for cmd, out in stab:
        report(out, *execute(cmd, out))
    if single:
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futures = {ex.submit(execute, cmd, out): out for cmd, out in single}
            for f in as_completed(futures):
                report(futures[f], *f.result())
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
