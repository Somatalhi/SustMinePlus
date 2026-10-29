"""gap_study.py -- how much does solver time buy on the hard designs? (designs are projected onto Omega first)

Takes the designs that needed a retry in a stored run (nsolve > 1 in the npz), solves each
one at fixed time caps with the retry disabled, and records gap, NPV, time and T per cap.

Usage (from the py folder, in the venv):
    python gap_study.py --in nrp\\nsga2_seed0.npz --caps 60 120 300 900 --workers 27 --out gap_study.csv

Output: one row per (design, cap) with  id, capex, rate, thr, pb, T, cap, seconds, gap, npv,
life, status, highs_status.  The summary at the end compares NPV at each cap with NPV at the
longest cap (the reference) and reports the gap distribution.
"""
import argparse, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool

from params import P
from sustmine_search.evaluate_mip import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--bp", default="bench_phases.pkl")
ap.add_argument("--caps", type=int, nargs="+", default=[60, 120, 300, 900])
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--gap", type=float, default=0.02)
ap.add_argument("--all", action="store_true", help="use all designs, not only the retried ones")
ap.add_argument("--out", default="gap_study.csv")
a = ap.parse_args()

d = np.load(a.inp)
X = d["X"]
sel = np.arange(len(X)) if a.all else np.where(d["nsolve"] > 1)[0]
print(f"{len(sel)} designs x {len(a.caps)} caps = {len(sel)*len(a.caps)} solves, {a.workers} workers")

bp = pd.read_pickle(a.bp)


def one(task):
    i, cap = task
    x = X[i]
    t0 = time.time()
    F, G, info = evaluate(x, bp, P, time_limit=cap, gap=a.gap, retry_time_limit=0)
    xe = info["x_eval"]
    return dict(id=int(i), capex=float(x[0]), rate=float(x[1]), thr=float(x[2]), pb=int(round(x[3])),
                pb_eval=int(xe[3]), projected=int(info["projected"]), T=info["T"], cap=cap, seconds=round(time.time() - t0, 1), gap=info["mip_gap"],
                npv=info["npv"], life=info["job_security"], status=info["status"],
                highs_status=info["highs_status"], incumbent=info["incumbent"])


if __name__ == "__main__":
    # longest caps first so the slow solves do not all land at the end
    tasks = sorted(itertools.product(sel.tolist(), a.caps), key=lambda t: -t[1])
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(one, tasks), 1):
            rows.append(r)
            if k % 10 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} done, {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows).sort_values(["id", "cap"])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({(time.time()-t0)/60:.1f} min)")
    pr = df[df["projected"] == 1].drop_duplicates("id")
    if len(pr):
        print(f"{len(pr)} designs were projected onto Omega before evaluation "
              f"(pit reduced to k_max): " + ", ".join(f"id {r.id}: pit {r.pb}->{r.pb_eval}" for r in pr.itertuples()))

    # ---- summary: NPV shortfall vs the longest cap, gap distribution -----------------
    ref_cap = max(a.caps)
    ref = df[df["cap"] == ref_cap].set_index("id")["npv"]
    df["npv_short_pct"] = 100 * (ref.loc[df["id"]].values - df["npv"].values) / np.abs(ref.loc[df["id"]].values)
    g = df.groupby("cap")
    out = pd.DataFrame({
        "n": g.size(),
        "median_gap_%": 100 * g["gap"].median(),
        "p90_gap_%": 100 * g["gap"].quantile(0.9),
        "gap>5%_count": g["gap"].apply(lambda s: int((s > 0.05).sum())),
        "median_npv_short_%": g["npv_short_pct"].median(),
        "max_npv_short_%": g["npv_short_pct"].max(),
        "median_sec": g["seconds"].median(),
    }).round(2)
    print("\nNPV shortfall is relative to the NPV found with the %d s cap:" % ref_cap)
    print(out.to_string())
