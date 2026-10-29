"""Experiment 1 with the MIP scheduler. One invocation = one run.

  python run_exp1_mip.py --algo nsga2 --seed 0 --budget 500
  python run_exp1_mip.py --algo nsga3 --seed 0 --budget 500
  python run_exp1_mip.py --algo lhs   --seed 0 --budget 500

Writes results/exp1_cpit_<budget>/<algo>_seed<seed>.npz with X, F, G and the per-evaluation
solver diagnostics gap, sec, nsolve, incumb, T (see problem_mip.DIAG_KEYS). X holds the
designs as evaluated, i.e. projected onto the design space Omega (OmegaRepair for the
evolutionary routes, evaluate_mip.project for the LHS sample).
Evaluations run in parallel over --workers processes (default: all cores but one).
"""
import argparse, os, pickle, time, multiprocessing as mp
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.parallelization import StarmapParallelization
from params import P
from sustmine_search.problem_mip import SustMineMIP, OmegaRepair, DIAG_KEYS
from sustmine_search.evaluate_mip import project

POP_II   = 100
REF_DIRS = get_reference_directions("das-dennis", 5, n_partitions=5)   # 126


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["nsga2", "nsga3", "lhs"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--time_limit", type=int, default=60)
    ap.add_argument("--retry_time_limit", type=int, default=300)
    ap.add_argument("--retry_gap", type=float, default=0.05)
    a = ap.parse_args()

    out_dir = f"results/exp1_cpit_{a.budget}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{a.algo}_seed{a.seed}.npz"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping"); return

    bp = pickle.load(open("bench_phases.pkl", "rb"))
    pool = mp.Pool(a.workers)
    prob = SustMineMIP(bp, P, time_limit=a.time_limit, retry_time_limit=a.retry_time_limit,
                       retry_gap=a.retry_gap,
                       elementwise_runner=StarmapParallelization(pool.starmap))
    rep = OmegaRepair(bp, P)
    t0 = time.time()
    if a.algo == "nsga2":
        res = minimize(prob, NSGA2(pop_size=POP_II, repair=rep), ("n_evals", a.budget), seed=a.seed, verbose=True)
        X, F, G = res.pop.get("X"), res.pop.get("F"), res.pop.get("G")
        diag = {k: res.pop.get(k) for k in DIAG_KEYS}
    elif a.algo == "nsga3":
        res = minimize(prob, NSGA3(pop_size=len(REF_DIRS), ref_dirs=REF_DIRS, repair=rep),
                       ("n_evals", a.budget), seed=a.seed, verbose=True)
        X, F, G = res.pop.get("X"), res.pop.get("F"), res.pop.get("G")
        diag = {k: res.pop.get(k) for k in DIAG_KEYS}
    else:
        X = LHS().do(prob, a.budget, seed=a.seed).get("X")
        X = np.array([project(x, bp, P) for x in X])              # onto Omega, as the EAs
        d = prob.evaluate(X, return_values_of=["F", "G"] + DIAG_KEYS, return_as_dictionary=True)
        F, G = d["F"], d["G"]
        diag = {k: d[k] for k in DIAG_KEYS}
    pool.close(); pool.join()

    np.savez(out_path, X=X, F=F, G=G, **diag)
    feas = (G <= 0).all(axis=1).sum()
    gap = np.asarray(diag["gap"], float)
    print(f"solver: {int((np.asarray(diag['nsolve'])==2).sum())} retried, "
          f"{int((np.asarray(diag['incumb'])==0).sum())} without incumbent, "
          f"{int((gap > 0.05).sum())} with final gap > 5%, median sec {np.median(np.asarray(diag['sec'], float)):.1f}")
    print(f"{a.algo} seed {a.seed} budget {a.budget}: {len(F)} saved, {feas} feasible, "
          f"{(time.time()-t0)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
