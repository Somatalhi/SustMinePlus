"""Step 5: short NSGA-II run with the MIP scheduler, evaluations in parallel."""
import pickle, time, multiprocessing as mp
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.parallelization import StarmapParallelization
from pymoo.optimize import minimize
from params import P
from sustmine_search.problem_mip import SustMineMIP

POP, GENS, WORKERS = 12, 3, max(1, mp.cpu_count() - 1)

if __name__ == "__main__":
    bp = pickle.load(open("bench_phases.pkl", "rb"))
    pool = mp.Pool(WORKERS)
    prob = SustMineMIP(bp, P, time_limit=60, elementwise_runner=StarmapParallelization(pool.starmap))
    print(f"{POP*GENS} evaluations on {WORKERS} workers ...")
    t0 = time.time()
    res = minimize(prob, NSGA2(pop_size=POP), ("n_gen", GENS), seed=1, verbose=True)
    pool.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(res.F)} nondominated")
    if res.F is not None and len(res.F):
        F = res.F.copy(); F[:, [0, 3, 4]] *= -1
        print("NPV $M, waste Mt, land ha.yr, employees, job security yr")
        print(np.round(F, 0))
        print("decision vectors [capex, rate, thr, pushback]:")
        print(np.round(res.X, 1))
    gap = res.pop.get("gap"); nsolve = res.pop.get("nsolve"); incumb = res.pop.get("incumb"); sec = res.pop.get("sec")
    print(f"solver: {int((nsolve==2).sum())} retried, {int((incumb==0).sum())} without incumbent, "
          f"{int((gap>0.05).sum())} with gap > 5%, median {np.median(sec):.1f}s, max {sec.max():.0f}s")