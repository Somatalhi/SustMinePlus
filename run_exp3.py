"""Experiment 3: robustness under indicator uncertainty.
Each objective value is perturbed by independent multiplicative noise
eps ~ N(1, sigma^2), REPS times, for sigma in SIGMAS.
 (a) Generation: does NSGA-III's hypervolume advantage over the matched LHS
     subset survive the noise?  -> A12 and p per sigma; sigma* = first sigma
     where the difference is no longer significant.
 (b) Ranking: how stable is SustMine's hierarchical Front 1 on the NSGA-III
     population? -> per-solution Front-1 stability, number of fronts.
Outputs: results/exp3/generation.csv, ranking.csv, and a printed summary.
"""
import argparse, glob, os, re, itertools, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions
from sustmine_search.ranking import sustmine_rank

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="IN", default="results/exp1_mip_500"); ap.add_argument("--reps", type=int, default=100)
_a = ap.parse_args(); IN = _a.IN; OUT = IN.replace("exp1", "exp3")
SIGMAS = [0.05, 0.10, 0.20]
REPS = _a.reps
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
nds = NonDominatedSorting()
REF_DIRS = get_reference_directions("das-dennis", 5, n_partitions=5)


class _Dummy:
    def has_constraints(self): return False


def load_nd(path):
    d = np.load(path); feas = (d["G"] <= 0).all(1); F = d["F"][feas]
    return F[nds.do(F, only_non_dominated_front=True)]


def truncate126(F):
    surv = ReferenceDirectionSurvival(REF_DIRS); surv.filter_infeasible = False
    return surv.do(_Dummy(), Population.new(F=F), n_survive=126).get("F")


seeds = sorted({int(re.search(r"seed(\d+)", f).group(1)) for f in glob.glob(f"{IN}/nsga3_seed*.npz")})
sets = {}
for sd in seeds:
    a, b = load_nd(f"{IN}/nsga3_seed{sd}.npz"), load_nd(f"{IN}/lhs_seed{sd}.npz")
    if len(a) == 0 or len(b) == 0:
        print(f"skipping seed {sd}: no feasible solutions in one set"); continue
    sets[sd] = dict(nsga3=a, lhs126=truncate126(b))
seeds = sorted(sets)
if not seeds:
    raise SystemExit(f"no usable seeds in {IN} (need nsga3 and lhs files with feasible solutions)")
pool = np.vstack([v for s in sets.values() for v in s.values()])
ideal, nadir = pool.min(0), pool.max(0)
norm = lambda F: np.clip((F - ideal) / (nadir - ideal), 0.0, None)   # clip at the ideal: noise cannot create volume beyond it
hv = HV(ref_point=np.full(5, 1.1))
print(f"{len(seeds)} seeds, {REPS} replicates, sigma {SIGMAS}")

# ---------------- (a) generation
gen = []
for sigma in SIGMAS:
    for sd in seeds:
        h = {}
        for t, F in sets[sd].items():
            vals = [hv(norm(F * rng.normal(1.0, sigma, F.shape))) for _ in range(REPS)]
            h[t] = np.median(vals)
        gen.append(dict(sigma=sigma, seed=sd, hv_nsga3=h["nsga3"], hv_lhs126=h["lhs126"]))
G = pd.DataFrame(gen); G.to_csv(f"{OUT}/generation.csv", index=False)

def a12(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))

print("\n(a) generation: NSGA-III vs LHS-126 hypervolume under noise (medians over replicates, tests over seeds)")
print(f"{'sigma':>6s} {'HV nsga3':>10s} {'HV lhs126':>10s} {'diff':>8s} {'p':>9s} {'A12':>6s}")
for sigma in [0.0] + SIGMAS:
    if sigma == 0.0:
        a = [hv(norm(sets[sd]["nsga3"])) for sd in seeds]; b = [hv(norm(sets[sd]["lhs126"])) for sd in seeds]
    else:
        g = G[G.sigma == sigma]; a, b = g.hv_nsga3.values, g.hv_lhs126.values
    p = mannwhitneyu(a, b).pvalue
    print(f"{sigma:6.2f} {np.median(a):10.4f} {np.median(b):10.4f} {np.median(a)-np.median(b):8.4f} {p:9.2e} {a12(a,b):6.2f}")

# ---------------- (b) ranking
rk = []
for sigma in SIGMAS:
    for sd in seeds:
        F = sets[sd]["nsga3"]
        _, _, _, nominal = sustmine_rank(F, [], [0], [1, 2], [3, 4])
        f1_nom = nominal == 1
        on_f1 = np.zeros(len(F)); nfronts = []
        for _ in range(REPS):
            _, _, _, comp = sustmine_rank(F * rng.normal(1.0, sigma, F.shape), [], [0], [1, 2], [3, 4])
            on_f1 += (comp == 1); nfronts.append(comp.max())
        stab = on_f1 / REPS
        rk.append(dict(sigma=sigma, seed=sd, n=len(F), F1_nominal=int(f1_nom.sum()),
                       stab_F1_members=stab[f1_nom].mean(), stab_others=stab[~f1_nom].mean(),
                       n_stable=int((stab > 0.5).sum()), fronts=np.median(nfronts)))
R = pd.DataFrame(rk); R.to_csv(f"{OUT}/ranking.csv", index=False)

print("\n(b) ranking: hierarchical Front 1 stability on the NSGA-III population (medians over seeds)")
print(f"{'sigma':>6s} {'|F1| nominal':>13s} {'P(F1|member)':>13s} {'P(F1|other)':>12s} {'n stable>0.5':>13s} {'fronts':>7s}")
for sigma, g in R.groupby("sigma"):
    print(f"{sigma:6.2f} {g.F1_nominal.median():13.0f} {g.stab_F1_members.median():13.2f} "
          f"{g.stab_others.median():12.2f} {g.n_stable.median():13.0f} {g.fronts.median():7.0f}")
