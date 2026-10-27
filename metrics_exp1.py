"""Experiment 1 metrics.
Reads <--in>/*.npz (default results/exp1_mip_500), builds the reference front from the pooled feasible
nondominated solutions of all runs, normalises objectives to [0,1] by the
pooled ideal/nadir, and computes HV, IGD+, spacing, spread and cardinality per
run. Then Wilcoxon rank-sum with Holm correction and Vargha-Delaney A12.
Outputs: results/exp1/metrics.csv, results/exp1/stats.csv, and a printed summary.
"""
import argparse, glob, os, re, itertools, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowding
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions
from sustmine_search.ranking import sustmine_rank

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="IN", default="results/exp1_mip_500")
IN = ap.parse_args().IN
nds = NonDominatedSorting()


def load(path):
    d = np.load(path)
    feas = (d["G"] <= 0).all(axis=1)
    F = d["F"][feas]
    return F[nds.do(F, only_non_dominated_front=True)]          # feasible nondominated set


runs = {}
for f in sorted(glob.glob(f"{IN}/*_seed*.npz")):
    m = re.match(r".*/(\w+)_seed(\d+)\.npz", f.replace("\\", "/"))
    F = load(f)
    if len(F) == 0:
        print(f"skipping {os.path.basename(f)}: no feasible solutions"); continue
    runs[(m.group(1), int(m.group(2)))] = F
print(f"{len(runs)} runs loaded")

# ---- reference front and normalisation bounds (pooled over everything)
pool = np.vstack(list(runs.values()))
ref_front = pool[nds.do(pool, only_non_dominated_front=True)]
ideal, nadir = pool.min(0), pool.max(0)
norm = lambda F: (F - ideal) / (nadir - ideal)
ref_n = norm(ref_front)
hv = HV(ref_point=np.full(5, 1.1))
igdp = IGDPlus(ref_n)
print(f"reference front: {len(ref_front)} solutions; HV reference point 1.1 in normalised space")


def spacing(Fn):
    if len(Fn) < 2: return np.nan
    D = np.abs(Fn[:, None, :] - Fn[None, :, :]).sum(-1)
    np.fill_diagonal(D, np.inf)
    d = D.min(1)
    return d.std()


def spread(Fn):
    return np.linalg.norm(Fn.max(0) - Fn.min(0))


REF_DIRS = get_reference_directions("das-dennis", 5, n_partitions=5)
def truncate(Fn, k, how):
    """Keep k points using an algorithm's own survival operator (matched cardinality)."""
    if len(Fn) <= k: return Fn
    pop = Population.new(F=Fn)
    surv = RankAndCrowding() if how == "nsga2" else ReferenceDirectionSurvival(REF_DIRS)
    surv.filter_infeasible = False              # F only; feasibility already filtered
    return surv.do(_DUMMY, pop, n_survive=k).get("F")


class _Dummy:                                   # survival operators only ask this one question
    def has_constraints(self): return False
_DUMMY = _Dummy()


def coverage(A, B):
    """C(A,B): fraction of B dominated by at least one point of A (minimisation)."""
    dom = (A[:, None, :] <= B[None, :, :]).all(-1) & (A[:, None, :] < B[None, :, :]).any(-1)
    return dom.any(0).mean()


rows = []
matched = {}                                   # (seed, tag) -> truncated LHS set, for coverage
for (treat, seed), F in runs.items():
    Fn = norm(F)
    rows.append(dict(treatment=treat, seed=seed, n=len(F), HV=hv(Fn), IGDp=igdp(Fn),
                     spacing=spacing(Fn), spread=spread(Fn)))
    if treat == "lhs":
        # SustMine paradigm: hierarchical Pareto ranking of the sample, keep Front 1
        # (ranking runs on raw F; dims: eco=[NPV], env=[waste, land], soc=[employees, job security])
        _, _, _, comp = sustmine_rank(F, [], [0], [1, 2], [3, 4])
        sm = Fn[comp == 1]
        matched[(seed, "sustmine")] = sm
        rows.append(dict(treatment="sustmine", seed=seed, n=len(sm), HV=hv(sm), IGDp=igdp(sm),
                         spacing=spacing(sm), spread=spread(sm)))
        # matched-cardinality controls: the sample truncated by each algorithm's own survival
        for k, how, tag in [(100, "nsga2", "lhs100"), (126, "nsga3", "lhs126")]:
            sub = truncate(Fn, k, how)
            matched[(seed, tag)] = sub
            rows.append(dict(treatment=tag, seed=seed, n=len(sub), HV=hv(sub), IGDp=igdp(sub),
                             spacing=spacing(sub), spread=spread(sub)))
M = pd.DataFrame(rows)
M.to_csv(f"{IN}/metrics.csv", index=False)

# ---- summary: median [IQR]
print("\nmedian [IQR] over seeds")
summ = M.groupby("treatment").agg(**{c: (c, lambda x: f"{np.median(x):.4g} [{np.percentile(x,25):.4g}, {np.percentile(x,75):.4g}]")
                                    for c in ["HV", "IGDp", "spacing", "spread", "n"]})
print(summ.to_string())

# ---- pairwise Wilcoxon rank-sum (Mann-Whitney U), Holm-corrected, with A12
def a12(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))

stats = []
treats = sorted(M.treatment.unique())
for metric in ["HV", "IGDp"]:
    pairs = list(itertools.combinations(treats, 2))
    ps = []
    for a, b in pairs:
        xa, xb = M[M.treatment == a][metric], M[M.treatment == b][metric]
        ps.append(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
    order = np.argsort(ps); holm = np.empty(len(ps))
    for rank, i in enumerate(order):
        holm[i] = min(1.0, ps[i] * (len(ps) - rank))
    for (a, b), p, ph in zip(pairs, ps, holm):
        xa, xb = M[M.treatment == a][metric], M[M.treatment == b][metric]
        stats.append(dict(metric=metric, A=a, B=b, p_raw=p, p_holm=ph, A12=a12(xa, xb),
                          better="A" if (metric == "HV") == (np.median(xa) > np.median(xb)) else "B"))
S = pd.DataFrame(stats)
S.to_csv(f"{IN}/stats.csv", index=False)
print("\npairwise tests (A12 > 0.5 means A tends to have the larger value)")
print(S.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---- set coverage C(A,B), same seed
cov = []
seeds = sorted({sd for _, sd in runs})
for seed in seeds:
    sets = {t: norm(runs[(t, seed)]) for t in treats if (t, seed) in runs}
    for tag in ["sustmine", "lhs100", "lhs126"]:
        if (seed, tag) in matched: sets[tag] = matched[(seed, tag)]
    for a, b in itertools.permutations(sets, 2):
        cov.append(dict(seed=seed, A=a, B=b, C=coverage(sets[a], sets[b])))
Cv = pd.DataFrame(cov).groupby(["A", "B"]).C.median().unstack()
Cv.to_csv(f"{IN}/coverage.csv")
print("\nset coverage C(A,B): median over seeds of the fraction of B dominated by A (rows = A)")
print(Cv.to_string(float_format=lambda v: f"{v:.2f}"))
 