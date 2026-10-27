"""Experiment 2: ranking.
For each saved population, rank the feasible solutions two ways on the five
objectives: flat nondominated sorting (what the search uses) and SustMine's
hierarchical composite (Jenks classification -> per-dimension Pareto rank ->
composite Pareto rank). Objectives are already minimised in F, so no sign flip.
Dimensions: economic = NPV; environmental = waste, land; social = employees, job security.
Outputs: results/exp2/ranking.csv (per run) and a printed summary.
"""
import argparse, glob, os, re, numpy as np, pandas as pd
from sustmine_search.ranking import pareto_rank, sustmine_rank, shannon

ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="IN", default="results/exp1_mip_500")
IN  = ap.parse_args().IN
OUT = IN.replace("exp1", "exp2")
GVF = 0.80
os.makedirs(OUT, exist_ok=True)

ECO, ENV, SOC = [0], [1, 2], [3, 4]
NAMES = ["NPV", "waste", "land", "employees", "job security"]

rows = []
for f in sorted(glob.glob(f"{IN}/*_seed*.npz")):
    m = re.match(r".*/(\w+)_seed(\d+)\.npz", f.replace("\\", "/"))
    treat, seed = m.group(1), int(m.group(2))
    #if treat == "lhs":
        #continue                                            # 3,000-point sets: rank the algorithm populations
    d = np.load(f)
    feas = (d["G"] <= 0).all(axis=1)
    F = d["F"][feas]
    n = len(F)

    flat = pareto_rank(F)
    eco, env, soc, hier = sustmine_rank(F, max_cols=[], eco_cols=ECO, env_cols=ENV, soc_cols=SOC, gvf_thresh=GVF)

    f1_flat, f1_hier = set(np.where(flat == 1)[0]), set(np.where(hier == 1)[0])
    jacc = len(f1_flat & f1_hier) / max(len(f1_flat | f1_hier), 1)
    # dimension balance of the hierarchical Front 1: how many of its members are rank 1 in each dimension
    h1 = np.array(sorted(f1_hier))
    bal = [(eco[h1] == 1).mean(), (env[h1] == 1).mean(), (soc[h1] == 1).mean()] if len(h1) else [np.nan] * 3

    rows.append(dict(treatment=treat, seed=seed, n=n,
                     flat_fronts=flat.max(), flat_F1=len(f1_flat), flat_H=shannon(flat),
                     hier_fronts=hier.max(), hier_F1=len(f1_hier), hier_H=shannon(hier),
                     F1_jaccard=jacc,
                     F1_dropped=len(f1_flat - f1_hier),      # flat-nondominated but not hierarchical Front 1
                     F1_added=len(f1_hier - f1_flat),        # hierarchical Front 1 but flat-dominated
                     bal_eco=bal[0], bal_env=bal[1], bal_soc=bal[2]))

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/ranking.csv", index=False)

med = lambda x: f"{np.median(x):.3g} [{np.percentile(x,25):.3g}, {np.percentile(x,75):.3g}]"
print("median [IQR] over seeds\n")
for treat, g in R.groupby("treatment"):
    print(f"--- {treat}  (n = {int(g.n.median())} feasible per run)")
    print(f"  flat:          {med(g.flat_fronts):>18s} fronts   Front 1 = {med(g.flat_F1):>16s}   H = {med(g.flat_H)}")
    print(f"  hierarchical:  {med(g.hier_fronts):>18s} fronts   Front 1 = {med(g.hier_F1):>16s}   H = {med(g.hier_H)}")
    print(f"  Front 1 overlap (Jaccard) {med(g.F1_jaccard)};  dropped by hierarchy {med(g.F1_dropped)};  added {med(g.F1_added)}")
    print(f"  hierarchical Front 1 members that are rank 1 in eco / env / soc: "
          f"{np.median(g.bal_eco):.2f} / {np.median(g.bal_env):.2f} / {np.median(g.bal_soc):.2f}\n")
