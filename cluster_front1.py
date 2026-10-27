"""Archetype clusters and stability scores for the NSGA-III + SustMine shortlist.
1. Pool the 30 NSGA-III populations at one budget, keep the feasible nondominated subset,
   rank hierarchically, take Front 1.
2. k-means in decision space (rate, throughput, pushback; standardised) on Front 1,
   k chosen by silhouette over 2..5.
3. Perturb objectives (multiplicative N(1, sigma^2), R replicates), re-rank the pooled set,
   and record for each nominal Front-1 design the fraction of replicates in which it stays
   on Front 1 (its stability score) and, for each cluster, the fraction of replicates in
   which at least one of its members is on Front 1 (archetype stability).
Usage: python cluster_front1.py --in results/exp1_mip_5000 --sigma 0.05 --reps 100"""
import argparse, glob, numpy as np, pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from sustmine_search.ranking import pareto_rank, sustmine_rank

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="IN", default="results/exp1_mip_5000")
ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--reps", type=int, default=100)
a = ap.parse_args()
rng = np.random.default_rng(0)
ECO, ENV, SOC = [0], [1, 2], [3, 4]

X, F = [], []
for f in sorted(glob.glob(f"{a.IN}/nsga3_seed*.npz")):
    d = np.load(f); feas = (d["G"] <= 0).all(axis=1)
    X.append(d["X"][feas]); F.append(d["F"][feas])
X, F = np.vstack(X), np.vstack(F)
nd = pareto_rank(F) == 1; X, F = X[nd], F[nd]
XF = np.unique(np.round(np.hstack([X, F]), 1), axis=0); X, F = XF[:, :4], XF[:, 4:]
*_, hier = sustmine_rank(F, max_cols=[], eco_cols=ECO, env_cols=ENV, soc_cols=SOC, gvf_thresh=0.80)
f1 = np.where(hier == 1)[0]
print(f"{len(F)} nondominated designs, Front 1 = {len(f1)}")

# --- k-means in decision space on Front 1 ---------------------------------
D = X[f1][:, [2, 3]]                                      # throughput, pushback
Z = (D - D.mean(0)) / D.std(0)
def silhouette(Z, lab):
    dist = cdist(Z, Z); s = []
    for i in range(len(Z)):
        same = lab == lab[i]; same[i] = False
        if same.sum() == 0: s.append(0); continue
        ai = dist[i, same].mean()
        bi = min(dist[i, lab == k].mean() for k in set(lab) if k != lab[i])
        s.append((bi - ai) / max(ai, bi))
    return np.mean(s)
best = None
for k in range(2, 6):
    cen, lab = kmeans2(Z, k, minit="++", seed=0)
    sil = silhouette(Z, lab)
    print(f"k = {k}: silhouette = {sil:.3f}")
    if best is None or sil > best[0]: best = (sil, k, lab)
sil, k, lab = best
print(f"chosen k = {k}")

# --- noise replicates ------------------------------------------------------
stay = np.zeros(len(f1)); arche = np.zeros(k)
for r in range(a.reps):
    Fn = F * rng.normal(1, a.sigma, F.shape)
    *_, h = sustmine_rank(Fn, max_cols=[], eco_cols=ECO, env_cols=ENV, soc_cols=SOC, gvf_thresh=0.80)
    on = h[f1] == 1
    stay += on
    for c in range(k): arche[c] += on[lab == c].any()
stay /= a.reps; arche /= a.reps

T = pd.DataFrame({"cluster": lab, "capex": X[f1, 0], "rate": X[f1, 1], "thr": X[f1, 2],
                  "pushback": np.round(X[f1, 3]).astype(int), "NPV_$M": -F[f1, 0], "waste_Mt": F[f1, 1],
                  "land_ha.yr": F[f1, 2], "employees": -F[f1, 3], "life_yr": -F[f1, 4], "stability": stay})
pd.set_option("display.width", 200)
print(f"\nper-cluster summary (sigma = {a.sigma}, {a.reps} replicates)")
S = T.groupby("cluster").agg(n=("stability", "size"), pushback=("pushback", lambda s: f"{s.min()}-{s.max()}"),
                             thr=("thr", lambda s: f"{s.min():.1f}-{s.max():.1f}"),
                             life=("life_yr", lambda s: f"{s.min():.0f}-{s.max():.0f}"),
                             NPV_B=("NPV_$M", lambda s: f"{s.min()/1e3:.1f}-{s.max()/1e3:.1f}"),
                             waste=("waste_Mt", lambda s: f"{s.min():.0f}-{s.max():.0f}"),
                             mean_stability=("stability", "mean"), max_stability=("stability", "max"))
S["archetype_stability"] = arche
print(S.round(2).to_string())
print(f"\noverall: median member stability {np.median(stay):.2f}, "
      f"members stable > 0.5: {(stay > .5).sum()} of {len(f1)}")
T.to_csv(f"{a.IN}/front1_clusters_sigma{a.sigma}.csv", index=False)
