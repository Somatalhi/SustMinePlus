"""Diagnostic: does the solver time cap bias the evaluations?

Takes designs from an Experiment-1 LHS run (half 'hard': largest pit / lowest rate;
half random), re-evaluates each with the CPIT scheduler and compares the stored
objectives with the fresh ones; reports retries, missing incumbents and final gaps.

  python diag_solver.py --in results/exp1_cpit_5000/lhs_seed0.npz --n 40
Writes results/diag_solver.csv and prints a summary.
"""
import argparse, time, numpy as np, pandas as pd
from params import P
from sustmine_search import evaluate_mip as new

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="IN", default="results/exp1_cpit_5000/lhs_seed0.npz")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--time_limit", type=int, default=60)
ap.add_argument("--retry_time_limit", type=int, default=300)
ap.add_argument("--out", default="results/diag_solver.csv")
a = ap.parse_args()

bp = pd.read_pickle("bench_phases.pkl")
d = np.load(a.IN)
X, F, G = d["X"], d["F"], d["G"]
feas = (G <= 0).all(1)
Xf, Ff = X[feas], F[feas]
print(f"{a.IN}: {len(X)} designs, {feas.sum()} feasible")

# half random, half 'hard' (largest pit, lowest rate), so both regimes are covered
rng = np.random.default_rng(0)
hard_score = Xf[:, 3] / Xf[:, 3].max() - Xf[:, 1] / Xf[:, 1].max()
idx_hard = np.argsort(-hard_score)[: a.n // 2]
rest = np.setdiff1d(np.arange(len(Xf)), idx_hard)
idx_rand = rng.choice(rest, size=a.n - len(idx_hard), replace=False)
idx = np.concatenate([idx_hard, idx_rand])

rows = []
for k, i in enumerate(idx):
    x = Xf[i]
    Fn, Gn, inn = new.evaluate(x, bp, P, time_limit=a.time_limit, gap=0.02, retry_time_limit=a.retry_time_limit)
    rows.append(dict(
        kind="hard" if k < len(idx_hard) else "random",
        capex=x[0], rate=x[1], thr=x[2], pb=int(round(x[3])),
        npv_stored=-Ff[i][0], life_stored=-Ff[i][4], waste_stored=Ff[i][1],
        npv_cpit=-Fn[0], life_cpit=-Fn[4], waste_cpit=Fn[1], cpit_status=inn["highs_status"],
        cpit_gap=inn["mip_gap"], cpit_sec=inn["seconds"], T_cpit=inn["T"],
        cpit_solves=inn["n_solves"], cpit_incumbent=int(inn["incumbent"]),
    ))
    r = rows[-1]
    print(f"{k+1:3d}/{len(idx)} {r['kind']:6s} pb={r['pb']:2d} rate={r['rate']:5.1f} thr={r['thr']:5.1f} | "
          f"NPV stored {r['npv_stored']:8.0f} "
          f"cpit {r['npv_cpit']:8.0f} ({r['cpit_sec']:5.1f}s gap {r['cpit_gap']:.3f} x{r['cpit_solves']}{'' if r['cpit_incumbent'] else ' NO-INC'}) | life {r['life_stored']:.0f}/{r['life_cpit']:.0f}")

R = pd.DataFrame(rows)
R["dnpv_cpit_vs_stored_pct"] = 100 * (R.npv_cpit - R.npv_stored) / R.npv_stored.abs()
R.to_csv(a.out, index=False)

print("\n=== summary ===")
print(f"CPIT: retried {int((R.cpit_solves==2).sum())} of {len(R)}, no incumbent {int((R.cpit_incumbent==0).sum())}, "
      f"final gap > 5%: {int((R.cpit_gap > 0.05).sum())}, sec median {R.cpit_sec.median():.1f} max {R.cpit_sec.max():.0f}")
for kind, g in R.groupby("kind"):
    print(f"{kind:6s}: NPV fresh vs stored  median {g.dnpv_cpit_vs_stored_pct.median():+.2f}%  "
          f"IQR [{g.dnpv_cpit_vs_stored_pct.quantile(.25):+.2f}, {g.dnpv_cpit_vs_stored_pct.quantile(.75):+.2f}]  "
          f"max |diff| {g.dnpv_cpit_vs_stored_pct.abs().max():.2f}% | life differs in {(g.life_cpit != g.life_stored).sum()} designs | "
          f"sec median {g.cpit_sec.median():.1f}")
