"""Budget curve: HV and IGD+ vs evaluation budget, median with IQR band over 30 seeds.
Reads results/exp1_mip_<budget>/metrics.csv written by metrics_exp1.py."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

BUDGETS = [500, 1000, 2000, 5000]
SHOW = {  # treatment -> (label, colour, linestyle)
    "nsga3":    ("NSGA-III",            "#1b6ca8", "-"),
    "nsga2":    ("NSGA-II",             "#7fb3d5", "-"),
    "lhs":      ("LHS sample (all)",    "#c0392b", "--"),
    "lhs126":   ("LHS sample (126)",    "#e59866", "--"),
    "sustmine": ("SustMine Front 1",    "#1e8449", ":"),
}

M = pd.concat([pd.read_csv(f"results/exp1_cpit_{b}/metrics.csv").assign(budget=b) for b in BUDGETS])

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for ax, metric, ylab in zip(axes, ["HV", "IGDp"], ["Hypervolume (higher is better)", "IGD$^+$ (lower is better)"]):
    for t, (lab, col, ls) in SHOW.items():
        g = M[M.treatment == t].groupby("budget")[metric]
        med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(med.index, q1, q3, color=col, alpha=.15, lw=0)
        ax.plot(med.index, med, color=col, ls=ls, marker="o", ms=4, lw=1.8, label=lab)
    ax.set_xscale("log");
    ax.set_xticks(BUDGETS);
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Evaluation budget"); ax.set_ylabel(ylab)
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("results/fig_budget_curve_cpit.pdf"); fig.savefig("results/fig_budget_curve_cpit.png", dpi=300)
print("saved results/fig_budget_curve_cpit.pdf / .png")