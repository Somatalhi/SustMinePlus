import pandas as pd, numpy as np, matplotlib.pyplot as plt

T = pd.read_csv("results/exp1_cpit_5000/nsga3_sustmine_front1.csv")
f1 = T.front == 1
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(T.loc[~f1, "NPV_$M"] / 1e3, T.loc[~f1, "life_yr"], s=8, color="#bbbbbb", alpha=.5,
           label=f"other nondominated (n = {(~f1).sum()})", zorder=1)
sc = ax.scatter(T.loc[f1, "NPV_$M"] / 1e3, T.loc[f1, "life_yr"], s=60, c=T.loc[f1, "waste_Mt"],
                cmap="viridis", edgecolor="k", lw=.5, label=f"SustMine Front 1 (n = {f1.sum()})", zorder=2)
cb = fig.colorbar(sc, ax=ax, pad=.02); cb.set_label("Waste rock (Mt)")
for lab, xy in [("A: large pit, long life", (15.9, 50.5)),
                ("B: small pit, fast", (15.9, 9.5)),
                ("C: mid pit, small plant", (15.9, 28.0)),
                ("D: small pit, min. plant", (8.8, 45.0))]:
    ax.annotate(lab, xy, fontsize=8, ha="center")
ax.set_xlabel("NPV ($B)"); ax.set_ylabel("Mine life (yr)")
ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("results/fig_front1_scatter_cpit.pdf"); fig.savefig("results/fig_front1_scatter_cpit.png", dpi=300)
print("saved results/fig_front1_scatter_cpit.pdf / .png")