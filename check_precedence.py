"""Step 2 check: build and inspect the precedence dictionary."""
import pickle
from sustmine_search.precedence import pushback_precedence

bp = pickle.load(open("bench_phases.pkl", "rb"))
preds = pushback_precedence(bp)
# covering some cases for checking: P{pushback}_{bench}
n_pred = [len(v) for v in preds.values()]
print(f"{len(preds)} bench-phases; predecessors per unit: "
      f"0 -> {n_pred.count(0)}, 1 -> {n_pred.count(1)}, 2 -> {n_pred.count(2)}")
print("\nExamples:")
for k in ["P1_2", "P1_3", "P4_14", "P2_5", "P5_32"]:
    if k in preds:
        print(f"  {k}: {preds[k]}")
