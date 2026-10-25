"""Step 2: blocks_small.csv + shell_of_block.npy -> bench_phases.pkl
A bench-phase is one bench within one pushback. Only blocks
inside the ultimate pit are included. Pushbacks group whole shells so that each
holds roughly equal rock tonnage."""
import pickle, numpy as np, pandas as pd
from params import COG

N_PB     = 20
BLOCK_HA = 30.0 ** 2 / 1e4

df = pd.read_csv("blocks_small_30.csv")
df = df[df.tonnes > 0].reset_index(drop=True)
shell = np.load("shell_of_block.npy")
assert len(shell) == len(df), "rerun make_shells.py"
df["shell"] = shell
df = df[df.shell > 0].copy()
n_shells = int(df.shell.max())

# pushbacks: equal-tonnage groups of whole shells, numbered 1..K with no gaps
rock = df.groupby("shell").tonnes.sum().reindex(range(1, n_shells + 1), fill_value=0.0)
cum  = (rock.cumsum() / rock.sum()).values
raw  = np.maximum.accumulate(np.minimum(np.ceil(cum * N_PB), N_PB).astype(int))
pb_of_shell = np.unique(raw, return_inverse=True)[1] + 1
df["pb"] = pb_of_shell[df.shell.values - 1]

df["bench"] = df["iz"] + 1
df["ore"]   = np.where(df["grade"] >= COG, df["tonnes"], 0.0)
df["waste"] = np.where(df["grade"] <  COG, df["tonnes"], 0.0)
df["metal"] = df["ore"] * df["grade"]
bp = (df.groupby(["pb", "bench"])
        .agg(ore_t=("ore", "sum"), waste_t=("waste", "sum"), metal_g=("metal", "sum"))
        .reset_index())
for c in ["ore_t", "waste_t", "metal_g"]:
    bp[c] /= 1e6
foot = df.drop_duplicates(["pb", "ix", "iy"]).groupby("pb").size().rename("cells").reset_index()
bp = bp.merge(foot, on="pb", how="left")
bp["area_ha"] = bp["cells"] * BLOCK_HA
bp = bp.drop(columns=["cells"]).sort_values(["pb", "bench"]).reset_index(drop=True)
last_shell = {k: int(np.where(pb_of_shell == k)[0].max() + 1) for k in range(1, bp.pb.max() + 1)}
bp["shell_hi"] = bp.pb.map(last_shell)

pickle.dump(bp, open("bench_phases.pkl", "wb"))
pbs = sorted(bp.pb.unique())
print(f"{len(bp)} bench-phases, {len(pbs)} pushbacks numbered {pbs[0]}..{pbs[-1]}"
      f" ({'no gaps' if pbs == list(range(1, len(pbs) + 1)) else 'GAPS - problem'})")
print(f"largest unit: {(bp.ore_t + bp.waste_t).max():.1f} Mt rock, {bp.ore_t.max():.1f} Mt ore")