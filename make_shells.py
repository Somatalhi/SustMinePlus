"""Step 1: blocks_small_30.csv -> shells.pkl + shell_of_block.npy
shell_of_block[i] = index (1..44) of the first nested pit containing block i,
or 0 if the block lies outside the RF 1.0 pit. Used by make_bench_phases.py."""
import pickle, numpy as np, pandas as pd
from params import P, COG
from sustmine_search.blocks import load_grid
from sustmine_search.pit import ultimate_pit, precedence, OZ_PER_G

df = pd.read_csv("blocks_small_30.csv")
M = load_grid(df, d=30.0)

rfs = list(np.round(np.concatenate([np.arange(0.06, 0.91, 0.02), [1.00]]), 3))
prec = precedence(M, P["slope"])
n = len(M["tonnes"])
shell_of_block = np.zeros(n, dtype=int)
prev = np.zeros(n, bool)
shells = []
for k, rf in enumerate(rfs, start=1):
    in_pit = ultimate_pit(M, rf, P, prec)
    inc = in_pit & ~prev
    shell_of_block[inc] = k
    ore = inc & (M["grade"] >= COG)
    surf = np.unique(np.c_[M["ix"][in_pit], M["iy"][in_pit]], axis=0)
    shells.append(dict(rf=rf,
                       ore_t=M["tonnes"][ore].sum() / 1e6,
                       waste_t=M["tonnes"][inc & ~ore].sum() / 1e6,
                       moz=(M["tonnes"][ore] * M["grade"][ore]).sum() * OZ_PER_G / 1e6,
                       area_ha=len(surf) * M["d"] ** 2 / 1e4))
    prev = in_pit
    print(f"shell {k:2d} RF {rf:.2f}: +{shells[-1]['ore_t']:6.1f} Mt ore, +{shells[-1]['waste_t']:6.1f} Mt waste")

pickle.dump(shells, open("shells.pkl", "wb"))
np.save("shell_of_block.npy", shell_of_block)
inside = shell_of_block > 0
print(f"\n{len(shells)} shells written to shells.pkl")
print(f"{inside.sum():,} of {n:,} blocks inside the RF 1.0 pit ({M['tonnes'][inside].sum()/1e6:.0f} Mt); shell_of_block.npy written")
