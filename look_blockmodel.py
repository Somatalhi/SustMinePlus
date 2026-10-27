"""Quick look at a reblocked model: pit at RF 1.0, plan views and one section.

  python look_blockmodel.py blocks_small.csv 30        # file, block size in m
Prints inventory and pit statistics; writes blockmodel_<d>m.png.
"""
import sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import P, COG
from sustmine_search.blocks import load_grid
from sustmine_search.pit import ultimate_pit, precedence, block_value, OZ_PER_G

path = sys.argv[1] if len(sys.argv) > 1 else "blocks_small_30.csv"
d    = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

# 1. load
M = load_grid(pd.read_csv(path), d=d)
ore = M["grade"] >= COG
moz = lambda m: (M["tonnes"][m] * M["grade"][m]).sum() * OZ_PER_G / 1e6
print(f"COG {COG:.3f} g/t; in-situ ore {M['tonnes'][ore].sum()/1e6:.0f} Mt "
      f"at {np.average(M['grade'][ore], weights=M['tonnes'][ore]):.3f} g/t = {moz(ore):.2f} Moz")

# 2. ultimate pit at revenue factor 1.0 (same calls as make_shells.py)
t0 = time.time(); prec = precedence(M, P["slope"])
print(f"precedence arcs {len(prec[0]):,} in {time.time()-t0:.1f}s")
t0 = time.time(); pit = ultimate_pit(M, 1.0, P, prec)
print(f"RF1.0 pit in {time.time()-t0:.1f}s: {pit.sum():,} blocks, "
      f"{M['tonnes'][pit].sum()/1e6:.0f} Mt rock, {M['tonnes'][pit & ore].sum()/1e6:.0f} Mt ore, "
      f"{moz(pit & ore):.2f} Moz, SR {M['tonnes'][pit & ~ore].sum()/M['tonnes'][pit & ore].sum():.2f}, "
      f"value ${block_value(M, 1.0, P)[pit].sum()/1e6:,.0f}M undiscounted, "
      f"depth {M['iz'][pit].max()+1} benches")

# 3. arrays on the 3-D grid
nx, ny, nz = M["nx"], M["ny"], M["nz"]
G  = np.full((nx, ny, nz), np.nan); G[M["ix"], M["iy"], M["iz"]] = M["grade"]
Pm = np.zeros((nx, ny, nz), bool);  Pm[M["ix"][pit], M["iy"][pit], M["iz"][pit]] = True

# 4. figures
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
im = ax[0].imshow(np.nanmax(G, axis=2).T, origin="lower", cmap="viridis", vmin=0, vmax=1)
ax[0].set_title("Plan: max grade in column (g/t)"); plt.colorbar(im, ax=ax[0])
depth = np.where(Pm.any(2), Pm.sum(2), np.nan)
im = ax[1].imshow(depth.T, origin="lower", cmap="magma_r")
ax[1].set_title(f"Plan: RF 1.0 pit depth (benches of {d:.0f} m)"); plt.colorbar(im, ax=ax[1])
j = int(np.argmax(np.nansum(np.where(G >= COG, 1, 0), axis=(0, 2))))   # row with most ore blocks
ax[2].imshow(G[:, j, :].T, origin="upper", cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax[2].contour(Pm[:, j, :].T.astype(float), levels=[0.5], colors="r")
ax[2].set_title(f"E-W section iy={j}: grade, red = RF 1.0 pit")
ax[2].set_xlabel(f"ix ({d:.0f} m)"); ax[2].set_ylabel("bench (0 = top)")
plt.tight_layout(); plt.savefig(f"blockmodel_{d:.0f}m.png", dpi=130)
print(f"wrote blockmodel_{d:.0f}m.png")
