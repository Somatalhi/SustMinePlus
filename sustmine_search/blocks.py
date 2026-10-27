"""Load the block grid (ix, iy, iz, tonnes, grade)."""
import numpy as np


def load_grid(df, d=30.0):
    """Block table (ix, iy, iz, tonnes, grade) -> dict M for the pit optimizer.
    iz = 0 at the TOP bench. d is the block size in metres."""
    df = df[df.tonnes > 0].copy()              # zero-tonnage cells have no grade
    M = dict(ix=df.ix.values, iy=df.iy.values, iz=df.iz.values,
             tonnes=df.tonnes.values, grade=df.grade.values, d=d)
    M["nx"], M["ny"], M["nz"] = M["ix"].max() + 1, M["iy"].max() + 1, M["iz"].max() + 1
    print(f"{len(M['tonnes']):,} blocks at {d:.0f} m, {M['tonnes'].sum()/1e6:.0f} Mt")
    return M
