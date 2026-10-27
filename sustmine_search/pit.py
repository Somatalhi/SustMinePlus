"""Ultimate pit by max-flow / min-cut (Lerchs-Grossmann as maximum closure)."""
import numpy as np
import igraph as ig

OZ_PER_G = 1 / 31.1035


def block_value(M, rf, p):
    rev = M["grade"] * OZ_PER_G * p["recovery"] * p["price"] * rf
    return M["tonnes"] * (np.maximum(rev - p["cost_proc"], 0.0) - p["cost_mine"])


def precedence(M, slope_deg, nb=4):
    """Arc list (from, to): each block -> every block in the cone above within nb benches."""
    d = M["d"]
    nx, ny, nz = M["nx"], M["ny"], M["nz"]
    lut = -np.ones((nx, ny, nz), dtype=np.int64)
    lut[M["ix"], M["iy"], M["iz"]] = np.arange(len(M["ix"]))
    tan_s = np.tan(np.radians(slope_deg))
    frm, to = [], []
    for k in range(1, nb + 1):
        r = k * d / tan_s
        o = int(np.ceil(r / d))
        OX, OY = np.meshgrid(np.arange(-o, o + 1), np.arange(-o, o + 1))
        keep = (OX * d) ** 2 + (OY * d) ** 2 <= r ** 2
        for ox, oy in zip(OX[keep], OY[keep]):
            jx, jy, jz = M["ix"] + ox, M["iy"] + oy, M["iz"] - k
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0)
            src = np.where(ok)[0]
            dst = lut[jx[ok], jy[ok], jz[ok]]
            good = dst >= 0
            frm.append(src[good]); to.append(dst[good])
    return np.concatenate(frm), np.concatenate(to)


def ultimate_pit(M, rf, p, prec=None):
    """Boolean mask of blocks inside the pit at revenue factor rf."""
    n = len(M["tonnes"])
    val = block_value(M, rf, p)
    if prec is None:
        prec = precedence(M, p["slope"])
    pf_, pt_ = prec
    s, t = n, n + 1
    pos = np.where(val > 0)[0]; neg = np.where(val < 0)[0]
    big = 1e3 * np.nansum(np.abs(val)) + 1.0
    edges = np.concatenate([np.c_[np.full(len(pos), s), pos],
                            np.c_[neg, np.full(len(neg), t)],
                            np.c_[pf_, pt_]])
    caps = np.concatenate([val[pos], -val[neg], np.full(len(pf_), big)])
    g = ig.Graph(n=n + 2, edges=edges.tolist(), directed=True)
    flow = g.maxflow(s, t, capacity=caps.tolist())
    side = np.array(flow.membership)          # partition of the min cut
    return side[:n] == side[s]                 # source side = pit


def nested_shells(M, rfs, p, cog):
    """Incremental shells at a FIXED cut-off. Same fields evalSchedule expects."""
    prec = precedence(M, p["slope"])
    prev = np.zeros(len(M["tonnes"]), bool)
    shells = []
    for rf in rfs:
        in_pit = ultimate_pit(M, rf, p, prec)
        inc = in_pit & ~prev
        ore = inc & (M["grade"] >= cog)
        surf = np.unique(np.c_[M["ix"][in_pit], M["iy"][in_pit]], axis=0)
        shells.append(dict(
            rf=rf,
            ore_t=M["tonnes"][ore].sum() / 1e6,
            waste_t=M["tonnes"][inc & ~ore].sum() / 1e6,
            moz=(M["tonnes"][ore] * M["grade"][ore]).sum() * OZ_PER_G / 1e6,
            area_ha=len(surf) * M["d"] ** 2 / 1e4,
        ))
        prev = in_pit
    return shells