"""SustMine hierarchical Pareto ranking. Port of jenks_breaks.m, jenks_gvf.m,
pareto_rank.m and the compute_indices / shannon helpers in sensitivity_analysis.m.
All functions assume MINIMISATION; flip maximised columns first (flip_signs)."""
import warnings
import numpy as np


def jenks_breaks(data, k, faithful=False):
    """Jenks natural breaks by dynamic programming. Returns k+1 boundaries (min..max).

    faithful=True reproduces jenks_breaks.m exactly. That implementation loops
    j = 2..k+1 and backtracks from column k+1, i.e. it computes a (k+1)-class
    partition and returns its boundaries with the minimum dropped. The default
    (faithful=False) is the correct k-class Fisher-Jenks, verified against jenkspy."""
    data = np.sort(np.asarray(data, float).ravel())
    n = len(data)
    if k >= n:
        warnings.warn(f"Too many classes ({k}) for data length ({n}). Reducing k.")
        k = max(2, n - 1)
    if not faithful:
        try:                                       # fast C implementation, verified identical
            import jenkspy
            return np.asarray(jenkspy.jenks_breaks(data, k), float)
        except ImportError:
            pass
        k = k - 1                                  # the MATLAB table's column j holds j classes;
                                                   # correct k-class result lives in column k
    mat1 = np.zeros((n + 1, k + 1), dtype=int)
    mat2 = np.full((n + 1, k + 1), np.inf)
    mat1[0, :] = 1
    mat2[0, :] = 0.0
    for l in range(2, n + 2):                      # 1-based l as in MATLAB
        s1 = s2 = w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            if i3 < 1 or i3 > n:
                continue
            val = data[i3 - 1]
            s1 += val; s2 += val * val; w += 1
            v = s2 - s1 * s1 / w
            if i3 != 1:
                for j in range(2, k + 2):
                    if mat2[l - 1, j - 1] >= v + mat2[i3 - 2, j - 2]:
                        mat1[l - 1, j - 1] = i3
                        mat2[l - 1, j - 1] = v + mat2[i3 - 2, j - 2]
        mat2[l - 1, 0] = s2 - s1 * s1 / w
    breaks = np.zeros(k + 2 if not faithful else k + 1)
    breaks[-1] = data[-1]
    c = n + 1
    for j in range(k + 1, 1, -1):
        idx = mat1[c - 1, j - 1] - 1
        if idx < 1:
            idx = 1
        breaks[j - 1 if not faithful else j - 2] = data[idx - 1]
        c = idx
    if not faithful:
        breaks[0] = data[0]
    return breaks


def _classify(data, breaks):
    k = len(breaks) - 1
    lab = np.zeros(len(data), dtype=int)
    lab[data <= breaks[1]] = 1
    for i in range(2, k + 1):
        lab[(data > breaks[i - 1]) & (data <= breaks[i])] = i
    return lab


def jenks_gvf(data, gvf_threshold=0.80, max_classes=8, faithful=False):
    """Smallest number of classes (2..max_classes) whose GVF >= threshold.
    Returns (class_labels 1..k, gvf)."""
    data = np.asarray(data, float).ravel()
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 3:
        warnings.warn("Not enough data points for Jenks classification. Returning all ones.")
        return np.ones(n, dtype=int), 0.0
    sdcm = np.sum((data - data.mean()) ** 2)
    kmax = min(max_classes, len(np.unique(data)) - 1)
    if kmax < 2:                                   # constant column
        return np.ones(n, dtype=int), 0.0
    labels, gvf = np.ones(n, dtype=int), 0.0
    for k in range(2, kmax + 1):
        breaks = jenks_breaks(data, k, faithful)
        labels = _classify(data, breaks)
        sdcmb = sum(np.sum((data[labels == i] - data[labels == i].mean()) ** 2)
                    for i in range(1, k + 1) if np.any(labels == i))
        gvf = (sdcm - sdcmb) / sdcm if sdcm > 0 else 1.0
        if gvf >= gvf_threshold:
            return labels, gvf
    warnings.warn(f"GVF threshold not met. Returning best result with {kmax} classes (GVF = {gvf:.2f})")
    return labels, gvf


def pareto_rank(data):
    """Successive nondominated fronts, minimisation. Rank 1 = nondominated."""
    data = np.asarray(data, float)
    n = len(data)
    ranks = np.zeros(n, dtype=int)
    remaining = np.arange(n)
    r = 1
    while len(remaining):
        D = data[remaining]
        dom = (D[:, None, :] <= D[None, :, :]).all(-1) & (D[:, None, :] < D[None, :, :]).any(-1)
        is_front = ~dom.any(0)                     # not dominated by any other remaining point
        ranks[remaining[is_front]] = r
        remaining = remaining[~is_front]
        r += 1
    return ranks


def flip_signs(DM, max_cols):
    DM = np.array(DM, float, copy=True)
    DM[:, max_cols] = -DM[:, max_cols]
    return DM


def minmax10(DM):
    DM = np.asarray(DM, float)
    lo, hi = DM.min(0), DM.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return 10.0 * (DM - lo) / span


def compute_indices(DM_norm, eco_cols, env_cols, soc_cols, gvf_thresh=0.80, max_classes=8, faithful=False):
    """Classify each indicator, rank per dimension, then rank the composite matrix.
    Returns (eco_idx, env_idx, soc_idx, comp_idx)."""
    DM_norm = np.asarray(DM_norm, float)
    DM_class = np.zeros_like(DM_norm, dtype=int)
    for i in range(DM_norm.shape[1]):
        DM_class[:, i], _ = jenks_gvf(DM_norm[:, i], gvf_thresh, max_classes, faithful)
    eco = pareto_rank(DM_class[:, eco_cols])
    env = pareto_rank(DM_class[:, env_cols])
    soc = pareto_rank(DM_class[:, soc_cols])
    comp = pareto_rank(np.c_[eco, env, soc])
    return eco, env, soc, comp


def shannon(comp_idx):
    _, counts = np.unique(comp_idx, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def sustmine_rank(DM_raw, max_cols, eco_cols, env_cols, soc_cols, gvf_thresh=0.80, faithful=False):
    """Full SustMine pipeline on a raw indicator matrix (rows = alternatives)."""
    DM = minmax10(flip_signs(DM_raw, max_cols))
    return compute_indices(DM, eco_cols, env_cols, soc_cols, gvf_thresh, faithful=faithful)