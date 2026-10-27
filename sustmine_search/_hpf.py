"""Direct call to the pseudoflow C library (Hochbaum 2008) with a NumPy arc matrix.
The pseudoflow package's Python wrapper builds Python lists for every arc, which
exhausts memory at ~10^7 arcs. This passes the array straight through instead."""
import sys
import numpy as np
from ctypes import c_int, c_double, POINTER, byref, cast
import pseudoflow

_m = sys.modules[pseudoflow.hpf.__module__]
_lib = _m.libhpf


def hpf_solve(n_nodes, source, sink, arcs):
    """arcs: float64 array (n_arcs, 4) = [from, to, capacity, 0]. Node ids 0..n_nodes-1.
    Returns int array of length n_nodes: 1 = source side of the min cut (in pit)."""
    arcs = np.ascontiguousarray(arcs, dtype=np.float64)
    n_arcs = arcs.shape[0]
    c_arc = arcs.ctypes.data_as(POINTER(c_double))
    lam = (c_double * 2)(0.0, 0.0)
    n_bp = c_int(0)
    cuts = POINTER(c_int)()
    bps = POINTER(c_double)()
    stats = (c_int * 5)()
    times = (c_double * 3)()
    f = _lib.hpf_solve
    f.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_double), c_double * 2, c_int,
                  POINTER(c_int), POINTER(POINTER(c_int)), POINTER(POINTER(c_double)),
                  c_int * 5, c_double * 3]
    f(c_int(n_nodes), c_int(n_arcs), c_int(source), c_int(sink), c_arc, lam, c_int(0),
      byref(n_bp), byref(cuts), byref(bps), stats, times)
    out = np.ctypeslib.as_array(cuts, shape=(n_nodes,)).copy()
    _lib.libfree(cuts); _lib.libfree(bps)
    return out
