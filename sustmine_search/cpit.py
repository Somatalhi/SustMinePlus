"""
cpit.py -- MineLib Constrained Pit Limit Problem (CPIT), exact MIP via PuLP + HiGHS.

Formulation (cpit.mod):
    max  sum_t sum_b (1/(1+delta))^t * p_b * x_{b,t}
    s.t. lb_{r,t} <= sum_b q_{b,r} x_{b,t} <= ub_{r,t}     (Resource)
         sum_t x_{b,t} <= 1                                (CliqueBlock / reserve)
         sum_{s<=t} x_{b,s} <= sum_{s<=t} x_{i,s}  i in pred(b)   (Precedence)
         x binary

Usage (validation on a MineLib instance):
    python -m sustmine_search.cpit minelib/newman1.cpit minelib/newman1.prec
"""
import sys
import time
import math
import numpy as np
import pulp


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------
def read_cpit(path):
    """Read a MineLib .cpit file. Returns a dict I with
    NBLOCKS, NPERIODS, NRES, DISCOUNT_RATE, p (nb,), Q (nb, nr), lb (nr, nt), ub (nr, nt)."""
    I = {}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            if line.endswith(":"):
                section = line[:-1]
                if section == "RESOURCE_CONSTRAINT_LIMITS":
                    nr, nt = I["NRES"], I["NPERIODS"]
                    I["lb"] = np.full((nr, nt), -np.inf)
                    I["ub"] = np.full((nr, nt), np.inf)
                elif section == "OBJECTIVE_FUNCTION":
                    I["p"] = np.zeros(I["NBLOCKS"])
                elif section == "RESOURCE_CONSTRAINT_COEFFICIENTS":
                    I["Q"] = np.zeros((I["NBLOCKS"], I["NRES"]))
                continue
            if section is None:                      # header "KEY: value"
                key, val = [s.strip() for s in line.split(":", 1)]
                if key in ("NBLOCKS", "NPERIODS"):
                    I[key] = int(val)
                elif key == "NRESOURCE_SIDE_CONSTRAINTS":
                    I["NRES"] = int(val)
                elif key == "DISCOUNT_RATE":
                    I["DISCOUNT_RATE"] = float(val)
                else:
                    I[key] = val
                continue
            tok = line.split()
            if section == "RESOURCE_CONSTRAINT_LIMITS":
                r, t, typ = int(tok[0]), int(tok[1]), tok[2]
                if typ == "L":
                    I["ub"][r, t] = float(tok[3])
                elif typ == "G":
                    I["lb"][r, t] = float(tok[3])
                elif typ == "I":
                    I["lb"][r, t] = float(tok[3]); I["ub"][r, t] = float(tok[4])
                else:
                    raise ValueError(f"unknown limit type {typ}")
            elif section == "OBJECTIVE_FUNCTION":
                I["p"][int(tok[0])] = float(tok[1])
            elif section == "RESOURCE_CONSTRAINT_COEFFICIENTS":
                I["Q"][int(tok[0]), int(tok[1])] = float(tok[2])
    return I


def read_prec(path, nblocks):
    """Read a MineLib .prec file: each line 'b n i1 ... in'. Returns list of predecessor lists."""
    pred = [[] for _ in range(nblocks)]
    with open(path) as f:
        for raw in f:
            tok = raw.split()
            if not tok:
                continue
            b, n = int(tok[0]), int(tok[1])
            pred[b] = [int(v) for v in tok[2:2 + n]]
    return pred


# ----------------------------------------------------------------------------
# Warm start: greedy feasible schedule (as-early-as-possible, capacity-limited)
# ----------------------------------------------------------------------------
def greedy_schedule(I, pred):
    """Return period[b] (-1 = not mined) of a schedule that respects precedence,
    at-most-once and the upper resource limits (lower limits are ignored, so with
    blending rows the start may be rejected by HiGHS; it is then simply unused).
    Blocks are taken in topological order and placed in the earliest period with
    room. Used only to give the MIP an incumbent before the time limit bites."""
    nb, nt, nr = I["NBLOCKS"], I["NPERIODS"], I["NRES"]
    Q, ub = I["Q"], I["ub"]
    # topological order (Kahn)
    indeg = np.array([len(pred[b]) for b in range(nb)])
    succ = [[] for _ in range(nb)]
    for b in range(nb):
        for i in pred[b]:
            succ[i].append(b)
    order, stack = [], [b for b in range(nb) if indeg[b] == 0]
    while stack:
        b = stack.pop()
        order.append(b)
        for s in succ[b]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    period = np.full(nb, -1, int)
    use = np.zeros((nr, nt))
    for b in order:
        t0 = max([period[i] for i in pred[b]], default=0)
        if t0 < 0:                                   # a predecessor was never mined
            continue
        for t in range(t0, nt):
            if np.all(use[:, t] + Q[b] <= ub[:, t] + 1e-9):
                use[:, t] += Q[b]
                period[b] = t
                break
    # prune: drop negative-value blocks with no mined successor (reverse topological
    # order, repeated) -- removing such a block keeps precedence and capacities feasible
    p = I["p"]
    changed = True
    while changed:
        changed = False
        for b in reversed(order):
            if period[b] >= 0 and p[b] < 0 and not any(period[s] >= 0 for s in succ[b]):
                period[b] = -1
                changed = True
    return period


class _HiGHSWarm(pulp.HiGHS):
    """pulp.HiGHS plus a user-supplied starting solution {LpVariable: value}."""
    def __init__(self, start=None, **kw):
        super().__init__(**kw)
        self._start = start

    def callSolver(self, lp):
        if self._start:
            import highspy
            vals = [0.0] * lp.solverModel.getNumCol()
            for v, val in self._start.items():
                vals[v.index] = float(val)
            sol = highspy.HighsSolution()
            sol.col_value = vals
            lp.solverModel.setSolution(sol)
        lp.solverModel.run()


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
def solve_cpit(I, pred, time_limit=600, gap=0.0, msg=False, threads=None, form="by",
               warm_start=True):
    """Solve CPIT. I as from read_cpit (or built by an adapter with the same keys).

    form="by": cumulative ('mined by period t') binaries y_bt = sum_{s<=t} x_bs; precedence
               y_bt <= y_it has two nonzeros per row. Same feasible set and objective as the
               MineLib 'at' statement.
    form="at": MineLib statement verbatim, x_bt binaries, precedence on cumulative sums.
    warm_start: hand HiGHS the greedy_schedule() solution so an incumbent always exists.
    Returns dict(status, objective, X (nb x nt int array), period (nb,) -1 = not mined, wall, ...)."""
    nb, nt, nr = I["NBLOCKS"], I["NPERIODS"], I["NRES"]
    p, Q, lb, ub = I["p"], I["Q"], I["lb"], I["ub"]
    disc = np.array([(1.0 / (1.0 + I["DISCOUNT_RATE"])) ** t for t in range(nt)])

    m = pulp.LpProblem("CPIT", pulp.LpMaximize)

    if form == "by":
        Y = [[pulp.LpVariable(f"y_{b}_{t}", cat="Binary") for t in range(nt)] for b in range(nb)]
        # x_bt = y_bt - y_b,t-1  (y_b,-1 = 0); objective coefficient of y_bt is p_b (disc_t - disc_{t+1})
        dd = np.append(disc, 0.0)
        m += pulp.lpSum((dd[t] - dd[t + 1]) * p[b] * Y[b][t]
                        for b in range(nb) for t in range(nt) if p[b] != 0.0)
        for b in range(nb):                                   # monotone => mined at most once
            for t in range(1, nt):
                m += Y[b][t] >= Y[b][t - 1], f"mono_{b}_{t}"
        for r in range(nr):                                   # resources on x = diff(y)
            rows = np.nonzero(Q[:, r])[0]
            if not rows.size:
                continue
            for t in range(nt):
                expr = pulp.lpSum(Q[b, r] * (Y[b][t] - (Y[b][t - 1] if t else 0)) for b in rows)
                if np.isfinite(ub[r, t]):
                    m += expr <= ub[r, t], f"res_ub_{r}_{t}"
                if np.isfinite(lb[r, t]):
                    m += expr >= lb[r, t], f"res_lb_{r}_{t}"
        for b in range(nb):                                   # precedence, 2 nonzeros per row
            for i in pred[b]:
                for t in range(nt):
                    m += Y[b][t] <= Y[i][t], f"prec_{b}_{i}_{t}"
    else:
        X = [[pulp.LpVariable(f"x_{b}_{t}", cat="Binary") for t in range(nt)] for b in range(nb)]
        m += pulp.lpSum(disc[t] * p[b] * X[b][t] for b in range(nb) for t in range(nt) if p[b] != 0.0)
        for r in range(nr):
            rows = np.nonzero(Q[:, r])[0]
            if not rows.size:
                continue
            for t in range(nt):
                expr = pulp.lpSum(Q[b, r] * X[b][t] for b in rows)
                if np.isfinite(ub[r, t]):
                    m += expr <= ub[r, t], f"res_ub_{r}_{t}"
                if np.isfinite(lb[r, t]):
                    m += expr >= lb[r, t], f"res_lb_{r}_{t}"
        for b in range(nb):
            m += pulp.lpSum(X[b]) <= 1, f"once_{b}"
        cum = [[None] * nt for _ in range(nb)]
        for b in range(nb):
            acc = 0
            for t in range(nt):
                acc = acc + X[b][t]
                cum[b][t] = acc
        for b in range(nb):
            for i in pred[b]:
                for t in range(nt):
                    m += cum[b][t] <= cum[i][t], f"prec_{b}_{i}_{t}"

    opts = dict(msg=msg, timeLimit=time_limit, gapRel=gap)
    if threads:
        opts["threads"] = threads
    start = None
    if warm_start:
        gper = greedy_schedule(I, pred)
        if form == "by":
            start = {Y[b][t]: 1.0 for b in range(nb) if gper[b] >= 0 for t in range(gper[b], nt)}
        else:
            start = {X[b][gper[b]]: 1.0 for b in range(nb) if gper[b] >= 0}
    t0 = time.time()
    m.solve(_HiGHSWarm(start=start, **opts))
    wall = time.time() - t0

    if form == "by":
        Yv = np.array([[int(round(Y[b][t].value() or 0)) for t in range(nt)] for b in range(nb)])
        Yv = np.maximum.accumulate(Yv, axis=1)                # guard against tiny tolerance dips
        Xv = np.diff(np.concatenate([np.zeros((nb, 1), int), Yv], axis=1), axis=1)
    else:
        Xv = np.array([[int(round(X[b][t].value() or 0)) for t in range(nt)] for b in range(nb)])
    period = np.where(Xv.sum(1) > 0, Xv.argmax(1), -1)
    mip_gap, hstatus = float("nan"), ""
    try:                                   # true solver gap / status (highspy)
        h = m.solverModel
        mip_gap = float(h.getInfo().mip_gap)
        hstatus = h.modelStatusToString(h.getModelStatus())
    except Exception:
        pass
    obj = pulp.value(m.objective)
    return dict(status=pulp.LpStatus[m.status], objective=obj, X=Xv, period=period, wall=wall,
                mip_gap=mip_gap, highs_status=hstatus,
                incumbent=obj is not None and Xv.sum() > 0, warm_start=start is not None)


# ----------------------------------------------------------------------------
# Independent feasibility / value check (does not use the model object)
# ----------------------------------------------------------------------------
def check_solution(I, pred, X):
    nb, nt, nr = I["NBLOCKS"], I["NPERIODS"], I["NRES"]
    disc = np.array([(1.0 / (1.0 + I["DISCOUNT_RATE"])) ** t for t in range(nt)])
    out = {}
    out["objective_recomputed"] = float((I["p"][:, None] * disc[None, :] * X).sum())
    out["once_violations"] = int((X.sum(1) > 1).sum())
    use = I["Q"].T @ X                                      # nr x nt
    out["resource_use"] = use
    out["resource_violations"] = int(((use > I["ub"] + 1e-6) | (use < I["lb"] - 1e-6)).sum())
    cum = X.cumsum(1)
    viol = 0
    for b in range(nb):
        for i in pred[b]:
            if (cum[b] > cum[i]).any():
                viol += 1
    out["precedence_violations"] = viol
    out["blocks_per_period"] = X.sum(0)
    return out


if __name__ == "__main__":
    cpit_path, prec_path = sys.argv[1], sys.argv[2]
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    form = sys.argv[4] if len(sys.argv) > 4 else "by"
    I = read_cpit(cpit_path)
    pred = read_prec(prec_path, I["NBLOCKS"])
    print(f"{I.get('NAME')}: {I['NBLOCKS']} blocks, {I['NPERIODS']} periods, "
          f"{I['NRES']} resources, delta={I['DISCOUNT_RATE']}, "
          f"{sum(len(p) for p in pred)} precedence arcs")
    res = solve_cpit(I, pred, time_limit=3600, gap=gap, msg=False, form=form)
    print(f"form {form}  status {res['status']} {res['highs_status']}  objective {res['objective']:,.2f}  "
          f"gap {res['mip_gap']:.5f}  wall {res['wall']:.1f}s")
    chk = check_solution(I, pred, res["X"])
    print(f"objective recomputed from X: {chk['objective_recomputed']:,.2f}")
    print("blocks mined per period:", chk["blocks_per_period"].tolist(),
          " total", int(res['X'].sum()), "of", I["NBLOCKS"])
    for r in range(I["NRES"]):
        print(f"resource {r} use/ub per period:",
              [f"{u:,.0f}/{b:,.0f}" for u, b in zip(chk['resource_use'][r], I['ub'][r])])
    print(f"violations: once {chk['once_violations']}, resource {chk['resource_violations']}, "
          f"precedence {chk['precedence_violations']}")