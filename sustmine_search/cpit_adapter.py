"""
cpit_adapter.py -- bench-phase instance -> MineLib CPIT -> schedule.

value_table(bp, p)        bench-phase economics (tonnes, processing tonnes, value in $)
build_cpit(...)           value table + precedence + capacities -> CPIT instance dict
solve_cpit_by(...)        solve and return dict(status, objective, seconds, mined_at, ...)
schedule_aggregates(d, out)  per-period ore/waste (Mt) and mine life from a solve

CPIT resources:
    r=0  rock tonnes mined in period t      <= mining_cap
    r=1  ore tonnes processed in period t   <= processing_cap
    r=2,3 (optional blending, if blend=(Gmin, Gmax) in g/t):
         sum_b (metal_b - Gmax*ore_b) x_bt <= 0      (head grade <= Gmax)
         sum_b (Gmin*ore_b - metal_b) x_bt <= 0      (head grade >= Gmin)
Discounting: evaluate_mip uses end-of-period discounting (1+i)^-t for t = 1..T; CPIT discounts
period t (0..T-1) by (1/(1+i))^t, so block values are pre-multiplied by 1/(1+i) to match.
"""
import time
import numpy as np
import pandas as pd
from .cpit import solve_cpit, check_solution

OZ_PER_G = 1.0 / 31.1035


def value_table(bp: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Bench-phase economics. Returns block_id, tonnes, processing_tonnes, value_usd."""
    d = pd.DataFrame({
        "block_id": [f"P{r.pb}_{r.bench}" for r in bp.itertuples()],
        "tonnes":   (bp.ore_t + bp.waste_t).values * 1e6,          # t
        "processing_tonnes": bp.ore_t.values * 1e6,               # t
    })
    rev  = bp.metal_g.values * 1e6 * OZ_PER_G * p["recovery"] * p["price"]   # $
    cost = bp.ore_t.values * 1e6 * p["cost_proc"] + d.tonnes.values * p["cost_mine"]
    d["value_usd"] = rev - cost
    return d


def schedule_aggregates(d: pd.DataFrame, out: dict) -> dict:
    """Per-period ore (Mt), waste (Mt) and life (periods) from a solve."""
    ton  = dict(zip(d.block_id, d.tonnes))
    proc = dict(zip(d.block_id, d.processing_tonnes))
    T = len(out["mined_at"])
    ore = np.array([sum(proc[b] for b in out["mined_at"][t]) for t in range(1, T + 1)]) / 1e6
    wst = np.array([sum(ton[b] - proc[b] for b in out["mined_at"][t]) for t in range(1, T + 1)]) / 1e6
    nz = np.where(ore + wst > 1e-9)[0]
    life = int(nz.max() + 1) if len(nz) else 0
    return dict(ore=ore[:life], waste=wst[:life], life=life)


def build_cpit(d: pd.DataFrame, preds: dict, mining_cap: float, processing_cap: float,
               T: int, discount: float, blend=None):
    """Return (I, pred_idx, ids). I has the keys read_cpit produces."""
    ids = d.block_id.tolist()
    idx = {b: i for i, b in enumerate(ids)}
    nb = len(ids)
    nr = 2 + (2 if blend is not None else 0)

    p = d.value_usd.values / (1.0 + discount)                 # end-of-period convention
    Q = np.zeros((nb, nr))
    Q[:, 0] = d.tonnes.values
    Q[:, 1] = d.processing_tonnes.values
    lb = np.full((nr, T), -np.inf)
    ub = np.full((nr, T), np.inf)
    ub[0, :] = mining_cap
    ub[1, :] = processing_cap
    if blend is not None:
        gmin, gmax = blend
        metal = d.metal_g.values                              # grams of metal per unit
        ore = d.processing_tonnes.values
        Q[:, 2] = metal - gmax * ore
        Q[:, 3] = gmin * ore - metal
        ub[2, :] = 0.0
        ub[3, :] = 0.0

    pred_idx = [[idx[q] for q in preds.get(b, [])] for b in ids]
    I = dict(NAME="benchphase", NBLOCKS=nb, NPERIODS=T, NRES=nr,
             DISCOUNT_RATE=discount, p=p, Q=Q, lb=lb, ub=ub)
    return I, pred_idx, ids


def solve_cpit_by(d: pd.DataFrame, preds: dict, mining_cap: float, processing_cap: float,
                  T: int, discount: float, time_limit: int = 120, gap: float = 0.02,
                  blend=None, check: bool = False,
                  retry_time_limit: int = 300, retry_gap: float = 0.05,
                  threads: int = 1) -> dict:
    """Solve the bench-phase instance as CPIT. Capacities in tonnes per period.

    Adaptive cap: solve with `time_limit`; if the solver stopped on time with no incumbent
    or with a gap above `retry_gap`, solve once more with `retry_time_limit`
    (retry_time_limit=0 disables the retry). `seconds` is the total wall time,
    `n_solves` the number of solver calls. `threads` is passed to HiGHS; one thread per
    solve is right when many evaluations run in parallel (one worker per CPU)."""
    I, pred_idx, ids = build_cpit(d, preds, mining_cap, processing_cap, T, discount, blend)
    t0 = time.time()
    res = solve_cpit(I, pred_idx, time_limit=time_limit, gap=gap, msg=False, threads=threads)
    n_solves = 1
    g = res["mip_gap"]
    timed_out = "time" in str(res["highs_status"]).lower()
    if retry_time_limit and timed_out and (not res["incumbent"] or not np.isfinite(g) or g > retry_gap):
        res = solve_cpit(I, pred_idx, time_limit=retry_time_limit, gap=gap, msg=False, threads=threads)
        n_solves = 2
    elapsed = time.time() - t0

    mined_at = {t: [] for t in range(1, T + 1)}
    for i, per in enumerate(res["period"]):
        if per >= 0:
            mined_at[int(per) + 1].append(ids[i])
    obj = res["objective"] if res["objective"] is not None else 0.0
    out = dict(status=res["status"], objective=obj, seconds=elapsed, mined_at=mined_at,
               mip_gap=res.get("mip_gap", float("nan")), highs_status=res.get("highs_status", ""),
               incumbent=res["incumbent"], n_solves=n_solves)
    if check:
        out["check"] = check_solution(I, pred_idx, res["X"])
    return out
