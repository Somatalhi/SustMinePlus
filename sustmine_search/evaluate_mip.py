"""Design vector -> objectives, with the MineLib CPIT scheduler as the inner level.

x = [capex ($M), mining rate (Mt/y), throughput (Mt/y), final pushback (1..K)]
F = [-NPV ($M), waste (Mt), land (ha.yr), -employees, -job security (yr)]   all minimised
Job security is the SustMine social indicator SDI 10.16, measured as mine life in years.
G = [life_min - life, life - life_max]                                      g <= 0 feasible

Design space Omega. Besides the box bounds, three couplings between design variables
define the admissible designs; they are bounds of the design space, not constraints:
    x3 <= x2                      the plant cannot outrun the mine
    x1 >= C(x2, x3)               capital covers the plant and fleet the capacities need
    x4 <= k_max(x3)               the plant can process the declared pit within life_max,
                                  k_max(x3) = max{k : ore(1..k) <= x3 * life_max}
The optimizer searches the enclosing box; project() maps a proposal onto Omega
(x3 <- min(x3, x2); x1 <- max(x1, C); x4 <- min(round(x4), k_max)), and evaluate() scores
the projected design. The two remaining constraints, g1 and g2 on mine life, need the
schedule and are returned in G.

Periods are period_years long; capacities and discounting are per period;
job security (mine life) is reported in years.
"""
import numpy as np, pandas as pd
from .precedence import pushback_precedence
from .cpit_adapter import value_table, solve_cpit_by, schedule_aggregates

PERIOD_YEARS = 2            # 2-year periods (annual periods: T=44, no incumbent in 60 s)


def capex_required(rate, thr, p):
    return (p["capex_plant_ref"] * (thr / p["thr_ref"]) ** 0.6
            + p["capex_fleet_ref"] * (rate / p["rate_ref"]) ** 0.6)


def employees(rate, thr, p):
    return p["emp_per_mtpa_mine"] * rate + p["emp_per_mtpa_proc"] * thr + p["emp_fixed"]


def cum_ore(bp):
    """Ore (Mt) in pushbacks 1..k, indexed by k."""
    return bp.groupby("pb").ore_t.sum().sort_index().cumsum()


def k_max(thr, bp, p, _cache={}):
    """Largest pushback index whose cumulative ore the plant can process within life_max."""
    key = id(bp)
    if key not in _cache:
        _cache[key] = cum_ore(bp)
    Q = _cache[key]
    ok = Q.index[Q.values <= thr * p["life_max"] + 1e-9]
    return int(ok.max()) if len(ok) else 1


def _rounded(x, bp):
    """x with the pit index rounded and clipped to 1..K (what evaluate() would use anyway)."""
    r = np.array(x, float).copy()
    r[3] = float(int(np.clip(round(r[3]), 1, int(bp.pb.max()))))
    return r


def project(x, bp, p):
    """Map a proposal from the enclosing box onto the design space Omega. Returns a new array."""
    x = np.array(x, float).copy()
    x[2] = min(x[2], x[1])                                    # throughput <= mining rate
    x[0] = max(x[0], capex_required(x[1], x[2], p))            # capital >= required
    K = int(bp.pb.max())
    x[3] = float(min(int(np.clip(round(x[3]), 1, K)), k_max(x[2], bp, p)))
    return x


def evaluate(x, bp, p, time_limit=60, gap=0.02, period_years=PERIOD_YEARS, blend=None,
             retry_time_limit=300, retry_gap=0.05):
    """retry_time_limit / retry_gap: CPIT adaptive cap (see cpit_adapter.solve_cpit_by).
    The design is projected onto Omega first; info['x_eval'] is the design actually scored."""
    xe = project(x, bp, p)
    cx, rate, thr, k_ult = float(xe[0]), float(xe[1]), float(xe[2]), int(xe[3])

    sub   = bp[bp.pb <= k_ult].reset_index(drop=True)
    preds = pushback_precedence(sub)
    d     = value_table(sub, p)
    d["metal_g"] = sub.metal_g.values * 1e6                                # grams, for blending rows

    T_max = int(np.ceil(p["life_max"] / period_years))
    mcap, pcap = rate * 1e6 * period_years, thr * 1e6 * period_years
    # horizon bound: an NPV-maximising schedule never outlasts the time to exhaust the
    # pit at capacity; 1.5x margin for precedence-induced idling, capped at life_max
    t_need = max(d.tonnes.sum() / mcap, d.processing_tonnes.sum() / pcap)
    T     = int(min(T_max, max(2, np.ceil(1.5 * t_need))))
    disc  = (1.0 + p["discount"]) ** period_years - 1.0
    out   = solve_cpit_by(d, preds, mining_cap=mcap, processing_cap=pcap, T=T, discount=disc,
                          time_limit=time_limit, gap=gap, blend=blend,
                          retry_time_limit=retry_time_limit, retry_gap=retry_gap)
    agg   = schedule_aggregates(d, out)
    L_per = agg["life"]
    L     = L_per * period_years                 # job security = mine life, years

    # NPV: MIP objective is discounted operating value in $; add capital and closure
    npv = out["objective"] / 1e6 - cx - p["reclam"] / (1.0 + p["discount"]) ** max(L, 1)
    waste = float(agg["waste"].sum())
    area  = float(sub.drop_duplicates("pb").area_ha.sum())        # footprint of pushbacks 1..k_ult
    land  = area * L
    emp   = employees(rate, thr, p)

    F = np.array([-npv, waste, land, -emp, -L], float)
    G = np.array([p["life_min"] - L, L - p["life_max"]], float)
    info = dict(npv=npv, job_security=L, waste=waste, land=land, employees=emp, pit=k_ult,
                ore=float(agg["ore"].sum()), status=out["status"], seconds=out["seconds"],
                T=T, mip_gap=out.get("mip_gap", float("nan")), highs_status=out.get("highs_status", ""),
                incumbent=out.get("incumbent", True), n_solves=out.get("n_solves", 1),
                x_eval=xe, projected=bool(np.any(np.abs(xe - _rounded(x, bp)) > 1e-9)))
    return F, G, info


if __name__ == "__main__":
    # Smoke test: one design through the CPIT scheduler.
    import sys, time
    try:
        from .params import P
    except ImportError:
        from params import P
    #   python -m sustmine_search.evaluate_mip bench_phases.pkl [period_years gap time_limit capex rate thr pushback]
    bp = pd.read_pickle(sys.argv[1])
    py_  = int(sys.argv[2])   if len(sys.argv) > 2 else PERIOD_YEARS
    gap_ = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    tl_  = int(sys.argv[4])   if len(sys.argv) > 4 else 60
    x = [float(v) for v in sys.argv[5:9]] if len(sys.argv) >= 9 else [950.0, 20.0, 10.0, 12]
    xe = project(x, bp, P)
    print("design x =", x, f" period_years={py_} gap={gap_} time_limit={tl_}s")
    if np.any(np.abs(xe - _rounded(x, bp)) > 1e-9):
        print("projected onto Omega ->", np.round(xe, 3).tolist(),
              f"(k_max for throughput {xe[2]:.2f} Mt/y is {k_max(xe[2], bp, P)})")
    t0 = time.time()
    F, G, info = evaluate(x, bp, P, time_limit=tl_, gap=gap_, period_years=py_)
    print(f"\n[cpit]  status {info['status']} {info['highs_status']}  T={info['T']}  "
          f"gap {info['mip_gap']:.4f}  solves {info['n_solves']}  incumbent {info['incumbent']}  "
          f"solve {info['seconds']:.1f}s  total {time.time()-t0:.1f}s")
    print("  NPV $M %.1f  waste Mt %.1f  land ha.yr %.0f  employees %.0f  life yr %d  ore Mt %.1f"
          % (info["npv"], info["waste"], info["land"], info["employees"], info["job_security"], info["ore"]))
    print("  F =", np.round(F, 2).tolist())
    print("  G =", np.round(G, 2).tolist(), " feasible" if (G <= 0).all() else " INFEASIBLE")
