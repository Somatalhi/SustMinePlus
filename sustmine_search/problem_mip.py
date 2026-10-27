"""pymoo problem with the MIP scheduler as evaluation model. Put in sustmine_search/.

Besides F and G, every evaluation records solver diagnostics that pymoo carries on the
population (res.pop.get("gap") etc.) so a run can report how many evaluations were
truncated by the time cap:
    gap      final MIP relative gap (nan if unknown)
    sec      scheduler wall time, s (both solves if the adaptive cap retried)
    nsolve   1 or 2 (2 = retried with the long cap)
    incumb   1 if a feasible schedule was returned, 0 if the solver found none
    T        horizon used (periods)
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from .evaluate_mip import evaluate

DIAG_KEYS = ["gap", "sec", "nsolve", "incumb", "T"]


class SustMineMIP(ElementwiseProblem):
    """x = [capex $M, mining rate Mt/y, throughput Mt/y, final pushback]"""
    def __init__(self, bp, p, time_limit=60, gap=0.02, retry_time_limit=300, retry_gap=0.05, **kwargs):
        self.bp, self.p, self.time_limit, self.gap = bp, p, time_limit, gap
        self.retry_time_limit, self.retry_gap = retry_time_limit, retry_gap
        K = int(bp.pb.max())
        super().__init__(n_var=4, n_obj=5, n_ieq_constr=4,
                         xl=np.array([500.0, 12.0,  6.0, 1.0]),
                         xu=np.array([1500.0, 30.0, 15.0, float(K)]),
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        F, G, info = evaluate(x, self.bp, self.p, time_limit=self.time_limit, gap=self.gap,
                              retry_time_limit=self.retry_time_limit, retry_gap=self.retry_gap)
        out["F"], out["G"] = F, G
        out["gap"]    = float(info["mip_gap"])
        out["sec"]    = float(info["seconds"])
        out["nsolve"] = int(info["n_solves"])
        out["incumb"] = int(bool(info["incumbent"]))
        out["T"]      = int(info["T"])