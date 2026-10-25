""" Hybrid Case study parameters. Single source of truth; import everywhere."""
#===============================================================
# 0. Baseline optimization parameters
#===============================================================
P = dict(
    # Economics
    price=4000.0,            # $/oz
    recovery=0.82,           #
    cost_mine=3.28,          # $/t moved
    cost_proc=19.25,         # $/t processed
    reclam=3.0,              # $M at closure
    discount=0.10,           #


    # Geotechnical
    slope=55.0,              # overall slope, deg

    # Capital cost model: C = plant_ref*(thr/thr_ref)^0.6 + fleet_ref*(rate/rate_ref)^0.6
    # Six-Tenth Rule:
    thr_ref=10.0,  capex_plant_ref=540.0,    # $M at 10 Mt/y
    rate_ref=17.0, capex_fleet_ref=360.0,    # $M at 17 Mt/y

    # Workforce: E = a*rate + b*thr + fixed
    emp_per_mtpa_mine=8.4,
    emp_per_mtpa_proc=7.5,
    emp_fixed=163.0,

    # Mine-life bounds, years
    life_min=10,
    life_max=50,
)
COG = P["cost_proc"] / (P["recovery"] * P["price"] / 31.1035)   # marginal cut-off, g/t
