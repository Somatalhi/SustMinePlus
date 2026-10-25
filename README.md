# sustmine-search

Benchmarking search-based and design-and-rank routes to sustainable strategic
mine planning alternatives. NSGA-II, NSGA-III and a Latin hypercube sample are
evaluated with the same two-level model: an outer design vector (capital,
mining rate, plant throughput, final pushback) and an inner exact production
scheduler, the MineLib constrained pit limit problem (CPIT), solved on
bench-phase units with pushback precedence. Sustainability indicators are read
off the NPV-optimal schedule and the final sets are ranked with SustMine's
hierarchical Pareto procedure.

## Pipeline

1. `make_shells.py` — nested pit shells by max-flow across 37 revenue factors
   (0.20–0.90 step 0.02, and 1.00) → `shells.pkl`, `shell_of_block.npy`
2. `make_bench_phases.py` — blocks inside the ultimate pit → equal-tonnage
   pushbacks × benches → `bench_phases.pkl`
3. `check_precedence.py` — verifies the pushback precedence structure
4. `look_blockmodel.py blocks_small_30.csv 30` — inventory, RF 1.0 pit, plan
   and section plots; `view_blockmodel_3d.py` — interactive 3-D view (Plotly)
5. `python -m sustmine_search.cpit minelib/newman1.cpit minelib/newman1.prec` —
   validates the CPIT solver on a MineLib instance
6. `python -m sustmine_search.evaluate_mip bench_phases.pkl` — one design
   through both schedulers (smoke test)
7. `run_smoke_mip.py` — short parallel NSGA-II run; proof of plumbing
8. `run_exp1_mip.py --algo {nsga2,nsga3,lhs} --seed S --budget B` —
   Experiment 1 (search versus sampling), one run per call →
   `results/exp1_cpit_<B>/<algo>_seed<S>.npz` with X, F, G and per-evaluation
   solver diagnostics (gap, seconds, retries, incumbent, horizon)
9. `metrics_exp1.py` — hypervolume, IGD+, set coverage, Mann–Whitney / A12
10. `run_exp2.py` — Experiment 2 (complementarity of search and hierarchical
    ranking): flat vs SustMine ranking of the nondominated sets
11. `run_exp3.py` — Experiment 3 (robustness of the shortlist): indicator
    noise, Front-1 stability, hypervolume gap
12. `list_front1.py`, `cluster_front1.py`, `plot_exp1.py`,
    `plot_front1_scatter.py` — shortlist, archetypes and figures
13. `diag_solver.py` — re-evaluates stored designs to check the effect of the
    solver time cap

## Package

* `sustmine_search/blocks.py` — loads the block grid (ix, iy, iz, tonnes, grade)
* `sustmine_search/pit.py` — pit limit by max-flow (igraph), slope cone, nested shells
* `sustmine_search/precedence.py` — pushback precedence for bench-phases
* `sustmine_search/cpit.py` — MineLib CPIT in PuLP/HiGHS, cumulative ("by")
  variables, greedy warm start, MineLib file readers, solution checker
* `sustmine_search/cpit_adapter.py` — bench-phase instance → CPIT resources
  (rock, ore, optional blending rows); adaptive time cap
* `sustmine_search/scheduler_mip.py` — value table and schedule aggregates;
  the earlier "by" scheduler kept for comparison
* `sustmine_search/evaluate_mip.py` — design vector → five objectives
  (NPV, waste, land, employees, job security) and four constraints;
  2-year periods, horizon bound
* `sustmine_search/problem_mip.py` — pymoo `ElementwiseProblem` with
  solver diagnostics carried on the population
* `sustmine_search/ranking.py` — SustMine hierarchical Pareto ranking
* `params.py` — all case-study parameters in one place

## Data

`blocks_small_30.csv`: 129,362 blocks of 30 × 30 × 30 m, 7,848 Mt, 21 benches;
columns `ix, iy, iz, tonnes, grade, sulphide_frac`, `iz = 0` at the top bench.

## Install

```
pip install -r requirements.txt
```

PuLP 3.x with `highspy`; `python-igraph`, `pymoo`, `numpy`, `pandas`, `scipy`.
Plotly and matplotlib only for the viewing scripts.

## On the cluster

Only `blocks_small_30.csv` needs to travel; steps 1–2 regenerate everything
else. Each `run_exp1_mip.py` call is independent and can be submitted as one
job (`nrp/make_jobs.py --workers 16`); `nrp/update-job.yaml` refreshes the code
on the volume without touching `results/`.
