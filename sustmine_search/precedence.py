"""Pushback precedence for bench-phases.
Bench-phase (pb k, bench b) can be mined only after
  the bench-phase directly above it: (k, b-1), or, if pushback k has no unit on
      bench b-1 (its ring is empty there, so its wall coincides with an inner
      wall), the nearest inner pushback k' < k that does have a unit on b-1;
  the bench-phase inside it on the same bench: (k-1, b), or, if that ring is
      empty on bench b, the nearest inner pushback k' < k-1 that has one.
With no gaps this is the standard rule (k, b-1) and (k-1, b) for nested-pit
designs (Munoz et al., 2018, OPPSP; Hustrulid, Kuchta & Martin, 2013).
Put this file in sustmine_search/.
"""
import pandas as pd


def bp_id(pb, bench):
    return f"P{pb}_{bench}"


def pushback_precedence(bp: pd.DataFrame) -> dict:
    """Returns {bench_phase_id: [predecessor_ids]} for every row of bp."""
    exists = {(int(r.pb), int(r.bench)) for r in bp.itertuples()}

    def nearest_inner(bench, k_max):
        """Largest k' <= k_max with a unit on this bench, or None."""
        for kk in range(k_max, 0, -1):
            if (kk, bench) in exists:
                return kk
        return None

    preds = {}
    for r in bp.itertuples():
        k, b = int(r.pb), int(r.bench)
        pr = []
        ka = nearest_inner(b - 1, k)        # unit above: this pushback or nearest inner one
        if ka is not None:
            pr.append(bp_id(ka, b - 1))
        kl = nearest_inner(b, k - 1)        # unit inside on the same bench, skipping empty rings
        if kl is not None:
            pr.append(bp_id(kl, b))
        preds[bp_id(k, b)] = pr
    return preds