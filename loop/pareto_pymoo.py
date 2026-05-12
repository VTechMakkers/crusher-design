"""
pymoo-backed multi-objective ranking and selection.

Our use case is post-hoc selection from a pre-evaluated candidate set
(not running an evolutionary search), so we use pymoo's underlying
utilities directly rather than instantiating an Algorithm.

What this module exposes:
  rank_pymoo(candidates)              — pymoo NonDominatedSorting
  select_nsga2(front_members, k)      — crowding-distance diversity
  select_nsga3(front_members, k)      — reference-direction diversity

NSGA-III is the practical reason to adopt pymoo: with 6-D objectives
(our system_fitness output) crowding distance loses its diversity
discrimination because most points have at least one objective at an
extreme and inherit ∞ crowding. Reference-direction association is the
standard fix from Deb & Jain (2014).

Our home-grown `loop.pareto` module remains the default backend; this
module is loaded only when an `--algorithm` flag selects it.
"""
from __future__ import annotations
from typing import Any


def _require_pymoo():
    try:
        import pymoo  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pymoo not installed; `pip install pymoo` to use this backend"
        ) from e


def _to_min_array(candidates: list[Any]):
    """Pack candidate objectives into a numpy array, flipped to minimization.

    Our Candidate convention is "all maximize"; pymoo's convention is "all
    minimize". Negate so non-dominated sorting orders the same way.
    """
    import numpy as np
    return -np.array([c.objectives for c in candidates], dtype=float)


def rank_pymoo(candidates: list[Any]) -> dict[str, int]:
    """Non-dominated sort via pymoo. Returns {candidate.id -> front_index}.

    Equivalent in meaning to `loop.pareto.rank_all` but uses pymoo's
    efficient implementation (best-rank-on-first-front variant)."""
    _require_pymoo()
    if not candidates:
        return {}
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    F = _to_min_array(candidates)
    fronts = NonDominatedSorting().do(F, only_non_dominated_front=False)
    ranks: dict[str, int] = {}
    for rank, idx_array in enumerate(fronts):
        for i in idx_array:
            ranks[candidates[int(i)].id] = rank
    return ranks


def select_nsga2(front_members: list[Any], k: int) -> list[Any]:
    """Select k diverse members from a Pareto front using NSGA-II
    crowding distance (Deb et al. 2002)."""
    _require_pymoo()
    if len(front_members) <= k:
        return list(front_members)
    from pymoo.operators.survival.rank_and_crowding.metrics import (
        calc_crowding_distance,
    )
    F = _to_min_array(front_members)
    cd = calc_crowding_distance(F)
    # Larger crowding distance = more isolated = better representative.
    # Endpoints (per objective) receive ∞.
    order = sorted(range(len(front_members)), key=lambda i: -cd[i])
    return [front_members[i] for i in order[:k]]


def select_nsga3(front_members: list[Any], k: int) -> list[Any]:
    """Select k diverse members using NSGA-III reference-direction
    association (Deb & Jain 2014).

    Delegates to pymoo's tested implementation of associate_to_niches +
    niching rather than re-rolling the tie-break logic. The tie-break
    rules in NSGA-III are subtle enough that a custom version reliably
    skews the spread on edge cases (verified by test_pareto_pymoo.py
    against simple 2-D Pareto curves).

    For dim ≤ 3 uses Das-Dennis ref directions (includes axis corners);
    for higher dimensions uses the energy method which still includes
    corners but spreads non-corner points more evenly than dense
    Das-Dennis would.
    """
    _require_pymoo()
    if len(front_members) <= k:
        return list(front_members)

    import numpy as np
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.algorithms.moo.nsga3 import (associate_to_niches, niching,
                                             calc_niche_count)
    from pymoo.core.population import Population

    F = _to_min_array(front_members)
    n_obj = F.shape[1]

    if n_obj <= 3:
        ref_dirs = get_reference_directions("das-dennis", n_obj,
                                              n_partitions=max(k, 4))
    else:
        ref_dirs = get_reference_directions("energy", n_obj,
                                              n_points=max(k * 2, 24))

    # Build the normalised objective matrix the way NSGA-III expects.
    # associate_to_niches takes (F, niches, ideal_point, nadir_point) and
    # handles its own normalisation internally — we just need to pass the
    # ideal + nadir points of the front.
    ideal_point = F.min(axis=0)
    nadir_point = F.max(axis=0)
    # Guard against degenerate axes (front constant on some objective)
    span = nadir_point - ideal_point
    if (span < 1e-12).any():
        nadir_point = nadir_point + (span < 1e-12) * 1.0

    niche_of_individuals, dist_to_niche, _dist_matrix = associate_to_niches(
        F, ref_dirs, ideal_point, nadir_point,
    )

    # Build a dummy Population for the niching helper (it indexes into pop).
    pop = Population.new(F=F)
    niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[[]])  # start empty

    survivor_idx = niching(
        pop, n_remaining=k,
        niche_count=niche_count,
        niche_of_individuals=niche_of_individuals,
        dist_to_niche=dist_to_niche,
    )

    return [front_members[int(i)] for i in survivor_idx]
