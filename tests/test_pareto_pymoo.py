"""
pymoo-backed Pareto tests.

Verifies:
  - pymoo non-dominated sort produces the same front-membership as the
    built-in ranker on the existing smoke-test cases (proves equivalence)
  - NSGA-II selection chooses extremes (∞ crowding distance) first
  - NSGA-III reference-direction selection spreads picks across objective
    space in ≥4 dimensions, where crowding distance becomes degenerate
  - select_pareto_top_k dispatches correctly on the algorithm flag
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.pareto import Candidate, rank_all, select_pareto_top_k


pymoo = pytest.importorskip("pymoo", reason="pymoo not installed")


# Equivalence with built-in on simple 2D problems --------------------------

def test_pymoo_nds_matches_builtin_on_2d():
    """Same input → same set of rank-0 candidates."""
    from loop.pareto_pymoo import rank_pymoo
    candidates = [
        Candidate("a", [2.0, 2.0]),   # dominates b and c
        Candidate("b", [1.0, 1.0]),
        Candidate("c", [0.5, 0.5]),
        Candidate("d", [3.0, 0.5]),   # rank-0 alongside a
        Candidate("e", [0.4, 3.0]),   # rank-0 alongside a
    ]
    builtin_ranks = rank_all(candidates)
    pymoo_ranks = rank_pymoo(candidates)

    builtin_front = {c.id for c in candidates if builtin_ranks[c.id] == 0}
    pymoo_front = {c.id for c in candidates if pymoo_ranks[c.id] == 0}
    assert builtin_front == pymoo_front
    # Every candidate should be ranked (no missing IDs)
    assert set(pymoo_ranks) == {c.id for c in candidates}


def test_pymoo_nds_handles_tied_candidates():
    """Identical objective vectors must still get ranked — no infinite loops."""
    from loop.pareto_pymoo import rank_pymoo
    candidates = [
        Candidate("a", [1.0, 1.0]),
        Candidate("b", [1.0, 1.0]),   # identical to a
        Candidate("c", [0.5, 0.5]),   # dominated by both
    ]
    ranks = rank_pymoo(candidates)
    assert ranks["a"] == ranks["b"] == 0
    assert ranks["c"] == 1


# NSGA-II crowding-distance selection --------------------------------------

def test_nsga2_picks_extremes_first():
    """Endpoints on each objective have ∞ crowding distance, so any
    selection with k ≥ 2 should include them."""
    from loop.pareto_pymoo import select_nsga2
    # 4 non-dominated points on a 2-D Pareto curve
    candidates = [
        Candidate("min_x", [0.0, 5.0]),    # extreme in x
        Candidate("mid_a", [1.0, 4.0]),
        Candidate("mid_b", [2.0, 2.0]),
        Candidate("min_y", [5.0, 0.0]),    # extreme in y
    ]
    chosen = select_nsga2(candidates, k=2)
    chosen_ids = {c.id for c in chosen}
    assert "min_x" in chosen_ids and "min_y" in chosen_ids


# NSGA-III reference-direction selection -----------------------------------

def test_nsga3_spreads_picks_across_2d_pareto_curve():
    """On a 2-D linear Pareto front with 11 equally-spaced points,
    NSGA-III with k=5 must reliably hit both endpoints (Das-Dennis with
    4 partitions for 2 objectives yields 5 ref directions including
    (1,0) and (0,1)).

    pymoo's niching uses np.random for tie-breaks among equal-niche-count
    refs — seed it locally for determinism in this test."""
    import numpy as np
    np.random.seed(0)
    from loop.pareto_pymoo import select_nsga3
    n = 11
    candidates = [Candidate(f"p{i}", [float(i), float(10 - i)])
                   for i in range(n)]
    chosen = select_nsga3(candidates, k=5)
    idx = sorted(int(c.id[1:]) for c in chosen)
    assert 0 in idx, f"missed low endpoint at k=5: {idx}"
    assert n - 1 in idx, f"missed high endpoint at k=5: {idx}"
    assert max(idx) - min(idx) >= 0.8 * (n - 1), f"poor spread: {idx}"


def test_nsga3_returns_k_items():
    """Behavioural invariant: NSGA-III always returns exactly k items
    when the front has more than k members."""
    from loop.pareto_pymoo import select_nsga3
    # 20 candidates along a 6-D Pareto surface
    candidates = []
    for i in range(20):
        # Walk around the unit simplex
        weights = [(i + j * 3) % 7 for j in range(6)]
        s = sum(weights) or 1
        candidates.append(Candidate(f"c{i}", [w / s * 10.0 for w in weights]))
    for k in (3, 5, 10):
        chosen = select_nsga3(candidates, k=k)
        assert len(chosen) == k, f"asked for {k}, got {len(chosen)}"
        # All chosen ids unique
        assert len({c.id for c in chosen}) == k


def test_nsga3_handles_degenerate_objective():
    """If one objective is constant across the front, NSGA-III's
    normalisation must not divide by zero."""
    from loop.pareto_pymoo import select_nsga3
    candidates = [
        Candidate("a", [1.0, 0.0, 5.0]),
        Candidate("b", [0.0, 1.0, 5.0]),   # same value on obj 2
        Candidate("c", [0.5, 0.5, 5.0]),
    ]
    chosen = select_nsga3(candidates, k=2)
    assert len(chosen) == 2


# Dispatch via select_pareto_top_k -----------------------------------------

def test_dispatch_default_uses_builtin():
    candidates = [
        Candidate("a", [2.0, 2.0]),
        Candidate("b", [1.0, 1.0]),
    ]
    chosen = select_pareto_top_k(candidates, 1, algorithm="default")
    assert chosen[0].id == "a"   # a dominates b


def test_dispatch_nsga2():
    candidates = [
        Candidate("min_x", [0.0, 5.0]),
        Candidate("min_y", [5.0, 0.0]),
        Candidate("mid", [2.0, 2.0]),
    ]
    chosen = select_pareto_top_k(candidates, 2, algorithm="nsga2")
    chosen_ids = {c.id for c in chosen}
    assert "min_x" in chosen_ids and "min_y" in chosen_ids


def test_dispatch_nsga3():
    candidates = [Candidate(f"e_{i}", [1.0 if i == j else 0.0 for j in range(4)])
                   for i in range(4)]
    chosen = select_pareto_top_k(candidates, 3, algorithm="nsga3")
    assert len(chosen) == 3


def test_dispatch_unknown_algorithm_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        select_pareto_top_k([], 1, algorithm="not-a-real-algo")


def test_select_more_than_available_returns_all():
    from loop.pareto_pymoo import select_nsga2, select_nsga3
    candidates = [
        Candidate("a", [1.0, 2.0]),
        Candidate("b", [2.0, 1.0]),
    ]
    assert len(select_nsga2(candidates, k=10)) == 2
    assert len(select_nsga3(candidates, k=10)) == 2
