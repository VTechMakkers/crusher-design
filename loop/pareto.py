"""
Multi-objective Pareto rank.

Given a set of candidate designs each scored on K objectives (some maximize,
some minimize), returns the Pareto-optimal subset (non-dominated front) plus
optional NSGA-II-style ranks for the rest.

No external dependencies. For our problem size (10–1000 candidates × 6
objectives), O(N^2) Pareto sort is fine.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    id: str
    objectives: list[float]   # ALL maximize (caller flips signs ahead of time)
    payload: Any = None       # original record for downstream use


def _dominates(a: list[float], b: list[float]) -> bool:
    """a dominates b iff a is >= b on every obj AND > b on at least one."""
    better_or_equal = all(ai >= bi for ai, bi in zip(a, b))
    strictly_better = any(ai > bi for ai, bi in zip(a, b))
    return better_or_equal and strictly_better


def front(candidates: list[Candidate]) -> list[Candidate]:
    """Return the Pareto-optimal subset (rank 0)."""
    out: list[Candidate] = []
    for i, ci in enumerate(candidates):
        dominated = False
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            if _dominates(cj.objectives, ci.objectives):
                dominated = True
                break
        if not dominated:
            out.append(ci)
    return out


def rank_all(candidates: list[Candidate]) -> dict[str, int]:
    """NSGA-II-style ranking. Front 0 = Pareto optimal. Returns {id: rank}."""
    remaining = list(candidates)
    ranks: dict[str, int] = {}
    current_rank = 0
    while remaining:
        f = front(remaining)
        for c in f:
            ranks[c.id] = current_rank
        remaining = [c for c in remaining if c.id not in ranks]
        current_rank += 1
        if current_rank > len(candidates):  # safety
            break
    return ranks


def crowding_distance(front_members: list[Candidate]) -> dict[str, float]:
    """NSGA-II crowding distance — favours spread across the Pareto front.
    Larger distance = more isolated point (more diversity contribution)."""
    n = len(front_members)
    if n == 0:
        return {}
    if n <= 2:
        return {c.id: float("inf") for c in front_members}

    k_objs = len(front_members[0].objectives)
    dist: dict[str, float] = {c.id: 0.0 for c in front_members}

    for o in range(k_objs):
        sorted_by_obj = sorted(front_members, key=lambda c: c.objectives[o])
        dist[sorted_by_obj[0].id] = float("inf")
        dist[sorted_by_obj[-1].id] = float("inf")
        obj_min = sorted_by_obj[0].objectives[o]
        obj_max = sorted_by_obj[-1].objectives[o]
        rng = obj_max - obj_min
        if rng == 0:
            continue
        for k in range(1, n - 1):
            prev_v = sorted_by_obj[k - 1].objectives[o]
            next_v = sorted_by_obj[k + 1].objectives[o]
            cur_id = sorted_by_obj[k].id
            if dist[cur_id] != float("inf"):
                dist[cur_id] += (next_v - prev_v) / rng
    return dist


def select_diverse(front_members: list[Candidate], k: int) -> list[Candidate]:
    """Select k diverse members from the Pareto front by crowding distance."""
    if len(front_members) <= k:
        return list(front_members)
    cd = crowding_distance(front_members)
    return sorted(front_members, key=lambda c: -cd[c.id])[:k]


# ---------------------------------------------------------------------------
# Algorithm-selectable top-K (unified entry point for callers that want to
# pick between the built-in NSGA-II-style and pymoo's NSGA-II / NSGA-III).

ALGORITHMS = ("default", "nsga2", "nsga3")


def select_pareto_top_k(candidates: list[Candidate], k: int,
                         algorithm: str = "default") -> list[Candidate]:
    """Find the Pareto front and select k diverse members from it.

    algorithm:
      'default' — built-in NSGA-II-style (crowding distance, no extra deps)
      'nsga2'   — pymoo NSGA-II (pymoo's tested non-dominated sort +
                   crowding distance; better-tested than the built-in)
      'nsga3'   — pymoo NSGA-III reference-direction association
                   (recommended when objectives ≥ 4 — crowding distance
                   degrades in high-D)
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}"
        )
    if not candidates:
        return []
    if algorithm == "default":
        ranks = rank_all(candidates)
        front_members = [c for c in candidates if ranks[c.id] == 0]
        return select_diverse(front_members, k)

    from .pareto_pymoo import rank_pymoo, select_nsga2, select_nsga3
    ranks = rank_pymoo(candidates)
    front_members = [c for c in candidates if ranks[c.id] == 0]
    if algorithm == "nsga2":
        return select_nsga2(front_members, k)
    if algorithm == "nsga3":
        return select_nsga3(front_members, k)
    raise AssertionError("unreachable")  # ALGORITHMS guard above
