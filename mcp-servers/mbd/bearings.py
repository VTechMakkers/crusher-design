"""
Rolling-element bearing life per ISO 281.

L10 = (C / P)^p  in millions of revolutions
    = (C / P)^p × 10^6 / (60 n)  in hours
    where:
      C = basic dynamic load rating, N (manufacturer catalog)
      P = equivalent dynamic load, N
      n = rotational speed, rpm
      p = 3 for ball bearings, 10/3 for roller bearings

For time-varying load over a duty cycle, the equivalent load is computed
by the cube-root-of-cube weighting (or 10/3 for rollers) over each duty
fraction. This is the standard ISO 281 treatment.

References:
  ISO 281:2007 (rolling bearings — dynamic load ratings and rating life)
  SKF General Catalogue (the practical engineering reference)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable


BALL_BEARING_EXPONENT = 3.0
ROLLER_BEARING_EXPONENT = 10.0 / 3.0


@dataclass(frozen=True)
class BearingSpec:
    """Catalog-derived parameters for a rolling bearing.

    `dynamic_load_rating_N` is C from the manufacturer's catalog.
    `exponent` follows ISO 281: 3.0 for ball, 10/3 for roller bearings.
    """
    designation: str
    dynamic_load_rating_N: float
    exponent: float = ROLLER_BEARING_EXPONENT

    def validate(self) -> None:
        if self.dynamic_load_rating_N <= 0:
            raise ValueError("dynamic load rating must be positive")
        if self.exponent not in (BALL_BEARING_EXPONENT, ROLLER_BEARING_EXPONENT):
            raise ValueError("exponent must be 3.0 (ball) or 10/3 (roller)")


def equivalent_dynamic_load(loads_N: Iterable[float],
                              duty_fractions: Iterable[float],
                              exponent: float = ROLLER_BEARING_EXPONENT) -> float:
    """Compute P_eq across a duty cycle.

    P_eq = (Σ (P_i^p × f_i))^(1/p)
    where f_i is the fraction of time at load P_i. Σ f_i must equal 1.
    """
    loads = list(loads_N)
    fracs = list(duty_fractions)
    if len(loads) != len(fracs):
        raise ValueError("loads and duty_fractions must have equal length")
    total_f = sum(fracs)
    if not math.isclose(total_f, 1.0, abs_tol=1e-6):
        raise ValueError(f"duty fractions must sum to 1.0, got {total_f}")
    if any(f < 0 for f in fracs):
        raise ValueError("duty fractions must be non-negative")
    if any(P < 0 for P in loads):
        raise ValueError("loads must be non-negative")
    weighted = sum((P ** exponent) * f for P, f in zip(loads, fracs))
    return weighted ** (1.0 / exponent)


def L10_hours(*, equivalent_load_N: float, bearing: BearingSpec,
              speed_rpm: float) -> float:
    """ISO 281 basic rating life in hours."""
    bearing.validate()
    if speed_rpm <= 0:
        raise ValueError("speed must be positive")
    if equivalent_load_N <= 0:
        raise ValueError("equivalent load must be positive")
    L10_millions_rev = (bearing.dynamic_load_rating_N / equivalent_load_N) ** bearing.exponent
    return L10_millions_rev * 1.0e6 / (60.0 * speed_rpm)


def life_from_time_history(*, load_history_N: Iterable[float],
                             dt_s: float, bearing: BearingSpec,
                             speed_rpm: float) -> dict[str, float]:
    """L10 from a fully sampled time history of bearing load magnitudes.

    Treats each time step as an equal duty fraction. Suitable for direct
    consumption of MBD output (one sample per integration step)."""
    loads = list(load_history_N)
    if not loads:
        raise ValueError("empty load history")
    n = len(loads)
    fracs = [1.0 / n] * n
    P_eq = equivalent_dynamic_load(loads, fracs, exponent=bearing.exponent)
    L10 = L10_hours(equivalent_load_N=P_eq, bearing=bearing, speed_rpm=speed_rpm)
    return {
        "equivalent_load_N": P_eq,
        "L10_hours": L10,
        "peak_load_N": max(loads),
        "mean_load_N": sum(loads) / n,
        "samples": n,
        "duty_duration_s": n * dt_s,
        "C_over_P": bearing.dynamic_load_rating_N / P_eq,
    }
