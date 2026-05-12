"""
Fitness scoring for design variants.

Score = weighted product of normalized metrics. Lower is better for
mass and stress; higher is better for safety factor and wear life.
Returns a dict so the loop driver can also surface component scores.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FitnessWeights:
    mass: float = 0.25            # minimize material cost / shipping
    safety_factor: float = 0.35   # maximize structural margin
    wear_life: float = 0.25       # maximize service life
    manufacturability: float = 0.15  # minimize fab complexity


def score(
    *,
    mass_kg: float,
    max_stress_MPa: float,
    material_yield_MPa: float,
    archard_k_relative: float,
    manufacturability_penalty: float = 0.0,  # 0..1 (1 = unmanufacturable)
    weights: FitnessWeights | None = None,
) -> dict[str, float]:
    w = weights or FitnessWeights()

    sf = material_yield_MPa / max(max_stress_MPa, 1e-6)

    # Normalize each metric to a 0..1 "goodness" score where higher is better
    mass_score = 1.0 / (1.0 + mass_kg / 50.0)                    # 50 kg ~ midpoint
    sf_score = min(sf / 4.0, 1.0)                                 # 4.0 = great
    wear_score = 1.0 / (1.0 + archard_k_relative)                 # lower k = better
    mfg_score = max(0.0, 1.0 - manufacturability_penalty)

    composite = (
        w.mass * mass_score
        + w.safety_factor * sf_score
        + w.wear_life * wear_score
        + w.manufacturability * mfg_score
    )

    return {
        "composite": composite,
        "mass_score": mass_score,
        "safety_factor_score": sf_score,
        "wear_score": wear_score,
        "manufacturability_score": mfg_score,
        "safety_factor_value": sf,
    }
