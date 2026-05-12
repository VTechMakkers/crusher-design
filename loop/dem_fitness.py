"""
DEM-derived fitness metrics.

Augments the structural fitness (loop/fitness.py) with functional metrics
extracted from LIGGGHTS DEM simulation:
  - throughput score      (TPH delivered vs target)
  - p80 score             (product gradation within customer spec)
  - wear uniformity score (penalize localized hot spots → premature failure)
  - energy efficiency     (kWh/t vs catalog)

Composite total fitness mixes structural + functional + material + manufacturability.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DEMFitnessWeights:
    throughput: float = 0.35
    p80_in_spec: float = 0.25
    wear_uniformity: float = 0.25
    energy_efficiency: float = 0.15


def dem_score(
    *,
    throughput_kg_per_s: float,
    target_tph: float,
    p80_mm: float,
    p80_target_mm: float,
    p80_tolerance_mm: float = 8.0,
    wear_uniformity_index: float,    # 0..1 (1 = perfectly uniform contact map)
    energy_kWh_per_t: float,
    energy_baseline_kWh_per_t: float = 1.5,
    weights: DEMFitnessWeights | None = None,
) -> dict[str, float]:
    w = weights or DEMFitnessWeights()

    measured_tph = throughput_kg_per_s * 3.6
    tph_ratio = measured_tph / max(target_tph, 1e-6)
    # Reward hitting target; penalize over- or under-shoot
    throughput_score = 1.0 - min(abs(tph_ratio - 1.0), 1.0)

    p80_error = abs(p80_mm - p80_target_mm)
    p80_score = max(0.0, 1.0 - p80_error / max(p80_tolerance_mm, 0.5))

    wear_score = max(0.0, min(1.0, wear_uniformity_index))

    energy_ratio = energy_baseline_kWh_per_t / max(energy_kWh_per_t, 1e-6)
    energy_score = min(energy_ratio, 1.2) / 1.2

    composite = (
        w.throughput * throughput_score
        + w.p80_in_spec * p80_score
        + w.wear_uniformity * wear_score
        + w.energy_efficiency * energy_score
    )

    return {
        "dem_composite": composite,
        "throughput_score": throughput_score,
        "p80_score": p80_score,
        "wear_uniformity_score": wear_score,
        "energy_efficiency_score": energy_score,
        "measured_tph": measured_tph,
    }


def combined_score(structural: dict[str, float], dem: dict[str, float],
                   structural_weight: float = 0.5) -> dict[str, float]:
    """Combine structural fitness (FEA) with DEM functional fitness.
    structural_weight ∈ [0..1] sets the balance."""
    s = structural.get("composite", 0.0)
    d = dem.get("dem_composite", 0.0)
    return {
        "total": structural_weight * s + (1 - structural_weight) * d,
        "structural": s,
        "dem": d,
        "balance": structural_weight,
    }
