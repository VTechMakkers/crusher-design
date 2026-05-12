"""
Assembly-level (system) fitness.

Aggregates part-level metrics + cross-part interactions into system KPIs:
  - tph_at_design_css       (from DEM + capacity model)
  - total_weight_kg         (sum of part masses)
  - energy_kWh_per_t        (from DEM energy + motor efficiency)
  - wear_part_life_hours    (min of wear-exposed parts' lives)
  - unit_cost_INR           (material + labor + procured)
  - bearing_L10_life_hours  (derived from FEA reactions × duty)

These are the dimensions of the Pareto front. A single assembly variant
yields one point in this 6-D objective space; Pareto search finds the
non-dominated frontier.

WIRING NOTE
-----------
This module is the aggregator for assembly-level (whole-machine) design
loops. It's exercised by `tests/test_assembly.py` to verify the
aggregation math. It is NOT yet wired into `bin/run_design.py`, which
operates on a single part at a time. The hook-up lands when an
`--assembly-mode` driver is added — that driver runs the inner part-level
loop for every node in the assembly graph, collects results, then calls
`aggregate(...)` here to produce a system-level KPI point for Pareto
search.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class SystemMetrics:
    tph_at_design_css: float
    total_weight_kg: float
    energy_kWh_per_t: float
    wear_part_life_hours: float
    unit_cost_INR: float
    bearing_L10_life_hours: float

    def to_objective_vector(self, signs: dict[str, str]) -> list[float]:
        """Return a vector where higher = better (flip min objectives)."""
        out: list[float] = []
        for key in ("tph_at_design_css", "total_weight_kg", "energy_kWh_per_t",
                    "wear_part_life_hours", "unit_cost_INR", "bearing_L10_life_hours"):
            v = getattr(self, key)
            sign = signs.get(key, "maximize")
            out.append(v if sign == "maximize" else -v)
        return out

    def as_dict(self) -> dict[str, float]:
        return {
            "tph_at_design_css": self.tph_at_design_css,
            "total_weight_kg": self.total_weight_kg,
            "energy_kWh_per_t": self.energy_kWh_per_t,
            "wear_part_life_hours": self.wear_part_life_hours,
            "unit_cost_INR": self.unit_cost_INR,
            "bearing_L10_life_hours": self.bearing_L10_life_hours,
        }


def aggregate(*, part_results: dict[str, dict[str, Any]],
              assembly_targets: dict[str, Any],
              ore: dict[str, Any]) -> SystemMetrics:
    """Aggregate part-level results into system KPIs.

    `part_results` map: node_id -> {mass_kg, max_stress_MPa, dem_metrics?, material, cost_INR}
    `assembly_targets` is the system_targets block from the assembly YAML.
    `ore` is the ore block (Bond Wi, density, ...) for energy calc.
    """
    total_mass = sum(r.get("mass_kg", 0.0) for r in part_results.values())
    total_cost = sum(r.get("cost_INR", 0.0) for r in part_results.values())

    # TPH from DEM if any wear part has it; else fall back to handbook from targets
    tph_candidates = []
    for nid, r in part_results.items():
        dem = r.get("dem") or {}
        if "measured_tph" in dem:
            tph_candidates.append(dem["measured_tph"])
    if tph_candidates:
        tph = sum(tph_candidates) / len(tph_candidates)
    else:
        low, high = assembly_targets.get("capacity_tph_range", [10, 50])
        tph = (low + high) / 2.0

    # Energy: prefer DEM-derived, else Bond
    energy_candidates = []
    for r in part_results.values():
        dem = r.get("dem") or {}
        if "energy_kWh_per_t" in dem:
            energy_candidates.append(dem["energy_kWh_per_t"])
    if energy_candidates:
        energy = sum(energy_candidates) / len(energy_candidates)
    else:
        wi = ore.get("bond_work_index_kWh_per_t", 14.0)
        energy = wi * 0.7

    # Wear-part service life: min across wear-exposed parts (placeholder model)
    wear_lives: list[float] = []
    for nid, r in part_results.items():
        if not r.get("wear_exposed"):
            continue
        archard_k = r.get("archard_k_relative", 1.0)
        contact_factor = (r.get("dem") or {}).get("wear_uniformity_score", 0.5)
        # crude model: life ~ inverse(k * (1 - uniformity_bonus))
        life = 4000.0 / max(archard_k, 0.01) * (0.6 + 0.8 * contact_factor)
        wear_lives.append(life)
    wear_life = min(wear_lives) if wear_lives else 8000.0

    # Bearing L10: derived from FEA reactions × duty. Placeholder until MBD exists.
    avg_stress = sum(r.get("max_stress_MPa", 0.0) for r in part_results.values()) \
                 / max(len(part_results), 1)
    bearing_L10 = 50000.0 * (300.0 / max(avg_stress, 50.0))  # crude

    return SystemMetrics(
        tph_at_design_css=tph,
        total_weight_kg=total_mass,
        energy_kWh_per_t=energy,
        wear_part_life_hours=wear_life,
        unit_cost_INR=total_cost,
        bearing_L10_life_hours=bearing_L10,
    )
