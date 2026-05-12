"""
Synthetic-data generator for bootstrapping the system surrogate.

We do not yet have a large corpus of real (assembly, KPI) records — to
get one requires running the full FEA + DEM stack thousands of times.
Until that corpus exists, this module fabricates a self-consistent
training set so the surrogate's architecture and training loop can be
validated end-to-end.

The KPIs are derived from features by an explicit physics-flavoured
function defined in `compute_kpis_from_features`. The surrogate learns
to reproduce that function. When real records start to accumulate (from
`templates/<part>/instances/<model>.history.jsonl` and from the eventual
DEM/FEA runs), replace this generator with a real loader — the surrogate
class itself doesn't need changing.

This is bootstrap data, NOT physics. Do not interpret surrogate output
trained on synthetic data as engineering ground truth.
"""
from __future__ import annotations
import math
import random
from dataclasses import replace
from pathlib import Path

from .assembly_features import AssemblyFeatures, extract_for_model

ROOT = Path(__file__).resolve().parents[1]


# Indices into the global_features vector (must match assembly_features.extract)
_GLOBAL = {
    "eccentric_throw_mm": 0,
    "pitman_length_mm": 1,
    "toggle_length_mm": 2,
    "rpm": 3,
    "motor_kW": 4,
}


def _perturb_features(base: AssemblyFeatures, rng: random.Random,
                       *, num_jitter_frac: float = 0.20,
                       global_jitter_frac: float = 0.15) -> AssemblyFeatures:
    """Return a copy of `base` with numerical entries perturbed multiplicatively.
    Categorical (one-hot) entries are preserved exactly."""
    schema = base.schema
    one_hot_len = (len(schema.part_classes) + len(schema.materials) + 1
                   + len(schema.criticality_vocab))
    new_node_features = []
    for row in base.node_features:
        new_row = list(row)
        for i in range(one_hot_len, len(new_row)):
            if new_row[i] != 0.0:
                new_row[i] = new_row[i] * rng.uniform(1.0 - num_jitter_frac,
                                                       1.0 + num_jitter_frac)
        new_node_features.append(new_row)

    new_globals = [g * rng.uniform(1.0 - global_jitter_frac, 1.0 + global_jitter_frac)
                   for g in base.global_features]

    return replace(base,
                   node_features=new_node_features,
                   global_features=new_globals)


def compute_kpis_from_features(features: AssemblyFeatures,
                                 *, crushing_force_N: float) -> dict[str, float]:
    """Deterministic KPI function for synthetic data. Captures dominant
    scaling relationships engineers would expect:
      - mass = sum of node numerical params * scaling
      - tph  ~ eccentric_throw * rpm * f(motor_kW)
      - energy ~ Wi-like constant, modulated by mechanism
      - bearing_L10 ~ (rating / force)^p / rpm
      - wear_part_life ~ inverse of force × node-mass density on wear nodes
      - cost ~ mass × material-cost-weight
    The exact constants are arbitrary — they only need to make a non-trivial
    function that the surrogate must learn to reproduce."""
    schema = features.schema
    one_hot_len = (len(schema.part_classes) + len(schema.materials) + 1
                   + len(schema.criticality_vocab))
    wear_flag_idx = len(schema.part_classes) + len(schema.materials)

    total_mass = 0.0
    wear_mass = 0.0
    for row in features.node_features:
        nums = row[one_hot_len:]
        mass = sum(abs(v) for v in nums) * 0.01
        total_mass += mass
        if row[wear_flag_idx] > 0.5:
            wear_mass += mass

    e_mm = features.global_features[_GLOBAL["eccentric_throw_mm"]]
    rpm = features.global_features[_GLOBAL["rpm"]]
    motor_kW = features.global_features[_GLOBAL["motor_kW"]]

    tph = max(2.0, 0.018 * e_mm * rpm * math.sqrt(max(motor_kW, 1.0)))
    energy = max(0.4, 15.0 * (1.0 + 0.0005 * (rpm - 280.0)))
    # ISO-281-shaped bearing life: assume big-end sees half the input force,
    # bearing rating C = 900 kN (typical for 22324-class spherical roller).
    P_eq = 0.5 * crushing_force_N
    C = 900_000.0
    bearing_L10 = max(20.0, (C / P_eq) ** (10.0 / 3.0) * 1.0e6
                              / (60.0 * max(rpm, 10.0)))
    wear_life = max(200.0, 4.0e7 / (crushing_force_N ** 0.6
                                      * (wear_mass + 1.0)))
    cost = 30.0 * total_mass + 250.0 * wear_mass

    return {
        "tph_at_design_css": tph,
        "total_weight_kg": total_mass,
        "energy_kWh_per_t": energy,
        "wear_part_life_hours": wear_life,
        "unit_cost_INR": cost,
        "bearing_L10_life_hours": bearing_L10,
    }


def generate(*, n_samples: int, model: str = "PE_400x600",
             root: Path = ROOT, seed: int = 42,
             ) -> list[tuple[AssemblyFeatures, dict[str, float]]]:
    """Generate `n_samples` synthetic (features, KPIs) pairs."""
    rng = random.Random(seed)
    base = extract_for_model(model, root=root)
    samples: list[tuple[AssemblyFeatures, dict[str, float]]] = []
    for _ in range(n_samples):
        feats = _perturb_features(base, rng)
        force_N = rng.uniform(400_000.0, 1_200_000.0)
        kpis = compute_kpis_from_features(feats, crushing_force_N=force_N)
        samples.append((feats, kpis))
    return samples


TARGET_KEYS = ["tph_at_design_css", "total_weight_kg", "energy_kWh_per_t",
                "wear_part_life_hours", "unit_cost_INR", "bearing_L10_life_hours"]
