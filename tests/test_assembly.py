"""
Assembly-level smoke tests.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_assembly_loads_PE_400x600():
    from assembly.crusher_assembly import load
    asm = load("PE_400x600", ROOT)
    assert asm.model == "PE_400x600"
    assert asm.family == "single_toggle_jaw"
    assert "swing_jaw" in asm.parts
    assert "fixed_jaw" in asm.parts
    assert "toggle" in asm.parts


def test_assembly_load_path_from_jaw_to_frame():
    from assembly.crusher_assembly import load
    asm = load("PE_400x600", ROOT)
    paths = asm.load_path("swing_jaw", "main_frame")
    assert len(paths) >= 1
    # at least one path traverses pitman + toggle
    assert any("pitman" in p and "toggle" in p for p in paths)


def test_assembly_consistency_check_passes_for_PE_400x600():
    """Constraints we can evaluate (instances exist) must pass; the rest
    are reported as skipped, never as crashes."""
    from assembly.crusher_assembly import load
    asm = load("PE_400x600", ROOT)
    result = asm.check_consistency()
    assert result["passes"], f"unexpected constraint failures: {result['issues']}"
    # We declared 3 constraints in the YAML. With only the jaw plates having
    # real instances, the bearing-housing constraint is skipped; the two
    # jaw-pitch constraints are evaluated and pass.
    assert len(result["skipped"]) + len(result["issues"]) <= len(asm.constraints)
    # At least the swing-vs-fixed jaw pitch was evaluated (both instances exist):
    evaluated_count = len(asm.constraints) - len(result["skipped"])
    assert evaluated_count >= 1


def test_assembly_materials_includes_seeded_parts():
    from assembly.crusher_assembly import load
    asm = load("PE_400x600", ROOT)
    mats = asm.material_list()
    # swing_jaw + fixed_jaw both Mn13; toggle is AR400
    assert "Mn13" in mats
    assert "AR400" in mats


def test_pareto_front_dominance():
    from loop.pareto import Candidate, front
    candidates = [
        Candidate("a", [1.0, 1.0]),
        Candidate("b", [2.0, 0.5]),   # better on obj0
        Candidate("c", [0.5, 2.0]),   # better on obj1
        Candidate("d", [0.9, 0.9]),   # dominated by a
    ]
    f = front(candidates)
    ids = {c.id for c in f}
    assert "d" not in ids   # dominated
    assert {"a", "b", "c"}.issubset(ids) or {"b", "c"} == ids
    # a is dominated by neither b nor c (trade-offs), so survives
    assert "a" in ids


def test_pareto_rank_levels():
    from loop.pareto import Candidate, rank_all
    candidates = [
        Candidate("a", [2.0, 2.0]),   # dominates b and c
        Candidate("b", [1.0, 1.0]),
        Candidate("c", [0.5, 0.5]),
    ]
    ranks = rank_all(candidates)
    assert ranks["a"] == 0
    assert ranks["b"] == 1
    assert ranks["c"] == 2


def test_system_fitness_aggregation():
    from loop.system_fitness import aggregate
    part_results = {
        "swing_jaw": {
            "mass_kg": 41, "max_stress_MPa": 320, "cost_INR": 22000,
            "material": "Mn13", "wear_exposed": True, "archard_k_relative": 0.35,
            "dem": {"measured_tph": 62, "energy_kWh_per_t": 1.6,
                    "wear_uniformity_score": 0.72},
        },
        "fixed_jaw": {
            "mass_kg": 45, "max_stress_MPa": 310, "cost_INR": 24000,
            "material": "Mn13", "wear_exposed": True, "archard_k_relative": 0.35,
            "dem": {"wear_uniformity_score": 0.70},
        },
        "toggle": {"mass_kg": 42, "max_stress_MPa": 280, "cost_INR": 8000,
                   "wear_exposed": False},
    }
    assembly_targets = {"capacity_tph_range": [16, 65]}
    ore = {"bond_work_index_kWh_per_t": 17.1}
    m = aggregate(part_results=part_results,
                  assembly_targets=assembly_targets, ore=ore)
    assert m.total_weight_kg == 41 + 45 + 42
    assert m.unit_cost_INR == 22000 + 24000 + 8000
    assert 1.4 <= m.energy_kWh_per_t <= 2.0
    assert m.tph_at_design_css > 0
    assert m.wear_part_life_hours > 0
