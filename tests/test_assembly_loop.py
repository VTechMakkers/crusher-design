"""
Assembly-loop tests.

Verifies the whole-machine driver:
  - baseline assembly produces a single valid whole crusher
  - cross-part constraints (paired jaw pitch) actually filter combinations
  - Pareto rank places non-dominated variants in rank 0
  - top_k selection picks at most K from the front
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.assembly_loop import (run_assembly, sweep_part,
                                 constraints_satisfied)
from assembly.crusher_assembly import load as load_assembly


def test_baseline_assembly_produces_one_valid_variant():
    """No sweep: every part uses its baseline params; exactly one whole
    crusher results, no constraints violated."""
    result = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=850_000.0,
        sweep_config=None, per_part_top_k=1, top_k=1,
    )
    assert result["n_valid_assemblies"] == 1
    assert result["pareto_front_size"] == 1
    front = result["pareto_front"][0]
    assert front["model"] == "PE_400x600"
    assert front["pareto_rank"] == 0
    metrics = front["system_metrics"]
    assert metrics["total_weight_kg"] > 0
    assert metrics["unit_cost_INR"] > 0
    assert metrics["bearing_L10_life_hours"] > 0


def test_per_part_sweep_top_k_caps_variants():
    """sweep_part returns at most top_k_per_part variants."""
    asm = load_assembly("PE_400x600", ROOT)
    variants = sweep_part(
        assembly=asm, node_id="swing_jaw",
        sweep_params={"tooth_pitch_mm": [80, 85, 90, 95, 100]},
        use_mbd=False, peak_crushing_force_N=850_000.0,
        top_k_per_part=2, root=ROOT,
    )
    assert len(variants) <= 2
    assert all(v.has_geometry for v in variants)


def test_constraints_filter_mismatched_jaw_pitch():
    """Sweep both jaws on tooth_pitch. Only combinations where swing and
    fixed share the same pitch survive the constraint filter."""
    result = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=850_000.0,
        sweep_config={
            "swing_jaw_plate": {"tooth_pitch_mm": [80, 90, 100]},
            "fixed_jaw_plate": {"tooth_pitch_mm": [80, 90, 100]},
        },
        per_part_top_k=10, top_k=10, use_mbd=False,
    )
    # 3×3 = 9 raw combinations; constraint allows only matching pitches → 3 valid
    assert result["n_valid_assemblies"] == 3
    for variant in result["all_valid_variants"]:
        swing_p = variant["part_variants"]["swing_jaw"]["params"]["tooth_pitch_mm"]
        fixed_p = variant["part_variants"]["fixed_jaw"]["params"]["tooth_pitch_mm"]
        assert swing_p == fixed_p, (
            f"constraint violation slipped through: swing={swing_p}, fixed={fixed_p}"
        )


def test_constraints_satisfied_helper_direct():
    asm = load_assembly("PE_400x600", ROOT)
    matching = {
        "swing_jaw": {"tooth_pitch_mm": 90, "width_mm": 400},
        "fixed_jaw": {"tooth_pitch_mm": 90, "width_mm": 400},
    }
    mismatched = {
        "swing_jaw": {"tooth_pitch_mm": 90, "width_mm": 400},
        "fixed_jaw": {"tooth_pitch_mm": 85, "width_mm": 400},
    }
    assert constraints_satisfied(asm, matching)
    assert not constraints_satisfied(asm, mismatched)


def test_pareto_front_is_non_dominated():
    """No member of the returned Pareto front should dominate any other
    member on every objective simultaneously."""
    result = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=850_000.0,
        sweep_config={
            "swing_jaw_plate": {"tooth_depth_mm": [18, 22, 26]},
            "toggle_plate": {"thickness_mm": [24, 28, 32]},
        },
        per_part_top_k=5, top_k=5, use_mbd=False,
    )
    front = result["pareto_front"]
    assert len(front) >= 1
    # All members of the returned front must have pareto_rank == 0
    for v in front:
        assert v["pareto_rank"] == 0


def test_top_k_caps_pareto_front_size():
    """Requesting top_k=2 must return at most 2 even if more are Pareto-optimal."""
    result = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=850_000.0,
        sweep_config={
            "swing_jaw_plate": {"tooth_depth_mm": [18, 22, 26],
                                  "tooth_pitch_mm": [85, 90, 95]},
        },
        per_part_top_k=10, top_k=2, use_mbd=False,
    )
    assert len(result["pareto_front"]) <= 2


def test_cli_runs_end_to_end():
    """The CLI driver runs and emits valid JSON with the expected shape."""
    proc = subprocess.run(
        [sys.executable, "bin/run_assembly.py", "PE_400x600",
         "--ore", "basalt", "--per-part-top-k", "1", "--top-k", "1",
         "--no-mbd"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["model"] == "PE_400x600"
    assert "pareto_front" in out
    assert "n_valid_assemblies" in out


def test_force_input_affects_system_kpis():
    """Higher crushing force → lower bearing life. Verifies the MBD-derived
    load actually flows through to the system aggregation."""
    low_force = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=400_000.0,
        per_part_top_k=1, top_k=1, use_mbd=True,
    )
    high_force = run_assembly(
        model="PE_400x600", ore="basalt",
        peak_crushing_force_N=1_200_000.0,
        per_part_top_k=1, top_k=1, use_mbd=True,
    )
    low_bearing = low_force["pareto_front"][0]["system_metrics"]["bearing_L10_life_hours"]
    high_bearing = high_force["pareto_front"][0]["system_metrics"]["bearing_L10_life_hours"]
    # Higher force → higher stress → lower L10 (the system aggregator derives
    # bearing life from average part stress)
    assert low_bearing > high_bearing, (
        f"bearing life should drop with higher force: low={low_bearing}, high={high_bearing}"
    )
