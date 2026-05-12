"""
Integration tests for the MBD -> FEA bridge.

These verify that the kinematically-derived joint reactions flow correctly
into the design loop as FEA load cases — the connection that ties the
mechanism layer to the structural layer.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcp-servers"))

from loop.mbd_to_fea import (
    derive_load_case,
    derive_load_case_for_model,
    load_mechanism_from_assembly,
)
from mbd.kinematics import FourBarGeometry


def _representative_geometry() -> FourBarGeometry:
    return FourBarGeometry(
        eccentric_throw_m=0.014,
        pitman_length_m=0.600,
        toggle_length_m=0.300,
        base_dx_m=-0.350,
        base_dy_m=-0.282,
    )


def test_toggle_plate_load_case_has_seat_force():
    case = derive_load_case(
        part_class="toggle_plate",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=850_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    assert case["source"] == "mbd_derived"
    assert case["name"] == "mbd_peak_compression"
    assert "seat_force_kN" in case
    assert case["seat_force_kN"] > 0
    assert case["seat_force_kN"] < case["input_crushing_force_kN"], (
        "toggle reaction should be less than input crushing force "
        "(mechanism partitions load between toggle and big-end)"
    )
    assert case["seat_force_rms_kN"] <= case["seat_force_kN"]


def test_jaw_plate_load_case_passes_crushing_force_through():
    case = derive_load_case(
        part_class="swing_jaw_plate",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=850_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    assert case["crushing_force_kN"] == 850.0
    assert case["direction"] == "normal_to_face"


def test_pitman_load_case_includes_both_reactions():
    case = derive_load_case(
        part_class="pitman",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=850_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    assert case["big_end_force_kN"] > 0
    assert case["toggle_seat_force_kN"] > 0
    # the two reactions together should be on the same order as the input
    assert (case["big_end_force_kN"] + case["toggle_seat_force_kN"]) > \
           0.5 * case["input_crushing_force_kN"]


def test_main_frame_load_case_includes_toggle_rear_and_bearings():
    case = derive_load_case(
        part_class="main_frame",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=850_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    assert case["toggle_rear_force_kN"] > 0
    assert case["bearing_force_per_housing_kN"] > 0


def test_eccentric_shaft_marks_rotating_bending():
    case = derive_load_case(
        part_class="eccentric_shaft",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=850_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    assert case["name"] == "mbd_rotating_bending"
    assert case["rotation_rpm"] == 280.0
    assert "rotating-bending" in case["note"].lower()


def test_unknown_part_raises():
    with pytest.raises(ValueError, match="no MBD->FEA mapping"):
        derive_load_case(
            part_class="not_a_real_part",
            mechanism_geometry=_representative_geometry(),
            rpm=280.0,
            peak_crushing_force_N=100_000.0,
            jaw_face_point_xy_mm=(0.0, -300.0),
        )


def test_load_mechanism_from_assembly_uses_yaml_block():
    geom, rpm = load_mechanism_from_assembly("PE_400x600", root=ROOT)
    assert math.isclose(geom.eccentric_throw_m, 0.014)
    assert math.isclose(geom.pitman_length_m, 0.600)
    assert rpm == 280.0


def test_derive_for_model_runs_end_to_end():
    case = derive_load_case_for_model(
        part_class="toggle_plate",
        model="PE_400x600",
        peak_crushing_force_N=850_000.0,
        root=ROOT,
    )
    assert "seat_force_kN" in case
    assert case["seat_force_kN"] > 100.0   # at 850 kN input, expect O(100s) kN


def test_design_loop_uses_mbd_override():
    """When override_load_case is provided, design_loop.evaluate must use it
    rather than the default from load_cases.yaml."""
    from loop import design_loop
    override = derive_load_case_for_model(
        part_class="toggle_plate",
        model="PE_400x600",
        peak_crushing_force_N=850_000.0,
        root=ROOT,
    )
    rec = design_loop.evaluate(
        part="toggle_plate", model="PE_400x600",
        params={"length_mm": 540, "width_mm": 180, "thickness_mm": 28,
                "seat_radius_mm": 35, "web_height_mm": 60, "web_thickness_mm": 12,
                "relief_groove_depth_mm": 6, "relief_groove_width_mm": 8},
        override_load_case=override,
    )
    # The dry-run path picks up seat_force_kN from override; metrics depend on it
    assert rec["load_case"] == "mbd_peak_compression"
    assert rec["metrics"]["max_von_mises_MPa"] > 0


def test_load_case_scaling_with_crushing_force():
    """Linearity check: doubling input force should roughly double reactions."""
    case_a = derive_load_case(
        part_class="toggle_plate",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=400_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    case_b = derive_load_case(
        part_class="toggle_plate",
        mechanism_geometry=_representative_geometry(),
        rpm=280.0,
        peak_crushing_force_N=800_000.0,
        jaw_face_point_xy_mm=(0.0, -300.0),
    )
    ratio = case_b["seat_force_kN"] / case_a["seat_force_kN"]
    assert math.isclose(ratio, 2.0, rel_tol=0.05), \
        f"linearity broken: doubled input gave {ratio:.3f}× output"
