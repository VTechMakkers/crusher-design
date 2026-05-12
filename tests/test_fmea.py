"""
FMEA rule engine tests.

Every assertion gated by analytical truth: standard FAT-class S-N
curves, Shigley's fatigue simplification, ratio arithmetic,
stress-vs-strength definitions. None of the tests check "my code
agrees with my old code" — they check "my code agrees with the
textbook formula or standard."
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.fmea import (DesignMetrics, FailureModeEvaluation, FMEAReport,
                        evaluate_failure_modes,
                        fat_class_endurance, material_endurance)
from loop.wear_evolution import (RegionExposure, WearPair,
                                   simulate_lifecycle)


# ---------------------------------------------------------------------------
# S-N curve helpers


def test_fat_class_endurance_at_reference_cycles_equals_class_value():
    """FAT class IS the endurance stress at 2·10⁶ cycles by definition.
    A class number that doesn't reproduce itself at N=2e6 means the curve
    formulation is wrong."""
    for fat in (36.0, 71.0, 100.0, 125.0):
        assert fat_class_endurance(f"FAT {fat:.0f}", 2.0e6) == pytest.approx(
            fat, rel=1e-12
        )


def test_fat_class_endurance_higher_at_lower_cycles():
    """S-N curve slope m=3: at 2·10⁵ cycles, endurance should be FAT × 10^(1/3)
    ≈ FAT × 2.154 — much higher than at the reference cycles."""
    σ_short_life = fat_class_endurance("FAT 71", 2.0e5)
    expected = 71.0 * (2.0e6 / 2.0e5) ** (1.0 / 3.0)
    assert σ_short_life == pytest.approx(expected, rel=1e-12)
    assert σ_short_life > 71.0


def test_fat_class_endurance_lower_at_higher_cycles():
    """At 1·10⁷ (past the slope transition), the second-segment slope m=5
    governs."""
    σ_long_life = fat_class_endurance("FAT 71", 1.0e7)
    σ_transition = 71.0 * (2.0e6 / 5.0e6) ** (1.0 / 3.0)
    expected = σ_transition * (5.0e6 / 1.0e7) ** (1.0 / 5.0)
    assert σ_long_life == pytest.approx(expected, rel=1e-12)


def test_fat_class_endurance_caps_at_cafl():
    """Beyond 10⁸ cycles → constant amplitude fatigue limit.  Endurance
    becomes constant (no further decline)."""
    σ_1e8 = fat_class_endurance("FAT 71", 1.0e8)
    σ_1e9 = fat_class_endurance("FAT 71", 1.0e9)
    σ_1e10 = fat_class_endurance("FAT 71", 1.0e10)
    assert σ_1e9 == pytest.approx(σ_1e8, rel=1e-12)
    assert σ_1e10 == pytest.approx(σ_1e8, rel=1e-12)


def test_fat_class_endurance_rejects_invalid_input():
    with pytest.raises(ValueError):
        fat_class_endurance("FAT -5", 1.0e6)
    with pytest.raises(ValueError):
        fat_class_endurance("FAT 71", -1.0)


# ---------------------------------------------------------------------------
# Material endurance


def test_material_endurance_shigley_simplification_at_1e6():
    """Shigley Eq. 6-8: σ_e' = 0.5 · σ_UTS for UTS ≤ 1400 MPa, at 10⁶ cycles."""
    props = {"ultimate_strength_MPa": 850.0}
    assert material_endurance(props, 1.0e6) == pytest.approx(425.0, rel=1e-12)


def test_material_endurance_plateaus_above_1400_uts():
    """For UTS > 1400 MPa, the Shigley simplification holds the endurance
    constant at 700 MPa (steel material limit)."""
    props_high = {"ultimate_strength_MPa": 2000.0}
    assert material_endurance(props_high, 1.0e6) == pytest.approx(700.0, rel=1e-12)
    props_at_1400 = {"ultimate_strength_MPa": 1400.0}
    assert material_endurance(props_at_1400, 1.0e6) == pytest.approx(700.0, rel=1e-12)


def test_material_endurance_at_1e3_is_09_uts():
    """At 10³ cycles, σ_a starts at 0.9 · σ_UTS (Basquin curve fit)."""
    props = {"ultimate_strength_MPa": 850.0}
    assert material_endurance(props, 1.0e3) == pytest.approx(0.9 * 850.0,
                                                                rel=1e-12)


def test_material_endurance_monotone_decreasing_in_N():
    """Between 10³ and 10⁶, endurance must decrease monotonically with N."""
    props = {"ultimate_strength_MPa": 850.0}
    values = [material_endurance(props, 10 ** k)
              for k in (3, 4, 5, 6)]
    for prev, nxt in zip(values, values[1:]):
        assert prev > nxt


# ---------------------------------------------------------------------------
# ratio_threshold rule


def test_ratio_threshold_above_is_failure_passes():
    """Bearing L10 rule: P_eq / C_dynamic = 0.15 < threshold 0.20 → safe."""
    fms = {"test_mode": {
        "applies_to": ["bearing_housing"], "severity": "major",
        "rule": {"kind": "ratio_threshold",
                  "numerator": "P_eq_N", "denominator": "C_dynamic_N",
                  "threshold": 0.20, "direction": "above_is_failure"},
        "consequence": "x", "prevention_rule": "x", "citation": "x",
    }}
    metrics = DesignMetrics(metrics={"P_eq_N": 150.0, "C_dynamic_N": 1000.0})
    report = evaluate_failure_modes(part_class="bearing_housing",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes
    # margin = (0.20 − 0.15) / 0.20 = 0.25
    assert e.margin == pytest.approx(0.25, rel=1e-12)


def test_ratio_threshold_above_is_failure_fails():
    fms = {"test_mode": {
        "applies_to": ["bearing_housing"], "severity": "major",
        "rule": {"kind": "ratio_threshold",
                  "numerator": "P_eq_N", "denominator": "C_dynamic_N",
                  "threshold": 0.20, "direction": "above_is_failure"},
        "consequence": "x", "prevention_rule": "x", "citation": "x",
    }}
    metrics = DesignMetrics(metrics={"P_eq_N": 400.0, "C_dynamic_N": 1000.0})
    report = evaluate_failure_modes(part_class="bearing_housing",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert not e.passes
    # margin = (0.20 − 0.40) / 0.20 = −1.0
    assert e.margin == pytest.approx(-1.0, rel=1e-12)


def test_ratio_threshold_accepts_literal_constant_denominator():
    """failure_modes.yaml has `denominator: 100.0` (a constant, not metric)
    for the bolt-loosening rule. The engine must treat that as a literal."""
    fms = {"test_mode": {
        "applies_to": ["x"], "severity": "major",
        "rule": {"kind": "ratio_threshold",
                  "numerator": "preload_loss_pct",
                  "denominator": "100.0",
                  "threshold": 0.30, "direction": "above_is_failure"},
        "consequence": "x", "prevention_rule": "x", "citation": "x",
    }}
    metrics = DesignMetrics(metrics={"preload_loss_pct": 20.0})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    # ratio = 0.20, threshold = 0.30, above_is_failure: passes
    assert e.passes
    assert e.evaluation_detail["denominator_value"] == 100.0


def test_ratio_threshold_skips_when_metric_missing():
    """Missing metric → skipped_reason set, not silently wrong."""
    fms = {"x": {"applies_to": ["x"], "severity": "minor",
                 "rule": {"kind": "ratio_threshold", "numerator": "foo",
                          "denominator": "bar", "threshold": 0.5,
                          "direction": "above_is_failure"},
                 "consequence": "", "prevention_rule": "", "citation": ""}}
    metrics = DesignMetrics(metrics={"foo": 1.0})        # missing 'bar'
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes is None
    assert e.margin is None
    assert e.skipped_reason == "missing metric 'bar'"


# ---------------------------------------------------------------------------
# stress_vs_strength rule


def test_stress_vs_strength_safe_with_margin():
    """σ · SF < strength → safe.  σ=200, SF=2.0, strength=500: σ·SF=400 < 500
    → margin = (500-400)/500 = 0.20"""
    fms = {"x": {"applies_to": ["x"], "severity": "critical",
                 "rule": {"kind": "stress_vs_strength",
                          "stress_metric": "σ_MPa",
                          "strength_metric": "σy_MPa",
                          "safety_factor": 2.0,
                          "direction": "above_is_failure"},
                 "consequence": "", "prevention_rule": "", "citation": ""}}
    metrics = DesignMetrics(metrics={"σ_MPa": 200.0, "σy_MPa": 500.0})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes
    assert e.margin == pytest.approx(0.20, rel=1e-12)


def test_stress_vs_strength_at_threshold_gives_zero_margin():
    """σ · SF = strength exactly → margin = 0 (knife edge, still passes)."""
    fms = {"x": {"applies_to": ["x"], "severity": "critical",
                 "rule": {"kind": "stress_vs_strength",
                          "stress_metric": "σ_MPa",
                          "strength_metric": "σy_MPa",
                          "safety_factor": 2.5,
                          "direction": "above_is_failure"},
                 "consequence": "", "prevention_rule": "", "citation": ""}}
    # σ=200, SF=2.5, strength = 200×2.5 = 500
    metrics = DesignMetrics(metrics={"σ_MPa": 200.0, "σy_MPa": 500.0})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes
    assert e.margin == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# wear_lifetime rule


def test_wear_lifetime_passes_when_depth_below_max():
    """Use a real LifecycleTrajectory; assert FMEA correctly queries depth
    at the rule's service_hours."""
    pair = WearPair(surface_material="Mn13", abrasive="basalt",
                     K_dimensionless=1.5e-4, hardness_Pa=5.3e9)
    exposure = RegionExposure(region_name="tooth_crests", area_m2=0.04,
                                normal_force_N=2000.0,
                                sliding_distance_m=0.03,
                                duty_period_s=0.2)
    traj = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=10000.0,
        sample_times_hours=[0.0, 4000.0, 10000.0],
    )

    fms = {"face_wear": {
        "applies_to": ["swing_jaw_plate"], "severity": "minor",
        "rule": {"kind": "wear_lifetime",
                  "region": "tooth_crests",
                  "max_depth_mm": 5.0,
                  "at_service_hours": 4000.0},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(metrics={}, wear_trajectory=traj)
    report = evaluate_failure_modes(part_class="swing_jaw_plate",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes
    actual = e.evaluation_detail["actual_depth_mm"]
    expected_margin = (5.0 - actual) / 5.0
    assert e.margin == pytest.approx(expected_margin, rel=1e-12)


def test_wear_lifetime_skips_without_trajectory():
    """No wear trajectory → skipped, not silently passing."""
    fms = {"x": {"applies_to": ["x"], "severity": "minor",
                 "rule": {"kind": "wear_lifetime",
                          "region": "r", "max_depth_mm": 5.0,
                          "at_service_hours": 1000.0},
                 "consequence": "", "prevention_rule": "", "citation": ""}}
    metrics = DesignMetrics(metrics={})        # no wear_trajectory
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes is None
    assert "wear_trajectory" in e.skipped_reason


# ---------------------------------------------------------------------------
# fatigue_life rule


def test_fatigue_life_fat_class_passes_when_stress_below_endurance():
    """FAT 71 at 2·10⁶ cycles → endurance 71 MPa. Stress amplitude 50 MPa → safe."""
    fms = {"weld_fatigue": {
        "applies_to": ["main_frame"], "severity": "critical",
        "rule": {"kind": "fatigue_life",
                  "stress_amplitude_MPa": "σ_a",
                  "target_cycles": 2.0e6,
                  "detail_category": "FAT 71",
                  "material_S_N_curve_required": True},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(metrics={"σ_a": 50.0})
    report = evaluate_failure_modes(part_class="main_frame",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes
    assert e.evaluation_detail["endurance_MPa"] == pytest.approx(71.0, rel=1e-12)
    assert e.margin == pytest.approx((71.0 - 50.0) / 71.0, rel=1e-12)


def test_fatigue_life_material_based_lookup_uses_uts():
    """Material-based fatigue: shaft material has UTS in props, target 10⁶."""
    fms = {"shaft_fatigue": {
        "applies_to": ["eccentric_shaft"], "severity": "critical",
        "rule": {"kind": "fatigue_life",
                  "stress_amplitude_MPa": "σ_a",
                  "target_cycles": 1.0e6,
                  "material_S_N_curve_required": True},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(
        metrics={"σ_a": 200.0},
        materials={"eccentric_shaft": "TestSteel"},
        material_props={"TestSteel": {"ultimate_strength_MPa": 800.0}},
    )
    report = evaluate_failure_modes(part_class="eccentric_shaft",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    # σ_e' = 0.5 × 800 = 400 MPa at 10⁶ cycles
    assert e.evaluation_detail["endurance_MPa"] == pytest.approx(400.0, rel=1e-12)
    assert e.margin == pytest.approx((400.0 - 200.0) / 400.0, rel=1e-12)


def test_fatigue_life_material_based_skips_without_material_decl():
    fms = {"x": {"applies_to": ["pitman"], "severity": "critical",
                 "rule": {"kind": "fatigue_life",
                          "stress_amplitude_MPa": "σ_a",
                          "target_cycles": 1.0e6,
                          "material_S_N_curve_required": True},
                 "consequence": "", "prevention_rule": "", "citation": ""}}
    metrics = DesignMetrics(metrics={"σ_a": 100.0})    # no material declared
    report = evaluate_failure_modes(part_class="pitman", design_metrics=metrics,
                                      failure_modes=fms)
    e = report.evaluations[0]
    assert e.passes is None
    assert "material" in e.skipped_reason


# ---------------------------------------------------------------------------
# Report query API


def test_passes_all_only_counts_evaluable_modes():
    """If a mode is skipped, it must not affect passes_all() — passes_all
    is about evaluable verdicts only."""
    fms = {
        "evaluable_pass": {
            "applies_to": ["x"], "severity": "minor",
            "rule": {"kind": "ratio_threshold", "numerator": "a",
                      "denominator": "b", "threshold": 0.5,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
        "skipped": {
            "applies_to": ["x"], "severity": "critical",
            "rule": {"kind": "ratio_threshold", "numerator": "missing",
                      "denominator": "b", "threshold": 0.5,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
    }
    metrics = DesignMetrics(metrics={"a": 0.1, "b": 1.0})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    assert report.passes_all()
    assert len(report.evaluable()) == 1
    assert len(report.unable_to_evaluate()) == 1


def test_critical_failures_filter():
    fms = {
        "minor_fail": {
            "applies_to": ["x"], "severity": "minor",
            "rule": {"kind": "ratio_threshold", "numerator": "a",
                      "denominator": "b", "threshold": 0.1,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
        "critical_fail": {
            "applies_to": ["x"], "severity": "critical",
            "rule": {"kind": "ratio_threshold", "numerator": "a",
                      "denominator": "b", "threshold": 0.1,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
    }
    metrics = DesignMetrics(metrics={"a": 0.5, "b": 1.0})    # ratio 0.5 > 0.1 → both fail
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    assert len(report.failing()) == 2
    crits = report.critical_failures()
    assert len(crits) == 1
    assert crits[0].failure_mode == "critical_fail"


def test_ranked_by_risk_orders_critical_first():
    """Severity comes first, smallest margin within severity."""
    fms = {
        "minor_close": {
            "applies_to": ["x"], "severity": "minor",
            "rule": {"kind": "ratio_threshold", "numerator": "a",
                      "denominator": "b", "threshold": 0.5,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
        "critical_safe": {
            "applies_to": ["x"], "severity": "critical",
            "rule": {"kind": "ratio_threshold", "numerator": "c",
                      "denominator": "b", "threshold": 0.5,
                      "direction": "above_is_failure"},
            "consequence": "", "prevention_rule": "", "citation": "",
        },
    }
    metrics = DesignMetrics(metrics={"a": 0.45, "b": 1.0, "c": 0.1})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    ranked = report.ranked_by_risk()
    assert ranked[0].failure_mode == "critical_safe"
    assert ranked[1].failure_mode == "minor_close"


def test_applies_to_discrimination():
    """A failure mode for toggle_plate must not fire on swing_jaw_plate."""
    fms = {"toggle_only": {
        "applies_to": ["toggle_plate"], "severity": "critical",
        "rule": {"kind": "ratio_threshold", "numerator": "a",
                  "denominator": "b", "threshold": 0.5,
                  "direction": "above_is_failure"},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(metrics={"a": 0.1, "b": 1.0})
    report = evaluate_failure_modes(part_class="swing_jaw_plate",
                                      design_metrics=metrics,
                                      failure_modes=fms)
    assert report.evaluations == []     # no rule applies → empty report


def test_metric_aliases_route_to_actual_metric_name():
    """Alias 'peak_compressive_stress_MPa' → 'max_von_mises_MPa'.
    Engine resolves the alias and evaluates."""
    fms = {"buckling": {
        "applies_to": ["toggle_plate"], "severity": "critical",
        "rule": {"kind": "ratio_threshold",
                  "numerator": "peak_compressive_stress_MPa",
                  "denominator": "critical_buckling_stress_MPa",
                  "threshold": 0.5, "direction": "above_is_failure"},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(metrics={
        "max_von_mises_MPa": 100.0,
        "critical_buckling_stress_MPa": 400.0,
    })
    report = evaluate_failure_modes(
        part_class="toggle_plate", design_metrics=metrics,
        failure_modes=fms,
        metric_aliases={"peak_compressive_stress_MPa": "max_von_mises_MPa"},
    )
    e = report.evaluations[0]
    assert e.passes
    # ratio = 100 / 400 = 0.25, threshold 0.5: margin = 0.5
    assert e.margin == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# Real failure_modes.yaml integration


def test_real_yaml_failure_modes_load_and_dispatch():
    """Engine accepts every rule kind in the on-disk failure_modes.yaml,
    routing to the correct evaluator (never falls through to 'unknown rule
    kind')."""
    from loop.fmea import _load_failure_modes, _RULE_EVALUATORS
    fms = _load_failure_modes(ROOT)
    used_kinds = {entry["rule"]["kind"] for entry in fms.values()}
    unsupported = used_kinds - set(_RULE_EVALUATORS)
    assert not unsupported, (
        f"failure_modes.yaml uses rule kinds {unsupported} that the engine "
        f"doesn't implement"
    )


def test_real_yaml_evaluates_some_toggle_plate_modes_when_metrics_given():
    """Feeding a plausible metric set for toggle_plate should evaluate the
    modes whose metric names match, and skip the ones whose names don't
    (with skipped_reason). At least one evaluable mode is required."""
    metrics = DesignMetrics(
        metrics={
            "peak_compressive_stress_MPa": 200.0,
            "critical_buckling_stress_MPa": 800.0,
            "shear_stress_at_relief_groove_MPa": 250.0,
            "shear_yield_MPa": 300.0,
        },
        materials={"toggle_plate": "AR400"},
    )
    report = evaluate_failure_modes(part_class="toggle_plate",
                                      design_metrics=metrics)
    assert report.evaluations, "no toggle_plate failure modes in YAML?"
    assert report.evaluable(), "every evaluation got skipped"


def test_summary_lines_render_cleanly():
    """The text report should be free of None placeholders and include
    severity + margin columns."""
    fms = {"m": {
        "applies_to": ["x"], "severity": "major",
        "rule": {"kind": "ratio_threshold", "numerator": "a",
                  "denominator": "b", "threshold": 0.5,
                  "direction": "above_is_failure"},
        "consequence": "", "prevention_rule": "", "citation": "",
    }}
    metrics = DesignMetrics(metrics={"a": 0.2, "b": 1.0})
    report = evaluate_failure_modes(part_class="x", design_metrics=metrics,
                                      failure_modes=fms)
    text = "\n".join(report.summary_lines())
    assert "FMEA report" in text
    assert "major" in text
    assert "None" not in text
