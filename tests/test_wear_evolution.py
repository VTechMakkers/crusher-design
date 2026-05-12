"""
Wear evolution tests.

Every test in this file is gated by **analytical truth** — Archard's
closed-form equation, dimensional identities, conservation laws — not
by agreement with prior code. The wear model has 50+ years of
peer-reviewed engineering theory behind it; we hold the implementation
to that standard.

Tolerances are exact (rel=1e-12) wherever the formula is closed-form;
loosened only where Python floating-point limits dictate.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.wear_evolution import (LifecycleTrajectory, RegionExposure,
                                   WearPair, WearState,
                                   archard_wear_depth,
                                   archard_wear_volume,
                                   simulate_lifecycle)


# ---------------------------------------------------------------------------
# Archard core identities (closed-form)


def test_archard_volume_matches_KNs_over_H_identity():
    """The fundamental Archard equation: V = K · N · s / H."""
    K, N, s, H = 1.5e-4, 2000.0, 0.5, 5.3e9
    V = archard_wear_volume(force_N=N, sliding_m=s,
                              K_dimensionless=K, hardness_Pa=H)
    assert V == pytest.approx(K * N * s / H, rel=1e-12)


def test_archard_depth_matches_KP_s_over_H_on_unit_area():
    """For area = 1 m², depth equals volume."""
    K, N, s, H, A = 1.5e-4, 2000.0, 0.5, 5.3e9, 1.0
    h = archard_wear_depth(force_N=N, sliding_m=s, area_m2=A,
                             K_dimensionless=K, hardness_Pa=H)
    V = archard_wear_volume(force_N=N, sliding_m=s,
                              K_dimensionless=K, hardness_Pa=H)
    assert h == pytest.approx(V, rel=1e-12)


def test_archard_doubling_force_doubles_volume():
    """Linear in force: V(2N) = 2V(N)."""
    base = dict(sliding_m=0.5, K_dimensionless=1e-4, hardness_Pa=5e9)
    V1 = archard_wear_volume(force_N=1000.0, **base)
    V2 = archard_wear_volume(force_N=2000.0, **base)
    assert V2 == pytest.approx(2.0 * V1, rel=1e-12)


def test_archard_doubling_sliding_doubles_volume():
    base = dict(force_N=1000.0, K_dimensionless=1e-4, hardness_Pa=5e9)
    V1 = archard_wear_volume(sliding_m=0.1, **base)
    V2 = archard_wear_volume(sliding_m=0.2, **base)
    assert V2 == pytest.approx(2.0 * V1, rel=1e-12)


def test_archard_doubling_hardness_halves_volume():
    """Inverse in hardness: V(2H) = V(H) / 2."""
    base = dict(force_N=1000.0, sliding_m=0.5, K_dimensionless=1e-4)
    V_soft = archard_wear_volume(hardness_Pa=2.5e9, **base)
    V_hard = archard_wear_volume(hardness_Pa=5.0e9, **base)
    assert V_hard == pytest.approx(0.5 * V_soft, rel=1e-12)


def test_archard_zero_load_or_sliding_gives_zero_wear():
    """Edge case: no load OR no sliding → no wear."""
    base = dict(K_dimensionless=1e-4, hardness_Pa=5e9)
    assert archard_wear_volume(force_N=0.0, sliding_m=0.5, **base) == 0.0
    assert archard_wear_volume(force_N=1000.0, sliding_m=0.0, **base) == 0.0


def test_archard_rejects_negative_inputs():
    with pytest.raises(ValueError):
        archard_wear_volume(force_N=-1.0, sliding_m=1.0,
                              K_dimensionless=1e-4, hardness_Pa=5e9)
    with pytest.raises(ValueError):
        archard_wear_depth(force_N=1.0, sliding_m=1.0, area_m2=0.0,
                             K_dimensionless=1e-4, hardness_Pa=5e9)
    with pytest.raises(ValueError):
        archard_wear_volume(force_N=1.0, sliding_m=1.0,
                              K_dimensionless=1e-4, hardness_Pa=0.0)


# ---------------------------------------------------------------------------
# Pair validation


def test_wear_pair_rejects_K_outside_physical_range():
    with pytest.raises(ValueError, match="K="):
        WearPair(surface_material="Mn13", abrasive="basalt",
                  K_dimensionless=1.0,            # > 1.0e-1 upper bound
                  hardness_Pa=5.3e9).validate()
    with pytest.raises(ValueError, match="K="):
        WearPair(surface_material="Mn13", abrasive="basalt",
                  K_dimensionless=1.0e-9,         # below 1.0e-7 lower bound
                  hardness_Pa=5.3e9).validate()


def test_wear_pair_rate_constant_units():
    """K/H has dimension 1/Pa. Multiplied by Pa × m gives meters."""
    pair = WearPair(surface_material="Mn13", abrasive="basalt",
                     K_dimensionless=1.5e-4, hardness_Pa=5.3e9)
    expected = 1.5e-4 / 5.3e9
    assert pair.wear_rate_constant == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Lifecycle integration


def _example_exposure(name="face", area=0.04, force=2000.0,
                       sliding=0.03, period=0.2):
    """One region's DEM exposure over a 0.2 s duty period (~one crank rev
    at 280 rpm). 4 m² jaw-face area · 2 kN average normal force ·
    30 mm sliding per revolution. Indicative numbers."""
    return RegionExposure(region_name=name, area_m2=area,
                           normal_force_N=force,
                           sliding_distance_m=sliding,
                           duty_period_s=period)


def _example_pair():
    return WearPair(surface_material="Mn13", abrasive="basalt",
                     K_dimensionless=1.5e-4, hardness_Pa=5.3e9)


def test_simulate_lifecycle_zero_hours_gives_zero_wear():
    traj = simulate_lifecycle(
        exposures=[_example_exposure()],
        wear_pair=_example_pair(),
        total_service_hours=1000.0,
        sample_times_hours=[0.0, 1000.0],
    )
    assert traj.states[0].service_hours == 0.0
    assert traj.states[0].peak_depth_mm() == ("face", 0.0)


def test_simulate_lifecycle_depth_grows_linearly_with_time():
    """h(t) = (K · P · s_per_period / H) · n_periods(t) — linear in t."""
    pair = _example_pair()
    exposure = _example_exposure(area=0.04, force=2000.0,
                                   sliding=0.03, period=0.2)
    traj = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=10000.0,
        sample_times_hours=[0.0, 1000.0, 5000.0, 10000.0],
        duty_active_fraction=1.0,
    )
    h_1k = traj.states[1].depth_per_region_m["face"]
    h_5k = traj.states[2].depth_per_region_m["face"]
    h_10k = traj.states[3].depth_per_region_m["face"]
    assert h_5k == pytest.approx(5.0 * h_1k, rel=1e-9)
    assert h_10k == pytest.approx(10.0 * h_1k, rel=1e-9)


def test_simulate_lifecycle_depth_matches_closed_form():
    """Exact closed-form check at 1 service hour:
        h = (K / H) · (N / A) · s_per_period · (3600 s / period_s) · η_duty
    where η is duty_active_fraction."""
    pair = _example_pair()
    A = 0.04
    N = 2000.0
    s_per_period = 0.03
    period_s = 0.2
    eta = 0.85
    exposure = _example_exposure(area=A, force=N, sliding=s_per_period,
                                   period=period_s)
    traj = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=1.0,
        sample_times_hours=[1.0],
        duty_active_fraction=eta,
    )
    n_periods = 1.0 * 3600.0 * eta / period_s
    expected = (pair.K_dimensionless * (N / A) * s_per_period
                 / pair.hardness_Pa) * n_periods
    actual = traj.states[0].depth_per_region_m["face"]
    assert actual == pytest.approx(expected, rel=1e-12)


def test_lifecycle_volume_matches_archard_volumetric_form():
    """Σ h·A across regions over total service should equal the
    Archard volumetric form V = K·N·s/H integrated over the same
    service time. This is the conservation-of-mass invariant — if it
    ever fails, the depth and volume formulations have drifted."""
    pair = _example_pair()
    exposures = [
        _example_exposure(name="upper", area=0.02, force=1500.0,
                           sliding=0.02, period=0.2),
        _example_exposure(name="lower", area=0.05, force=2500.0,
                           sliding=0.04, period=0.2),
    ]
    hours = 5000.0
    traj = simulate_lifecycle(exposures=exposures, wear_pair=pair,
                                total_service_hours=hours,
                                sample_times_hours=[hours],
                                duty_active_fraction=1.0)
    n_periods = hours * 3600.0 / 0.2
    total_depth_volume = sum(
        traj.states[0].depth_per_region_m[r] * area for r, area in
        [("upper", 0.02), ("lower", 0.05)]
    )
    expected_archard_volume = sum(
        archard_wear_volume(force_N=ex.normal_force_N,
                              sliding_m=ex.sliding_distance_m * n_periods,
                              K_dimensionless=pair.K_dimensionless,
                              hardness_Pa=pair.hardness_Pa)
        for ex in exposures
    )
    assert total_depth_volume == pytest.approx(expected_archard_volume,
                                                  rel=1e-12)


def test_lifecycle_duty_fraction_scales_wear_linearly():
    """Halving duty_active_fraction at fixed total_service_hours should
    halve wear (same total time, half active)."""
    pair = _example_pair()
    exposure = _example_exposure()
    traj_full = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=5000.0,
        sample_times_hours=[5000.0],
        duty_active_fraction=1.0,
    )
    traj_half = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=5000.0,
        sample_times_hours=[5000.0],
        duty_active_fraction=0.5,
    )
    h_full = traj_full.states[0].depth_per_region_m["face"]
    h_half = traj_half.states[0].depth_per_region_m["face"]
    assert h_half == pytest.approx(0.5 * h_full, rel=1e-12)


def test_lifecycle_interpolation_is_exact_in_time():
    """Linear time-scaling means linear interpolation between sample
    points is EXACT, not approximate. Verify."""
    pair = _example_pair()
    exposure = _example_exposure()
    traj = simulate_lifecycle(
        exposures=[exposure], wear_pair=pair,
        total_service_hours=10000.0,
        sample_times_hours=[0.0, 10000.0],
    )
    h_5k_interp = traj.state_at_hours(5000.0).depth_per_region_m["face"]
    # The exact answer is 0.5 × h(10000):
    h_10k = traj.states[1].depth_per_region_m["face"]
    assert h_5k_interp == pytest.approx(0.5 * h_10k, rel=1e-12)


def test_lifecycle_rejects_invalid_inputs():
    pair = _example_pair()
    exposure = _example_exposure()
    with pytest.raises(ValueError, match="total_service_hours"):
        simulate_lifecycle(exposures=[exposure], wear_pair=pair,
                            total_service_hours=-1.0)
    with pytest.raises(ValueError, match="duty_active_fraction"):
        simulate_lifecycle(exposures=[exposure], wear_pair=pair,
                            total_service_hours=1000.0,
                            duty_active_fraction=0.0)
    with pytest.raises(ValueError, match="duty_active_fraction"):
        simulate_lifecycle(exposures=[exposure], wear_pair=pair,
                            total_service_hours=1000.0,
                            duty_active_fraction=1.5)
    with pytest.raises(ValueError, match="at least one region"):
        simulate_lifecycle(exposures=[], wear_pair=pair,
                            total_service_hours=1000.0)


def test_wear_state_peak_depth_returns_worst_region():
    state = WearState(service_hours=5000.0,
                       depth_per_region_m={"a": 0.001, "b": 0.003, "c": 0.002})
    name, depth_mm = state.peak_depth_mm()
    assert name == "b"
    assert depth_mm == pytest.approx(3.0, rel=1e-12)


def test_wear_state_total_mass_uses_density():
    """Σ ρ × h × A — verifies the mass aggregation across regions."""
    state = WearState(service_hours=1000.0,
                       depth_per_region_m={"a": 0.001, "b": 0.002})
    areas = {"a": 0.04, "b": 0.05}
    rho = 7870.0     # Mn13 density kg/m³
    expected = rho * (0.001 * 0.04 + 0.002 * 0.05)
    assert state.total_mass_kg(areas, rho) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Failure-mode library structural integrity


def test_failure_modes_yaml_loads_and_is_well_formed():
    """Every failure mode must declare applies_to, severity, rule, and
    citation. The FMEA engine (next phase) depends on this contract."""
    import yaml
    data = yaml.safe_load((ROOT / "knowledge/failure_modes.yaml").read_text())
    modes = data["failure_modes"]
    assert len(modes) >= 15, "expected at least 15 encoded failure modes"

    valid_severities = {"critical", "major", "minor"}
    valid_rule_kinds = {"ratio_threshold", "stress_vs_strength",
                         "wear_lifetime", "contact_pressure", "fatigue_life"}

    for name, entry in modes.items():
        assert "applies_to" in entry, f"{name} missing applies_to"
        assert isinstance(entry["applies_to"], list)
        assert entry["severity"] in valid_severities, f"{name} bad severity"
        assert "rule" in entry, f"{name} missing rule"
        assert entry["rule"]["kind"] in valid_rule_kinds, \
            f"{name} unknown rule kind {entry['rule']['kind']}"
        assert "prevention_rule" in entry
        assert "citation" in entry and entry["citation"], \
            f"{name} lacks citation (every claim must be sourced)"


def test_failure_modes_reference_real_part_classes():
    """applies_to must list actual part classes from catalog/parts.yaml."""
    import yaml
    catalog = yaml.safe_load(
        (ROOT / "catalog/parts.yaml").read_text())["parts"]
    modes = yaml.safe_load(
        (ROOT / "knowledge/failure_modes.yaml").read_text())["failure_modes"]
    valid_parts = set(catalog.keys())
    for name, entry in modes.items():
        for part in entry["applies_to"]:
            assert part in valid_parts, (
                f"{name}.applies_to references unknown part class {part!r}; "
                f"add to catalog/parts.yaml or fix the typo"
            )


def test_wear_coefficients_yaml_loads_and_is_well_formed():
    """Each wear pair must have K, hardness_Pa, and a citation."""
    import yaml
    data = yaml.safe_load(
        (ROOT / "knowledge/wear_coefficients.yaml").read_text())
    pairs = data["wear_pairs"]
    assert len(pairs) >= 5
    for name, entry in pairs.items():
        K = entry["K_dimensionless"]
        H = entry["hardness_Pa"]
        assert 1.0e-7 < K < 1.0e-1, (
            f"{name}: K={K} outside physical range; verify citation"
        )
        assert H > 0
        assert entry["citation"], f"{name} lacks citation"
