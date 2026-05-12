"""
Project Chrono runner tests.

Splits into two layers:
  - dataclass validation + crushing-pulse ramp math + module importability
    (run anywhere — no chrono required)
  - full transient simulation against the closed-form quasi-static solver
    (skip cleanly when Project Chrono's Python bindings are absent)
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcp-servers"))

from mbd.chrono_runner import (CrushingPulse, MechanismMasses,
                                 default_pe400_masses)
from mbd.kinematics import FourBarGeometry


# ---- dataclass validation (no chrono needed) -----------------------------

def test_masses_validate_positive():
    m = default_pe400_masses()
    m.validate()
    assert m.pitman_kg > 0
    assert m.eccentric_inertia_kgm2 > 0


def test_masses_reject_nonpositive():
    with pytest.raises(ValueError, match="pitman_kg"):
        MechanismMasses(
            eccentric_shaft_kg=10.0, eccentric_inertia_kgm2=1.0,
            pitman_kg=0.0,   # invalid
            pitman_Ix_kgm2=1.0, pitman_Iy_kgm2=1.0, pitman_Iz_kgm2=1.0,
            pitman_centroid_offset_from_A_m=(0.0, -0.3),
        ).validate()


def test_masses_reject_negative_inertia():
    with pytest.raises(ValueError, match="pitman_Iy"):
        MechanismMasses(
            eccentric_shaft_kg=10.0, eccentric_inertia_kgm2=1.0,
            pitman_kg=50.0,
            pitman_Ix_kgm2=1.0, pitman_Iy_kgm2=-0.1, pitman_Iz_kgm2=1.0,
            pitman_centroid_offset_from_A_m=(0.0, -0.3),
        ).validate()


# ---- crushing pulse math (no chrono needed) ------------------------------

def test_pulse_zero_outside_arc():
    p = CrushingPulse(
        force_xy_N=(-1000.0, 0.0),
        application_point_local_m=(0.0, -0.3),
        active_arc_rad=(-math.pi / 6, math.pi / 6),
    )
    # Half-way around the cycle — outside the arc
    fx, fy = p.magnitude_at_angle(math.pi)
    assert fx == 0.0 and fy == 0.0


def test_pulse_full_magnitude_at_centre():
    p = CrushingPulse(force_xy_N=(-1000.0, 200.0),
                       application_point_local_m=(0.0, -0.3))
    fx, fy = p.magnitude_at_angle(0.0)
    assert fx == pytest.approx(-1000.0, rel=1e-9)
    assert fy == pytest.approx(200.0, rel=1e-9)


def test_pulse_ramps_smoothly_at_edges():
    """Half-cosine ramp: magnitude at the arc edge should be 0, not the full
    value (continuity is critical for stable time integration)."""
    p = CrushingPulse(
        force_xy_N=(-1000.0, 0.0),
        application_point_local_m=(0.0, -0.3),
        active_arc_rad=(-math.pi / 6, math.pi / 6),
    )
    fx_edge, _ = p.magnitude_at_angle(math.pi / 6)
    fx_just_inside, _ = p.magnitude_at_angle(math.pi / 6 - 0.01)
    assert abs(fx_edge) < 1e-9, "force must be zero at arc edge for continuity"
    assert abs(fx_just_inside) > abs(fx_edge), "force must rise inside arc"


def test_pulse_periodicity():
    """The active arc check uses crank angle mod 2π — should give the same
    force at θ and θ + 2π."""
    p = CrushingPulse(force_xy_N=(-1000.0, 0.0),
                       application_point_local_m=(0.0, -0.3))
    a = p.magnitude_at_angle(0.05)
    b = p.magnitude_at_angle(0.05 + 2 * math.pi)
    c = p.magnitude_at_angle(0.05 - 2 * math.pi)
    assert a == pytest.approx(b)
    assert a == pytest.approx(c)


# ---- importability of the runner module (no chrono needed) ---------------

def test_runner_module_imports_without_chrono():
    """Critical: the chrono_runner module must be importable so callers
    can introspect dataclasses + pulse logic even when Project Chrono
    isn't installed. Only `simulate()` should raise."""
    from mbd import chrono_runner
    assert chrono_runner.CrushingPulse is not None
    assert chrono_runner.MechanismMasses is not None
    assert chrono_runner.SimulationResult is not None


def test_simulate_raises_clean_import_error_without_chrono():
    """If chrono isn't installed, simulate() must raise a clear ImportError
    naming the install path — not crash with AttributeError or fail
    silently with placeholder output."""
    from mbd.chrono_runner import simulate
    geom = FourBarGeometry(0.014, 0.600, 0.300, -0.350, -0.282)
    masses = default_pe400_masses()
    pulse = CrushingPulse(force_xy_N=(-1000.0, 0.0),
                           application_point_local_m=(0.0, -0.3))
    try:
        import pychrono  # type: ignore[import-not-found]
        has_chrono = hasattr(pychrono, "ChSystemNSC") or hasattr(pychrono, "core")
    except ImportError:
        has_chrono = False
    if has_chrono:
        pytest.skip("Project Chrono is installed — see test_chrono_simulate_runs")
    with pytest.raises(ImportError, match="Project Chrono|projectchrono"):
        simulate(geometry=geom, masses=masses, rpm=280.0, crushing=pulse,
                  duration_s=0.01, timestep_s=1e-4)


# ---- full transient simulation (chrono required) -------------------------

def _require_chrono():
    try:
        import pychrono as pc  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("Project Chrono Python bindings not installed")
    if not (hasattr(pc, "ChSystemNSC") or hasattr(pc, "core")):
        pytest.skip("imported 'pychrono' is not Project Chrono "
                     "(pip's pychrono is an unrelated scheduling library); "
                     "install from projectchrono.org")
    return pc


def test_chrono_simulate_one_revolution():
    """Smoke test: the simulator runs one full revolution without raising
    and returns a SimulationResult with non-empty time-history."""
    _require_chrono()
    from mbd.chrono_runner import simulate
    geom = FourBarGeometry(0.014, 0.600, 0.300, -0.350, -0.282)
    masses = default_pe400_masses()
    pulse = CrushingPulse(
        force_xy_N=(-200_000.0, 0.0),
        application_point_local_m=(0.0, -0.3),
        active_arc_rad=(-math.pi / 6, math.pi / 6),
    )
    result = simulate(geometry=geom, masses=masses, rpm=280.0,
                       crushing=pulse, sample_every=20)
    assert len(result.times_s) > 50
    assert result.crank_angle_rad[-1] > 2.0 * math.pi - 0.1
    summary = result.summarise()
    assert summary["big_end_peak_N"] > 0


def test_chrono_low_speed_approximates_quasi_static():
    """At low rpm, inertial forces are small and Chrono's peak reactions
    should be within ±25% of the closed-form quasi-static result on the
    big-end joint (loose tolerance — accounts for transient ramp + the
    half-cosine pulse shape vs the square step the quasi-static uses)."""
    _require_chrono()
    from mbd.chrono_runner import simulate
    from mbd.dynamics import reactions_over_cycle, summarise_reactions
    from mbd import kinematics as kin

    geom = FourBarGeometry(0.014, 0.600, 0.300, -0.350, -0.282)
    masses = default_pe400_masses()
    rpm_slow = 50.0          # ω² scales inertial loads — slow ω = quasi-static regime
    F = 200_000.0
    pulse = CrushingPulse(
        force_xy_N=(-F, 0.0),
        application_point_local_m=(0.0, -0.3),
        active_arc_rad=(-math.pi / 6, math.pi / 6),
    )

    dyn = simulate(geometry=geom, masses=masses, rpm=rpm_slow,
                    crushing=pulse, sample_every=10)
    dyn_summary = dyn.summarise()

    omega = rpm_slow * 2.0 * math.pi / 60.0
    poses = kin.trajectory(geom, omega_rad_s=omega,
                            duration_s=60.0 / rpm_slow, n_steps=720)
    qs = reactions_over_cycle(
        poses=poses, geom=geom,
        crushing_force_xy_N=(-F, 0.0),
        crushing_force_point_xy_m=(0.0, -0.3),
        crushing_active_arc_rad=(-math.pi / 6, math.pi / 6),
    )
    qs_summary = summarise_reactions(qs)

    ratio = dyn_summary["big_end_peak_N"] / qs_summary["big_end_peak_N"]
    assert 0.75 < ratio < 1.30, (
        f"dynamic/quasi-static ratio {ratio:.2f} at low rpm should be near 1.0"
    )
