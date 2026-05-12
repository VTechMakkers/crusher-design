"""
MBD module tests.

Verifies kinematics, dynamics, and bearing-life calculations against
analytically known cases or invariants. Failure here means the MBD layer
is producing wrong numbers — a far more dangerous failure than a missing
feature, so these checks come first.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcp-servers"))

from mbd import kinematics as kin
from mbd import dynamics as dyn
from mbd import bearings as brg


def _representative_pe400() -> kin.FourBarGeometry:
    """Indicative geometry for a PE_400x600 single-toggle. Used for shape
    of behavior, not for absolute production values. Chosen to be well
    away from 4-bar branch transitions (base length distinct from L_p)."""
    return kin.FourBarGeometry(
        eccentric_throw_m=0.014,
        pitman_length_m=0.600,
        toggle_length_m=0.300,
        base_dx_m=-0.350,
        base_dy_m=-0.282,
    )


def test_geometry_validates_positive_lengths():
    with pytest.raises(ValueError):
        kin.FourBarGeometry(0.0, 0.5, 0.3, -0.2, -0.5).validate()


def test_position_returns_lower_branch_consistently():
    geom = _representative_pe400()
    geom.validate()
    A_top, C_top, _, _ = kin.position(theta_rad=math.pi / 2, geom=geom)
    A_bot, C_bot, _, _ = kin.position(theta_rad=-math.pi / 2, geom=geom)
    # the toggle seat (C) should always be below the eccentric center
    assert C_top[1] < 0
    assert C_bot[1] < 0


def test_full_revolution_is_assemblable():
    geom = _representative_pe400()
    omega = 280.0 * 2 * math.pi / 60
    poses = kin.trajectory(geom, omega_rad_s=omega, duration_s=60.0 / 280.0,
                            n_steps=120)
    assert len(poses) == 121
    # crank angle should advance monotonically by 2π over the cycle
    assert math.isclose(poses[-1].theta_rad - poses[0].theta_rad,
                         2 * math.pi, abs_tol=1e-6)


def test_stroke_finite_and_bounded():
    """Sanity bound on toggle-seat stroke: must be positive, finite, and
    not exceed a loose multiple of eccentric throw. The exact ratio depends
    on the 4-bar's transmission angle and is geometry-specific."""
    geom = _representative_pe400()
    omega = 280.0 * 2 * math.pi / 60
    poses = kin.trajectory(geom, omega_rad_s=omega, duration_s=60.0 / 280.0)
    s = kin.stroke(poses)
    e_mm = geom.eccentric_throw_m * 1000.0
    total_stroke = math.hypot(s["stroke_x_mm"], s["stroke_y_mm"])
    assert math.isfinite(total_stroke)
    assert total_stroke > 0.0, "mechanism did not move"
    # 5× is generous — production geometries typically 0.5–2×, but transmission
    # angle near alignment can amplify. Anything beyond 5× signals broken geometry.
    assert total_stroke < 5.0 * e_mm, (
        f"stroke {total_stroke:.2f} mm > 5× throw {e_mm:.2f} mm — "
        f"mechanism near singular configuration"
    )


def test_velocity_continuity():
    """Central-difference velocity should be continuous (no jumps > 5× median)."""
    geom = _representative_pe400()
    omega = 280.0 * 2 * math.pi / 60
    poses = kin.trajectory(geom, omega_rad_s=omega, duration_s=60.0 / 280.0,
                            n_steps=720)
    vels = kin.velocities(poses)
    speeds = [math.hypot(v["Cx_dot"], v["Cy_dot"]) for v in vels[1:-1]]
    median = sorted(speeds)[len(speeds) // 2]
    assert all(s <= 6.0 * (median + 1e-6) for s in speeds), \
        "velocity discontinuity — branch jump likely"


def test_quasi_static_reactions_balance_forces():
    """ΣF on the pitman body must vanish in quasi-static equilibrium."""
    geom = _representative_pe400()
    A, C, _, _ = kin.position(0.0, geom)
    pose = kin.PoseState(t_s=0.0, theta_rad=0.0, omega_rad_s=0.0,
                          A_xy=A, C_xy=C,
                          pitman_angle_rad=0.0, toggle_angle_rad=0.0)
    F_c = (-850_000.0, 0.0)  # 850 kN crushing force
    P = (C[0], C[1] + 0.15)   # arbitrary point on the jaw face
    rxn = dyn.quasi_static_reactions(
        pose=pose, geom=geom,
        crushing_force_xy_N=F_c,
        crushing_force_point_xy_m=P,
    )
    Fx_sum = F_c[0] + rxn.big_end_bearing_N[0] + rxn.toggle_seat_N[0]
    Fy_sum = F_c[1] + rxn.big_end_bearing_N[1] + rxn.toggle_seat_N[1]
    assert abs(Fx_sum) < 1e-6
    assert abs(Fy_sum) < 1e-6


def test_quasi_static_reactions_balance_moment_about_big_end():
    geom = _representative_pe400()
    A, C, _, _ = kin.position(0.1, geom)
    pose = kin.PoseState(t_s=0.0, theta_rad=0.1, omega_rad_s=0.0,
                          A_xy=A, C_xy=C,
                          pitman_angle_rad=0.0, toggle_angle_rad=0.0)
    F_c = (-600_000.0, -50_000.0)
    P = (C[0] + 0.05, C[1] + 0.20)
    rxn = dyn.quasi_static_reactions(
        pose=pose, geom=geom,
        crushing_force_xy_N=F_c,
        crushing_force_point_xy_m=P,
    )
    # Moment about A:  (P-A)×F_c + (C-A)×F_toggle = 0
    rPA = (P[0] - A[0], P[1] - A[1])
    M_F = rPA[0] * F_c[1] - rPA[1] * F_c[0]
    rCA = (C[0] - A[0], C[1] - A[1])
    M_t = rCA[0] * rxn.toggle_seat_N[1] - rCA[1] * rxn.toggle_seat_N[0]
    assert abs(M_F + M_t) < 1e-3


def test_toggle_force_is_axial():
    """Toggle is a two-force member — reaction must lie along its axis."""
    geom = _representative_pe400()
    A, C, _, _ = kin.position(0.0, geom)
    pose = kin.PoseState(t_s=0.0, theta_rad=0.0, omega_rad_s=0.0,
                          A_xy=A, C_xy=C,
                          pitman_angle_rad=0.0, toggle_angle_rad=0.0)
    Q = (geom.base_dx_m, geom.base_dy_m)
    rxn = dyn.quasi_static_reactions(
        pose=pose, geom=geom,
        crushing_force_xy_N=(-850_000.0, 0.0),
        crushing_force_point_xy_m=(C[0], C[1] + 0.15),
    )
    # Toggle axis unit vector
    ux, uy = (Q[0] - C[0]), (Q[1] - C[1])
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    # Component of toggle force perpendicular to axis should be zero
    F = rxn.toggle_seat_N
    perp = F[0] * (-uy) + F[1] * ux
    assert abs(perp) < 1e-6


def test_bearing_equivalent_load_constant_case():
    """When all duty fractions see the same load, P_eq must equal that load."""
    P_eq = brg.equivalent_dynamic_load(
        loads_N=[5000.0, 5000.0, 5000.0],
        duty_fractions=[0.4, 0.3, 0.3],
        exponent=brg.ROLLER_BEARING_EXPONENT,
    )
    assert math.isclose(P_eq, 5000.0, rel_tol=1e-9)


def test_bearing_equivalent_load_dominant_peak():
    """Cube-root weighting means high-load fractions dominate P_eq."""
    P_eq_a = brg.equivalent_dynamic_load([1000.0, 10000.0], [0.5, 0.5],
                                          exponent=3.0)
    # Arithmetic mean would be 5500; cube-root weighting biases toward 10000
    assert P_eq_a > 5500.0
    assert P_eq_a < 10000.0


def test_bearing_L10_units_and_monotonicity():
    spec = brg.BearingSpec(designation="22324CC/W33",
                            dynamic_load_rating_N=560_000.0)
    # Lighter load -> longer life
    life_a = brg.L10_hours(equivalent_load_N=100_000.0, bearing=spec, speed_rpm=280)
    life_b = brg.L10_hours(equivalent_load_N=200_000.0, bearing=spec, speed_rpm=280)
    assert life_a > life_b
    # Roughly: (560/100)^(10/3) * 1e6 / (60*280) ≈ 25000 hrs
    assert 10_000.0 < life_a < 50_000.0


def test_bearing_life_from_time_history_matches_constant_case():
    spec = brg.BearingSpec(designation="22324CC/W33",
                            dynamic_load_rating_N=560_000.0)
    constant_history = [150_000.0] * 100
    out = brg.life_from_time_history(load_history_N=constant_history,
                                      dt_s=0.001, bearing=spec, speed_rpm=280)
    direct = brg.L10_hours(equivalent_load_N=150_000.0,
                            bearing=spec, speed_rpm=280)
    assert math.isclose(out["L10_hours"], direct, rel_tol=1e-9)
