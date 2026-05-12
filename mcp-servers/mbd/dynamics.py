"""
Quasi-static dynamics of the single-toggle mechanism.

Given an external crushing force F_jaw applied to the swing jaw face at a
known crank angle, find the joint reaction forces and the required crank
torque. Inertial effects of the pitman + jaw assembly are added on top via
a separate inertia pass.

Assumptions and their consequences:
  - Quasi-static: the crusher operates near 4–6 Hz; inertial forces of the
    pitman are non-negligible at peak operation. We compute quasi-static
    first (dominant at peak crushing) and add inertial reactions separately.
  - The pitman + swing jaw are treated as a single rigid body whose mass is
    lumped at the swing jaw centroid. Real designs split this further.
  - The toggle is a two-force member (forces only along its axis) — true for
    a slender, pin-jointed toggle plate at both ends.
  - Friction in pin joints is neglected (acceptable for journal bearings
    well-lubricated; would otherwise add 2–5% to crank torque).

All forces in newtons, torques in N·m, all lengths in meters.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from .kinematics import PoseState, FourBarGeometry


@dataclass(frozen=True)
class JointReactions:
    """Forces (N) acting on each named joint at one instant.

    Convention: forces are reported in the GLOBAL XY frame, acting ON the
    pitman (so equal-and-opposite act on the conjugate body)."""
    t_s: float
    big_end_bearing_N: tuple[float, float]
    toggle_seat_N: tuple[float, float]
    toggle_rear_N: tuple[float, float]
    crank_torque_Nm: float


def two_force_member_unit(C: tuple[float, float],
                          Q: tuple[float, float]) -> tuple[float, float]:
    """Unit vector along the toggle, from C (front seat) to Q (rear pivot).
    A two-force member can only carry load along this axis."""
    dx, dy = Q[0] - C[0], Q[1] - C[1]
    L = math.hypot(dx, dy)
    if L < 1e-12:
        raise ValueError("toggle has zero length")
    return dx / L, dy / L


def quasi_static_reactions(*, pose: PoseState, geom: FourBarGeometry,
                            crushing_force_xy_N: tuple[float, float],
                            crushing_force_point_xy_m: tuple[float, float]
                            ) -> JointReactions:
    """Resolve reactions assuming negligible inertia (peak crushing instant).

    The pitman + swing jaw is one rigid body in static equilibrium under:
      - Crushing force F_c applied at the swing jaw face point
      - Big-end bearing reaction at A
      - Toggle seat reaction at C (constrained along toggle axis)

    Three unknowns (Ax, Ay, |F_toggle|) in three equations (ΣFx, ΣFy, ΣM_A).
    """
    A = pose.A_xy
    C = pose.C_xy
    Q = (geom.base_dx_m, geom.base_dy_m)
    F_cx, F_cy = crushing_force_xy_N
    P = crushing_force_point_xy_m

    # toggle axis unit vector
    ux, uy = two_force_member_unit(C, Q)

    # Moment of crushing force about A (z-component of (P - A) × F_c)
    rx, ry = P[0] - A[0], P[1] - A[1]
    M_F_about_A = rx * F_cy - ry * F_cx

    # Toggle force passes through C, along (ux, uy). Its moment arm about A:
    rcx, rcy = C[0] - A[0], C[1] - A[1]
    M_toggle_per_unit = rcx * uy - rcy * ux   # moment of unit toggle force at C, about A

    if abs(M_toggle_per_unit) < 1e-9:
        raise ValueError(
            "toggle moment arm about big-end is zero — singular configuration "
            f"at crank θ={pose.theta_rad:.3f} rad"
        )
    # Solve ΣM_A = 0:  M_F_about_A + |F_toggle| * M_toggle_per_unit = 0
    F_toggle_mag = -M_F_about_A / M_toggle_per_unit
    F_toggle = (F_toggle_mag * ux, F_toggle_mag * uy)

    # ΣF = 0 gives big-end reaction
    F_bigend = (-(F_cx + F_toggle[0]), -(F_cy + F_toggle[1]))

    # Crank torque = moment of -F_bigend about origin O (eccentric center)
    # The force ON the eccentric pin from the pitman is opposite to F_bigend
    F_on_pin = (-F_bigend[0], -F_bigend[1])
    crank_torque = A[0] * F_on_pin[1] - A[1] * F_on_pin[0]

    return JointReactions(
        t_s=pose.t_s,
        big_end_bearing_N=F_bigend,
        toggle_seat_N=F_toggle,
        toggle_rear_N=(-F_toggle[0], -F_toggle[1]),
        crank_torque_Nm=crank_torque,
    )


def reactions_over_cycle(*, poses: list[PoseState], geom: FourBarGeometry,
                          crushing_force_xy_N: tuple[float, float],
                          crushing_force_point_xy_m: tuple[float, float],
                          crushing_active_arc_rad: tuple[float, float] = (-math.pi / 6,
                                                                          math.pi / 6),
                          ) -> list[JointReactions]:
    """Reactions sampled across the trajectory.

    Crushing force is applied only when the crank angle is within
    `crushing_active_arc_rad` (centered at θ=0, the closed-jaw position).
    Outside that arc, the jaw is opening and crushing force is zero.
    """
    a0, a1 = crushing_active_arc_rad
    out: list[JointReactions] = []
    for pose in poses:
        # Reduce crank angle to (-π, π]
        theta = ((pose.theta_rad + math.pi) % (2 * math.pi)) - math.pi
        active = a0 <= theta <= a1
        F = crushing_force_xy_N if active else (0.0, 0.0)
        out.append(quasi_static_reactions(
            pose=pose, geom=geom,
            crushing_force_xy_N=F,
            crushing_force_point_xy_m=crushing_force_point_xy_m,
        ))
    return out


def summarise_reactions(reactions: list[JointReactions]) -> dict[str, float]:
    """Aggregate statistics useful for bearing sizing + fatigue analysis."""
    if not reactions:
        raise ValueError("empty reactions list")

    def mag(v: tuple[float, float]) -> float:
        return math.hypot(v[0], v[1])

    big_end_mags = [mag(r.big_end_bearing_N) for r in reactions]
    toggle_mags = [mag(r.toggle_seat_N) for r in reactions]
    torques = [r.crank_torque_Nm for r in reactions]

    return {
        "big_end_peak_N": max(big_end_mags),
        "big_end_mean_N": sum(big_end_mags) / len(big_end_mags),
        "toggle_peak_N": max(toggle_mags),
        "toggle_mean_N": sum(toggle_mags) / len(toggle_mags),
        "crank_torque_peak_Nm": max(abs(t) for t in torques),
        "crank_torque_mean_Nm": sum(torques) / len(torques),
        "crank_torque_rms_Nm": math.sqrt(sum(t * t for t in torques) / len(torques)),
    }
