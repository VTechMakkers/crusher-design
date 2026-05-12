"""
Single-toggle jaw crusher mechanism kinematics.

The mechanism is a planar 4-bar linkage:

      O (eccentric center, fixed)
       \\
        \\  crank, length e
         A (eccentric pin / pitman big-end)
         |
         |  pitman, length L_p
         |
         C (toggle seat) ---- toggle, length L_t ---- Q (rear pivot, fixed)

Inputs:  crank angle θ from a constant-speed motor.
Outputs: positions, velocities, accelerations of all moving joints.

Branch selection: the C below A (pitman hanging from eccentric), which is
the physical configuration in every production jaw crusher. The other
intersection branch is non-physical for this mechanism.

All units SI: meters, radians, seconds.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FourBarGeometry:
    eccentric_throw_m: float
    pitman_length_m: float
    toggle_length_m: float
    base_dx_m: float
    base_dy_m: float

    def validate(self) -> None:
        e, L_p, L_t = self.eccentric_throw_m, self.pitman_length_m, self.toggle_length_m
        base = math.hypot(self.base_dx_m, self.base_dy_m)
        if not (e > 0 and L_p > 0 and L_t > 0):
            raise ValueError("all lengths must be positive")
        # Grashof: longest + shortest <= sum of other two (assemblable for any crank angle)
        lengths = sorted([e, L_p, L_t, base])
        if lengths[0] + lengths[3] > lengths[1] + lengths[2] + 1e-9:
            raise ValueError(
                f"non-Grashof — mechanism cannot rotate fully: lengths {lengths}"
            )


@dataclass(frozen=True)
class PoseState:
    t_s: float
    theta_rad: float
    omega_rad_s: float
    A_xy: tuple[float, float]
    C_xy: tuple[float, float]
    pitman_angle_rad: float
    toggle_angle_rad: float


def _circle_circle_lower(p1: tuple[float, float], r1: float,
                          p2: tuple[float, float], r2: float
                          ) -> tuple[float, float]:
    """Return the intersection point of two circles with the LOWER y value
    (physically correct branch for a downward-hanging pitman)."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    d_sq = dx * dx + dy * dy
    d = math.sqrt(d_sq)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        raise ValueError(
            f"4-bar at this crank angle is not assemblable "
            f"(d={d:.4f}, r1+r2={r1+r2:.4f}, |r1-r2|={abs(r1-r2):.4f})"
        )
    a = (r1 * r1 - r2 * r2 + d_sq) / (2.0 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    mx = x1 + a * dx / d
    my = y1 + a * dy / d
    s1 = (mx + h * dy / d, my - h * dx / d)
    s2 = (mx - h * dy / d, my + h * dx / d)
    return s1 if s1[1] <= s2[1] else s2


def position(theta_rad: float, geom: FourBarGeometry) -> tuple[tuple[float, float],
                                                                tuple[float, float],
                                                                float, float]:
    """Return (A, C, pitman_angle, toggle_angle) for a given crank angle.
    A is the big-end pin position; C is the toggle seat position.
    Angles are measured from +x axis, in (-π, π]."""
    A = (geom.eccentric_throw_m * math.cos(theta_rad),
         geom.eccentric_throw_m * math.sin(theta_rad))
    Q = (geom.base_dx_m, geom.base_dy_m)
    C = _circle_circle_lower(A, geom.pitman_length_m, Q, geom.toggle_length_m)
    pitman_angle = math.atan2(C[1] - A[1], C[0] - A[0])
    toggle_angle = math.atan2(C[1] - Q[1], C[0] - Q[0])
    return A, C, pitman_angle, toggle_angle


def trajectory(geom: FourBarGeometry, omega_rad_s: float,
               duration_s: float, n_steps: int = 360,
               theta_0_rad: float = 0.0) -> list[PoseState]:
    """Sample positions across one or more revolutions at uniform time steps."""
    geom.validate()
    dt = duration_s / n_steps
    out: list[PoseState] = []
    for i in range(n_steps + 1):
        t = i * dt
        theta = theta_0_rad + omega_rad_s * t
        A, C, p_ang, t_ang = position(theta, geom)
        out.append(PoseState(t_s=t, theta_rad=theta, omega_rad_s=omega_rad_s,
                              A_xy=A, C_xy=C,
                              pitman_angle_rad=p_ang, toggle_angle_rad=t_ang))
    return out


def velocities(poses: list[PoseState]) -> list[dict[str, float]]:
    """Central-difference velocities of A, C, pitman angle, toggle angle.
    First and last samples use forward/backward differences."""
    n = len(poses)
    if n < 3:
        raise ValueError("need at least 3 poses for velocity estimate")
    out: list[dict[str, float]] = []
    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = poses[j1].t_s - poses[j0].t_s
        if dt == 0:
            raise ValueError("zero dt in velocity differentiation")
        Ax_dot = (poses[j1].A_xy[0] - poses[j0].A_xy[0]) / dt
        Ay_dot = (poses[j1].A_xy[1] - poses[j0].A_xy[1]) / dt
        Cx_dot = (poses[j1].C_xy[0] - poses[j0].C_xy[0]) / dt
        Cy_dot = (poses[j1].C_xy[1] - poses[j0].C_xy[1]) / dt
        phi_p_dot = (poses[j1].pitman_angle_rad - poses[j0].pitman_angle_rad) / dt
        phi_t_dot = (poses[j1].toggle_angle_rad - poses[j0].toggle_angle_rad) / dt
        out.append({
            "Ax_dot": Ax_dot, "Ay_dot": Ay_dot,
            "Cx_dot": Cx_dot, "Cy_dot": Cy_dot,
            "pitman_angle_dot": phi_p_dot, "toggle_angle_dot": phi_t_dot,
        })
    return out


def stroke(poses: list[PoseState]) -> dict[str, float]:
    """Peak-to-peak motion of the toggle seat C over the supplied trajectory.
    Stroke maps directly to crusher CSS change (closed-side-setting modulation)."""
    if not poses:
        raise ValueError("empty trajectory")
    xs = [p.C_xy[0] for p in poses]
    ys = [p.C_xy[1] for p in poses]
    return {
        "stroke_x_mm": (max(xs) - min(xs)) * 1000.0,
        "stroke_y_mm": (max(ys) - min(ys)) * 1000.0,
        "css_change_proxy_mm": (max(xs) - min(xs)) * 1000.0,
    }
