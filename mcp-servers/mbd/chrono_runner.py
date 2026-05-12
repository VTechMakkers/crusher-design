"""
Project Chrono runner for the single-toggle jaw crusher mechanism.

Adds transient rigid-body dynamics to the MBD layer. The closed-form
solver in `kinematics.py` / `dynamics.py` gives quasi-static joint
reactions — exact for the rigid 4-bar but neglects inertial loads, which
are non-trivial at the 4–6 Hz operating frequency of a crusher.

What Chrono adds that the closed-form does not:
  - Inertial forces from accelerating the pitman + jaw mass
  - Transient response (start-up, shut-down, tramp-iron impact pulse)
  - Time-history of reactions at every joint (input to fatigue analysis)
  - Future extension path: flexible-body pitman (ChElement::Beam), bearing
    contact mechanics (ChContactSurface), DEM-MBD coupling (Chrono::Granular)

Install: PyChrono is NOT on PyPI under the name "pychrono" — that name is
taken by an unrelated scheduling library. Project Chrono publishes its
Python bindings via:
  - conda:  `conda install -c projectchrono pychrono`
  - source: https://projectchrono.org/pychrono/  (CMake + SWIG build)
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from .kinematics import FourBarGeometry


@dataclass(frozen=True)
class MechanismMasses:
    """Rigid-body masses + simplified inertia tensors for the 4-bar.

    All units SI: kg for mass, kg·m² for inertia, m for length.

    The pitman + swing-jaw assembly is treated as one rigid body — this is
    the dominant inertia in single-toggle crushers. Eccentric shaft mass
    is accounted for in `eccentric_inertia_kgm2` (rotational inertia about
    its own axis); its translational mass is irrelevant since it doesn't
    translate. Toggle is light and rigid.

    `pitman_centroid_offset_from_A_m` locates the pitman centroid relative
    to the big-end pin (point A in the kinematics module's frame). For a
    pitman hanging vertically with the eccentric at top, this is roughly
    (0, -L_p/2, 0) where L_p is the pitman length.
    """
    eccentric_shaft_kg: float
    eccentric_inertia_kgm2: float
    pitman_kg: float
    pitman_Ix_kgm2: float
    pitman_Iy_kgm2: float
    pitman_Iz_kgm2: float
    pitman_centroid_offset_from_A_m: tuple[float, float]
    toggle_kg: float = 5.0
    toggle_inertia_kgm2: float = 0.02

    def validate(self) -> None:
        if self.eccentric_shaft_kg <= 0:
            raise ValueError("eccentric_shaft_kg must be positive")
        if self.eccentric_inertia_kgm2 <= 0:
            raise ValueError("eccentric_inertia_kgm2 must be positive")
        if self.pitman_kg <= 0:
            raise ValueError("pitman_kg must be positive")
        for label, I in (("Ix", self.pitman_Ix_kgm2),
                          ("Iy", self.pitman_Iy_kgm2),
                          ("Iz", self.pitman_Iz_kgm2)):
            if I <= 0:
                raise ValueError(f"pitman_{label}_kgm2 must be positive")
        if self.toggle_kg <= 0:
            raise ValueError("toggle_kg must be positive")


@dataclass(frozen=True)
class CrushingPulse:
    """Crushing force applied to the pitman face, active in a crank-angle arc.

    The force is applied at a fixed point on the pitman in its local frame
    (typically the centroid of the jaw face). When the crank angle is
    within `active_arc_rad` of bottom-dead-centre (θ=0), the force ramps
    on with a half-cosine profile and ramps off the same way — this
    avoids a step discontinuity that would alias into the time integrator.
    """
    force_xy_N: tuple[float, float]
    application_point_local_m: tuple[float, float]
    active_arc_rad: tuple[float, float] = (-math.pi / 6, math.pi / 6)

    def magnitude_at_angle(self, crank_angle_rad: float) -> tuple[float, float]:
        """Return (Fx, Fy) in world frame at this crank angle."""
        theta = ((crank_angle_rad + math.pi) % (2 * math.pi)) - math.pi
        a0, a1 = self.active_arc_rad
        if theta < a0 or theta > a1:
            return (0.0, 0.0)
        # Half-cosine ramp: 0 at arc edges, 1 at centre
        centre = 0.5 * (a0 + a1)
        half = 0.5 * (a1 - a0)
        if half <= 0:
            return self.force_xy_N
        ramp = 0.5 * (1.0 + math.cos(math.pi * (theta - centre) / half))
        return (ramp * self.force_xy_N[0], ramp * self.force_xy_N[1])


@dataclass
class SimulationResult:
    """Time-history of joint reactions from one simulation run."""
    times_s: list[float]
    crank_angle_rad: list[float]
    big_end_force_N: list[tuple[float, float]]
    toggle_seat_force_N: list[tuple[float, float]]
    toggle_rear_force_N: list[tuple[float, float]]
    crank_torque_Nm: list[float]
    pitman_position_xy_m: list[tuple[float, float]]
    integrator: str
    timestep_s: float

    def summarise(self) -> dict[str, float]:
        """Peak + RMS of each reaction magnitude over the run."""
        def mag(v: tuple[float, float]) -> float:
            return math.hypot(v[0], v[1])

        def rms(values: list[float]) -> float:
            return math.sqrt(sum(v * v for v in values) / max(len(values), 1))

        big = [mag(v) for v in self.big_end_force_N]
        toggle = [mag(v) for v in self.toggle_seat_force_N]
        rear = [mag(v) for v in self.toggle_rear_force_N]
        torques = self.crank_torque_Nm
        return {
            "big_end_peak_N": max(big) if big else 0.0,
            "big_end_rms_N": rms(big),
            "toggle_seat_peak_N": max(toggle) if toggle else 0.0,
            "toggle_seat_rms_N": rms(toggle),
            "toggle_rear_peak_N": max(rear) if rear else 0.0,
            "toggle_rear_rms_N": rms(rear),
            "crank_torque_peak_Nm": max(abs(t) for t in torques) if torques else 0.0,
            "crank_torque_rms_Nm": rms(torques),
            "n_samples": len(self.times_s),
        }


def _require_chrono():
    try:
        import pychrono as pc
    except ImportError as e:
        raise ImportError(
            "Project Chrono Python bindings not available. NOTE: pip's "
            "'pychrono' on PyPI is an unrelated scheduling library — Project "
            "Chrono ships its bindings via "
            "`conda install -c projectchrono pychrono` or a CMake+SWIG build "
            "from https://projectchrono.org/pychrono/"
        ) from e
    # Project Chrono exposes a `core` submodule; the unrelated PyPI package
    # does not. Differentiate explicitly so callers get a clear error.
    if not hasattr(pc, "ChSystemNSC") and not hasattr(pc, "core"):
        raise ImportError(
            "the imported `pychrono` is not Project Chrono (no ChSystemNSC). "
            "Uninstall the PyPI 'pychrono' package and install the real one "
            "from projectchrono.org"
        )
    return pc.core if hasattr(pc, "core") else pc


def simulate(*,
              geometry: FourBarGeometry,
              masses: MechanismMasses,
              rpm: float,
              crushing: CrushingPulse,
              duration_s: float | None = None,
              timestep_s: float = 1.0e-4,
              integrator: str = "euler_implicit_linearized",
              sample_every: int = 10) -> SimulationResult:
    """Run a transient rigid-body simulation of the 4-bar mechanism.

    Parameters
    ----------
    geometry, masses : geometry and inertia of the mechanism
    rpm             : eccentric shaft speed (motor-prescribed)
    crushing        : the crushing-force pulse (active arc + magnitude)
    duration_s      : total simulation time. Defaults to one revolution.
    timestep_s      : integrator step. Default 1e-4 is conservative for
                      a 4–6 Hz mechanism; reduce for impact studies.
    integrator      : 'euler_implicit_linearized' (default, stable + fast)
                      or 'hht' (better for stiff systems with contact)
    sample_every    : record every N-th step in the output (default 10 →
                      10 kHz sample rate at 1e-4 step, plenty for fatigue
                      post-processing)

    Returns
    -------
    SimulationResult containing time-aligned positions and joint reactions.
    """
    geometry.validate()
    masses.validate()
    if rpm <= 0:
        raise ValueError("rpm must be positive")

    chrono = _require_chrono()
    omega = rpm * 2.0 * math.pi / 60.0
    if duration_s is None:
        duration_s = 60.0 / rpm

    sys = chrono.ChSystemNSC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))

    # ---- bodies ----------------------------------------------------------
    ground = chrono.ChBody()
    ground.SetFixed(True)
    sys.Add(ground)

    eccentric = chrono.ChBody()
    eccentric.SetMass(masses.eccentric_shaft_kg)
    eccentric.SetInertiaXX(chrono.ChVector3d(
        masses.eccentric_inertia_kgm2 * 0.5,
        masses.eccentric_inertia_kgm2 * 0.5,
        masses.eccentric_inertia_kgm2,
    ))
    eccentric.SetPos(chrono.ChVector3d(0.0, 0.0, 0.0))
    sys.Add(eccentric)

    A_x = geometry.eccentric_throw_m
    A_y = 0.0
    pitman = chrono.ChBody()
    pitman.SetMass(masses.pitman_kg)
    pitman.SetInertiaXX(chrono.ChVector3d(
        masses.pitman_Ix_kgm2, masses.pitman_Iy_kgm2, masses.pitman_Iz_kgm2,
    ))
    offset_x, offset_y = masses.pitman_centroid_offset_from_A_m
    pitman.SetPos(chrono.ChVector3d(A_x + offset_x, A_y + offset_y, 0.0))
    sys.Add(pitman)

    # Initialize toggle position from kinematics so all joints close
    from .kinematics import position as _kin_position
    _, C, _, _ = _kin_position(0.0, geometry)
    Q_x, Q_y = geometry.base_dx_m, geometry.base_dy_m

    toggle = chrono.ChBody()
    toggle.SetMass(masses.toggle_kg)
    toggle.SetInertiaXX(chrono.ChVector3d(
        masses.toggle_inertia_kgm2 * 0.1,
        masses.toggle_inertia_kgm2 * 0.1,
        masses.toggle_inertia_kgm2,
    ))
    toggle.SetPos(chrono.ChVector3d(0.5 * (C[0] + Q_x), 0.5 * (C[1] + Q_y), 0.0))
    sys.Add(toggle)

    # ---- joints (all revolute about Z) -----------------------------------
    z_axis_quat = chrono.QuatFromAngleAxis(0.0, chrono.ChVector3d(0, 0, 1))

    def revolute_at(body_a, body_b, x, y):
        joint = chrono.ChLinkRevolute()
        joint.Initialize(body_a, body_b,
                          chrono.ChFramed(chrono.ChVector3d(x, y, 0.0), z_axis_quat))
        return joint

    # Eccentric to ground at origin — driven by motor (constructed below)
    big_end_joint = revolute_at(eccentric, pitman, A_x, A_y)
    sys.Add(big_end_joint)

    toggle_seat_joint = revolute_at(pitman, toggle, C[0], C[1])
    sys.Add(toggle_seat_joint)

    toggle_rear_joint = revolute_at(toggle, ground, Q_x, Q_y)
    sys.Add(toggle_rear_joint)

    # ---- motor: constant angular velocity on eccentric -------------------
    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(eccentric, ground,
                      chrono.ChFramed(chrono.ChVector3d(0, 0, 0), z_axis_quat))
    motor.SetSpeedFunction(chrono.ChFunctionConst(omega))
    sys.Add(motor)

    # ---- integrator ------------------------------------------------------
    if integrator == "hht":
        sys.SetTimestepperType(chrono.ChTimestepper.Type_HHT)
    else:
        sys.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

    # ---- run + sample ----------------------------------------------------
    n_steps = int(round(duration_s / timestep_s))

    times_s: list[float] = []
    angles: list[float] = []
    big_end: list[tuple[float, float]] = []
    toggle_seat: list[tuple[float, float]] = []
    toggle_rear: list[tuple[float, float]] = []
    torques: list[float] = []
    pitman_pos: list[tuple[float, float]] = []

    app_x, app_y = crushing.application_point_local_m
    crushing_force_node = chrono.ChForce()
    crushing_force_node.SetMode(chrono.ChForce.FORCE)
    crushing_force_node.SetVrelpoint(chrono.ChVector3d(app_x, app_y, 0.0))
    pitman.AddForce(crushing_force_node)

    for step in range(n_steps + 1):
        t = step * timestep_s
        theta = omega * t
        fx, fy = crushing.magnitude_at_angle(theta)
        crushing_force_node.SetVector(chrono.ChVector3d(fx, fy, 0.0))

        if step > 0:
            sys.DoStepDynamics(timestep_s)

        if step % sample_every == 0:
            times_s.append(t)
            angles.append(theta)
            r_big = big_end_joint.GetReaction1().force
            r_seat = toggle_seat_joint.GetReaction1().force
            r_rear = toggle_rear_joint.GetReaction1().force
            tau = motor.GetMotorTorque()
            p = pitman.GetPos()
            big_end.append((r_big.x, r_big.y))
            toggle_seat.append((r_seat.x, r_seat.y))
            toggle_rear.append((r_rear.x, r_rear.y))
            torques.append(tau)
            pitman_pos.append((p.x, p.y))

    return SimulationResult(
        times_s=times_s,
        crank_angle_rad=angles,
        big_end_force_N=big_end,
        toggle_seat_force_N=toggle_seat,
        toggle_rear_force_N=toggle_rear,
        crank_torque_Nm=torques,
        pitman_position_xy_m=pitman_pos,
        integrator=integrator,
        timestep_s=timestep_s,
    )


def default_pe400_masses() -> MechanismMasses:
    """Indicative masses for a PE_400x600-class mechanism.

    Representative values, NOT TechMakkers production. Replace with
    measured masses (weighed parts on a calibrated scale) before relying
    on absolute reaction magnitudes; the relative dynamic-vs-static
    comparison is robust to ~30% mass error.
    """
    # Pitman ~ slender bar of mass m, length L: I_zz ≈ m*L²/12 about centroid
    L_p = 0.600
    m_p = 95.0
    Izz = m_p * L_p * L_p / 12.0
    return MechanismMasses(
        eccentric_shaft_kg=85.0,
        eccentric_inertia_kgm2=2.8,        # includes flywheels (idealised)
        pitman_kg=m_p,
        pitman_Ix_kgm2=0.5,
        pitman_Iy_kgm2=Izz,
        pitman_Iz_kgm2=Izz,
        pitman_centroid_offset_from_A_m=(0.0, -L_p / 2.0),
        toggle_kg=8.0,
        toggle_inertia_kgm2=0.05,
    )
