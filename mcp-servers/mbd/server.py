"""
Multibody dynamics MCP server.

Closed-form 4-bar single-toggle solver + bearing-life calculator.
Fast (sub-millisecond per crank revolution) and exact for the rigid-link
assumption, which dominates the dynamics of every jaw crusher in service.

Higher-fidelity flexible-body or contact-rich simulation belongs in a
PyChrono or MBDyn back-end; this module is the production-grade analytical
core that those would be benchmarked against.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-mbd")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "mcp-servers"))

from mbd import kinematics as _kin
from mbd import dynamics as _dyn
from mbd import bearings as _brg


@mcp.tool()
def simulate_cycle(*, eccentric_throw_mm: float, pitman_length_mm: float,
                   toggle_length_mm: float, base_dx_mm: float, base_dy_mm: float,
                   rpm: float, samples_per_rev: int = 360
                   ) -> dict[str, Any]:
    """Sample the single-toggle 4-bar mechanism across one full revolution.
    Returns trajectory + stroke + velocity range for the toggle seat."""
    geom = _kin.FourBarGeometry(
        eccentric_throw_m=eccentric_throw_mm / 1000.0,
        pitman_length_m=pitman_length_mm / 1000.0,
        toggle_length_m=toggle_length_mm / 1000.0,
        base_dx_m=base_dx_mm / 1000.0,
        base_dy_m=base_dy_mm / 1000.0,
    )
    omega = rpm * 2.0 * math.pi / 60.0
    duration = 60.0 / rpm
    poses = _kin.trajectory(geom, omega_rad_s=omega,
                             duration_s=duration, n_steps=samples_per_rev)
    stroke = _kin.stroke(poses)
    vels = _kin.velocities(poses)
    Cx_dots = [v["Cx_dot"] for v in vels]
    Cy_dots = [v["Cy_dot"] for v in vels]
    return {
        "stroke": stroke,
        "samples": len(poses),
        "toggle_seat_speed_peak_m_s": max(math.hypot(vx, vy)
                                           for vx, vy in zip(Cx_dots, Cy_dots)),
        "geometry_mm": {
            "eccentric_throw": eccentric_throw_mm,
            "pitman_length": pitman_length_mm,
            "toggle_length": toggle_length_mm,
            "base_dx": base_dx_mm, "base_dy": base_dy_mm,
        },
        "rpm": rpm,
    }


@mcp.tool()
def reactions(*, eccentric_throw_mm: float, pitman_length_mm: float,
              toggle_length_mm: float, base_dx_mm: float, base_dy_mm: float,
              rpm: float, peak_crushing_force_N: float,
              jaw_face_point_xy_mm: tuple[float, float],
              crushing_arc_deg: float = 60.0,
              samples_per_rev: int = 360
              ) -> dict[str, Any]:
    """Resolve quasi-static joint reactions across one full revolution.

    Crushing force is applied only within ±crushing_arc_deg/2 of bottom-dead-
    centre (θ=0), pointing in -X direction with magnitude peak_crushing_force_N.
    """
    geom = _kin.FourBarGeometry(
        eccentric_throw_m=eccentric_throw_mm / 1000.0,
        pitman_length_m=pitman_length_mm / 1000.0,
        toggle_length_m=toggle_length_mm / 1000.0,
        base_dx_m=base_dx_mm / 1000.0,
        base_dy_m=base_dy_mm / 1000.0,
    )
    omega = rpm * 2.0 * math.pi / 60.0
    poses = _kin.trajectory(geom, omega_rad_s=omega,
                             duration_s=60.0 / rpm, n_steps=samples_per_rev)
    arc = math.radians(crushing_arc_deg) / 2.0
    rxns = _dyn.reactions_over_cycle(
        poses=poses, geom=geom,
        crushing_force_xy_N=(-peak_crushing_force_N, 0.0),
        crushing_force_point_xy_m=(jaw_face_point_xy_mm[0] / 1000.0,
                                    jaw_face_point_xy_mm[1] / 1000.0),
        crushing_active_arc_rad=(-arc, arc),
    )
    return {
        "summary": _dyn.summarise_reactions(rxns),
        "samples": len(rxns),
        "crushing_active_arc_deg": crushing_arc_deg,
    }


@mcp.tool()
def bearing_life(*, mechanism_args: dict[str, Any],
                 bearing_designation: str,
                 bearing_dynamic_load_rating_N: float,
                 bearing_type: str = "roller",
                 duty_factor: float = 0.85
                 ) -> dict[str, Any]:
    """Compute ISO 281 L10 life of the big-end (pitman) bearing.

    `mechanism_args` matches the arguments to `reactions()`. `duty_factor`
    accounts for downtime / idle running — life is scaled by it.
    """
    rxn = reactions(**mechanism_args)
    rpm = mechanism_args["rpm"]

    # Need the per-sample load magnitudes. Re-run with explicit per-sample collection.
    geom = _kin.FourBarGeometry(
        eccentric_throw_m=mechanism_args["eccentric_throw_mm"] / 1000.0,
        pitman_length_m=mechanism_args["pitman_length_mm"] / 1000.0,
        toggle_length_m=mechanism_args["toggle_length_mm"] / 1000.0,
        base_dx_m=mechanism_args["base_dx_mm"] / 1000.0,
        base_dy_m=mechanism_args["base_dy_mm"] / 1000.0,
    )
    omega = rpm * 2.0 * math.pi / 60.0
    poses = _kin.trajectory(geom, omega_rad_s=omega,
                             duration_s=60.0 / rpm,
                             n_steps=mechanism_args.get("samples_per_rev", 360))
    arc = math.radians(mechanism_args.get("crushing_arc_deg", 60.0)) / 2.0
    F_point = mechanism_args["jaw_face_point_xy_mm"]
    rxns = _dyn.reactions_over_cycle(
        poses=poses, geom=geom,
        crushing_force_xy_N=(-mechanism_args["peak_crushing_force_N"], 0.0),
        crushing_force_point_xy_m=(F_point[0] / 1000.0, F_point[1] / 1000.0),
        crushing_active_arc_rad=(-arc, arc),
    )
    big_end_mags = [math.hypot(*r.big_end_bearing_N) for r in rxns]
    dt = (60.0 / rpm) / len(rxns)

    bearing = _brg.BearingSpec(
        designation=bearing_designation,
        dynamic_load_rating_N=bearing_dynamic_load_rating_N,
        exponent=(_brg.BALL_BEARING_EXPONENT if bearing_type == "ball"
                  else _brg.ROLLER_BEARING_EXPONENT),
    )
    life = _brg.life_from_time_history(load_history_N=big_end_mags, dt_s=dt,
                                        bearing=bearing, speed_rpm=rpm)
    life["L10_hours_duty_adjusted"] = life["L10_hours"] / max(duty_factor, 0.01)
    life["bearing"] = bearing.designation
    life["bearing_type"] = bearing_type
    return life


@mcp.tool()
def simulate_dynamic(*, eccentric_throw_mm: float, pitman_length_mm: float,
                      toggle_length_mm: float, base_dx_mm: float,
                      base_dy_mm: float, rpm: float,
                      peak_crushing_force_N: float,
                      jaw_face_point_xy_mm: tuple[float, float],
                      crushing_arc_deg: float = 60.0,
                      duration_revolutions: float = 1.0,
                      timestep_s: float = 1.0e-4,
                      integrator: str = "euler_implicit_linearized",
                      ) -> dict[str, Any]:
    """Project Chrono transient rigid-body simulation of the 4-bar mechanism.

    The dynamic counterpart of `reactions()` which is quasi-static. Captures
    inertial loads that the quasi-static solver omits. Requires Project
    Chrono's Python bindings (NOT pip's `pychrono`, which is unrelated).
    Returns {summary: {...}, dynamic_vs_quasi_static: float ratio} or
    `_not_implemented` if the bindings aren't installed.
    """
    try:
        from mbd.chrono_runner import (CrushingPulse, default_pe400_masses,
                                         simulate as chrono_simulate)
    except ImportError as e:
        return {"_not_implemented": True, "reason": str(e)}

    try:
        geom = _kin.FourBarGeometry(
            eccentric_throw_m=eccentric_throw_mm / 1000.0,
            pitman_length_m=pitman_length_mm / 1000.0,
            toggle_length_m=toggle_length_mm / 1000.0,
            base_dx_m=base_dx_mm / 1000.0,
            base_dy_m=base_dy_mm / 1000.0,
        )
        masses = default_pe400_masses()
        arc_rad = math.radians(crushing_arc_deg) / 2.0
        pulse = CrushingPulse(
            force_xy_N=(-peak_crushing_force_N, 0.0),
            application_point_local_m=(jaw_face_point_xy_mm[0] / 1000.0,
                                         jaw_face_point_xy_mm[1] / 1000.0),
            active_arc_rad=(-arc_rad, arc_rad),
        )
        result = chrono_simulate(
            geometry=geom, masses=masses, rpm=rpm, crushing=pulse,
            duration_s=duration_revolutions * 60.0 / rpm,
            timestep_s=timestep_s, integrator=integrator,
        )
    except ImportError as e:
        return {"_not_implemented": True, "reason": str(e)}

    summary = result.summarise()
    summary["integrator"] = result.integrator
    summary["timestep_s"] = result.timestep_s
    return summary


if __name__ == "__main__":
    mcp.run()
