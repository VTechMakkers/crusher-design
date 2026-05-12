"""
MBD-to-FEA bridge.

Translates the kinematically-derived joint reactions produced by the MBD
layer into FEA load cases for each part class. The bridge is the only
place in the codebase that knows how forces from the single-toggle 4-bar
mechanism map to each part's load locations:

  big-end bearing reaction  →  pitman big-end load, eccentric shaft rotating load,
                                bearing housing radial load
  toggle seat reaction      →  toggle plate axial compression, pitman toggle seat
  toggle rear reaction      →  main frame rear-seat load
  crushing force (input)    →  swing jaw + fixed jaw face pressure

Peak magnitudes drive static FEA. RMS magnitudes are returned alongside for
fatigue analysis (S-N curves, Goodman/Soderberg). Direction is reported in
human-readable form; downstream FEA decks apply it according to their own
geometry conventions.

This module is the single source of truth for force-flow assumptions. If a
load-flow choice changes (e.g., we start modelling pin friction, or split
pitman into multiple bodies), this is where it changes.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-servers"))

from mbd.kinematics import FourBarGeometry, trajectory
from mbd.dynamics import reactions_over_cycle


def _vec_mag(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _peak(reactions, attr: str) -> float:
    return max(_vec_mag(getattr(r, attr)) for r in reactions)


def _rms(reactions, attr: str) -> float:
    mags = [_vec_mag(getattr(r, attr)) for r in reactions]
    return math.sqrt(sum(m * m for m in mags) / len(mags))


def derive_load_case(*,
                     part_class: str,
                     mechanism_geometry: FourBarGeometry,
                     rpm: float,
                     peak_crushing_force_N: float,
                     jaw_face_point_xy_mm: tuple[float, float],
                     crushing_arc_deg: float = 60.0,
                     samples_per_rev: int = 360,
                     ) -> dict[str, Any]:
    """Compute MBD-derived FEA load case for one part class.

    Returns a dict in the same schema as templates/<part>/load_cases.yaml
    `cases:` entries, so it can be passed to design_loop.evaluate as an
    override_load_case.
    """
    mechanism_geometry.validate()
    omega = rpm * 2.0 * math.pi / 60.0
    poses = trajectory(mechanism_geometry, omega_rad_s=omega,
                        duration_s=60.0 / rpm, n_steps=samples_per_rev)
    arc = math.radians(crushing_arc_deg) / 2.0
    rxns = reactions_over_cycle(
        poses=poses, geom=mechanism_geometry,
        crushing_force_xy_N=(-peak_crushing_force_N, 0.0),
        crushing_force_point_xy_m=(jaw_face_point_xy_mm[0] / 1000.0,
                                    jaw_face_point_xy_mm[1] / 1000.0),
        crushing_active_arc_rad=(-arc, arc),
    )

    common = {
        "source": "mbd_derived",
        "rpm": rpm,
        "samples_per_rev": samples_per_rev,
        "crushing_arc_deg": crushing_arc_deg,
        "input_crushing_force_kN": peak_crushing_force_N / 1000.0,
    }

    if part_class == "toggle_plate":
        return {
            **common,
            "name": "mbd_peak_compression",
            "description": "toggle peak axial reaction over one revolution",
            "seat_force_kN": _peak(rxns, "toggle_seat_N") / 1000.0,
            "seat_force_rms_kN": _rms(rxns, "toggle_seat_N") / 1000.0,
            "direction": "axial along toggle",
        }

    if part_class in ("swing_jaw_plate", "fixed_jaw_plate"):
        # Jaw face load is the MBD input, not output. The reaction-derived
        # quantity here is what propagates further into the mechanism; the
        # jaw face FEA itself uses the crushing force directly.
        return {
            **common,
            "name": "mbd_peak_crushing",
            "description": "crushing force applied normal to jaw face",
            "crushing_force_kN": peak_crushing_force_N / 1000.0,
            "direction": "normal_to_face",
        }

    if part_class == "pitman":
        return {
            **common,
            "name": "mbd_peak_combined",
            "description": "pitman sees both big-end and toggle seat reactions",
            "big_end_force_kN": _peak(rxns, "big_end_bearing_N") / 1000.0,
            "big_end_force_rms_kN": _rms(rxns, "big_end_bearing_N") / 1000.0,
            "toggle_seat_force_kN": _peak(rxns, "toggle_seat_N") / 1000.0,
            "toggle_seat_force_rms_kN": _rms(rxns, "toggle_seat_N") / 1000.0,
        }

    if part_class == "main_frame":
        return {
            **common,
            "name": "mbd_peak_reactions",
            "description": "frame load path: rear toggle seat + bearing housings",
            "toggle_rear_force_kN": _peak(rxns, "toggle_rear_N") / 1000.0,
            "toggle_rear_force_rms_kN": _rms(rxns, "toggle_rear_N") / 1000.0,
            "bearing_force_per_housing_kN": _peak(rxns, "big_end_bearing_N") / 2.0 / 1000.0,
        }

    if part_class == "eccentric_shaft":
        return {
            **common,
            "name": "mbd_rotating_bending",
            "description": "shaft sees big-end reaction rotating at shaft rpm",
            "load_kN": _peak(rxns, "big_end_bearing_N") / 1000.0,
            "load_rms_kN": _rms(rxns, "big_end_bearing_N") / 1000.0,
            "rotation_rpm": rpm,
            "note": "use rotating-bending fatigue (fully reversed) at this load",
        }

    if part_class == "bearing_housing":
        return {
            **common,
            "name": "mbd_bearing_radial",
            "description": "radial load on bearing housing from shaft reaction",
            "radial_force_kN": _peak(rxns, "big_end_bearing_N") / 2.0 / 1000.0,
            "radial_force_rms_kN": _rms(rxns, "big_end_bearing_N") / 2.0 / 1000.0,
        }

    raise ValueError(f"no MBD->FEA mapping for part_class={part_class!r}")


def load_mechanism_from_assembly(model: str, root: Path | None = None
                                  ) -> tuple[FourBarGeometry, float]:
    """Read the `mechanism:` block from assembly/<model>.yaml.

    Falls back to representative PE-class values when the block is absent
    so older assembly definitions remain usable. Returns (geometry, rpm).
    """
    root = root or ROOT
    asm_path = root / "assembly" / f"{model}.yaml"
    asm = yaml.safe_load(asm_path.read_text())
    mech = asm.get("mechanism")
    if mech is None:
        return (FourBarGeometry(eccentric_throw_m=0.014,
                                 pitman_length_m=0.600,
                                 toggle_length_m=0.300,
                                 base_dx_m=-0.350,
                                 base_dy_m=-0.282),
                280.0)
    geom = FourBarGeometry(
        eccentric_throw_m=mech["eccentric_throw_mm"] / 1000.0,
        pitman_length_m=mech["pitman_length_mm"] / 1000.0,
        toggle_length_m=mech["toggle_length_mm"] / 1000.0,
        base_dx_m=mech["base_dx_mm"] / 1000.0,
        base_dy_m=mech["base_dy_mm"] / 1000.0,
    )
    return geom, float(mech.get("rpm", 280.0))


def derive_load_case_for_model(*,
                                part_class: str,
                                model: str,
                                peak_crushing_force_N: float,
                                jaw_face_point_xy_mm: tuple[float, float] = (0.0, -300.0),
                                crushing_arc_deg: float = 60.0,
                                samples_per_rev: int = 360,
                                root: Path | None = None,
                                ) -> dict[str, Any]:
    """Convenience: load mechanism from assembly YAML, derive load case."""
    geom, rpm = load_mechanism_from_assembly(model, root=root)
    return derive_load_case(
        part_class=part_class,
        mechanism_geometry=geom,
        rpm=rpm,
        peak_crushing_force_N=peak_crushing_force_N,
        jaw_face_point_xy_mm=jaw_face_point_xy_mm,
        crushing_arc_deg=crushing_arc_deg,
        samples_per_rev=samples_per_rev,
    )
