"""
Fixed jaw plate parametric geometry.

The stationary crushing surface. Pairs with the swing jaw — both are
corrugated, both Mn13/Mn18, similar wear profile. Mounted to the main
frame via tapered wedges + bolts. Tooth profile is OFFSET from swing
jaw by half a pitch (so peak meets valley) for optimal nip + breakage.

Parameters bounded; defaults are placeholders — replace with TechMakkers'
validated master geometry per crusher model.

DEM consumes the STL export to define the static wall in the simulation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FixedJawParams:
    # overall plate
    height_mm: float = 700.0
    width_mm: float = 400.0
    thickness_mm: float = 70.0          # slightly thicker than swing (no impact accel)
    backing_thickness_mm: float = 40.0

    # corrugation (tooth) profile — matches swing jaw pitch, OFFSET by half-pitch
    tooth_pitch_mm: float = 90.0
    tooth_depth_mm: float = 22.0
    tooth_angle_deg: float = 95.0
    tooth_phase_offset_frac: float = 0.5   # 0.5 = peak-meets-valley vs swing jaw

    # mounting interface (matches frame)
    mount_holes: int = 4
    mount_hole_dia_mm: float = 24.0
    mount_hole_inset_mm: float = 40.0

    def validate(self) -> None:
        assert 300 <= self.height_mm <= 1600
        assert 200 <= self.width_mm <= 1500
        assert 30 <= self.thickness_mm <= 150
        assert 15 <= self.backing_thickness_mm <= 80
        assert 30 <= self.tooth_pitch_mm <= 200
        assert 5 <= self.tooth_depth_mm <= 60
        assert 60 <= self.tooth_angle_deg <= 130
        assert 0.0 <= self.tooth_phase_offset_frac <= 1.0
        assert self.mount_holes in (2, 3, 4, 6)


def _corrugated_profile(p: FixedJawParams):
    """2D corrugation profile in the YZ plane, extruded along X (width)."""
    import cadquery as cq
    half_pitch = p.tooth_pitch_mm / 2.0
    n_teeth = max(1, int(p.height_mm / p.tooth_pitch_mm))
    phase_z = p.tooth_pitch_mm * p.tooth_phase_offset_frac
    points = [(0.0, -phase_z)]
    z = -phase_z
    for _ in range(n_teeth + 1):
        z += half_pitch
        points.append((p.tooth_depth_mm, z))
        z += half_pitch
        points.append((0.0, z))
    points.append((0.0, p.height_mm))
    points.append((-p.thickness_mm, p.height_mm))
    points.append((-p.thickness_mm, -phase_z))
    points.append((0.0, -phase_z))
    return cq.Workplane("YZ").polyline(points).close().extrude(p.width_mm)


def build(params: FixedJawParams):
    params.validate()
    p = params
    body = _corrugated_profile(p)

    if p.mount_holes:
        if p.mount_holes > 1:
            offsets_z = [
                p.mount_hole_inset_mm + i * (p.height_mm - 2 * p.mount_hole_inset_mm)
                / (p.mount_holes - 1) for i in range(p.mount_holes)
            ]
        else:
            offsets_z = [p.height_mm / 2]
        for z in offsets_z:
            body = (body
                    .faces("<X")
                    .workplane(offset=0.1)
                    .pushPoints([(p.width_mm / 2, z)])
                    .hole(p.mount_hole_dia_mm))

    return body


def export_step(params: FixedJawParams, out_path) -> Path:
    import cadquery as cq
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STEP")
    return out


def export_stl_for_dem(params: FixedJawParams, out_path) -> Path:
    import cadquery as cq
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STL",
                        tolerance=0.5, angularTolerance=0.2)
    return out


if __name__ == "__main__":
    import json
    from dataclasses import asdict
    p = FixedJawParams()
    step = export_step(p, "out/fixed_jaw_default.step")
    print(json.dumps({"step": str(step), "params": asdict(p)}, indent=2))
