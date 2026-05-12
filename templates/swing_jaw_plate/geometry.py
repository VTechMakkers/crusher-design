"""
Swing jaw plate parametric geometry.

The swing jaw is one of the two crushing surfaces (the moving one). Highest
wear part class in the entire crusher. Made of Mn13/Mn18 to work-harden
under impact + abrasion. Cast with corrugated face to grip + bite feed.

Parameters bounded; defaults are placeholders — replace with TechMakkers'
validated master geometry per crusher model.

The corrugation pattern (tooth profile) is the high-leverage design variable —
optimized via DEM simulation for breakage efficiency + wear distribution.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SwingJawParams:
    # overall plate
    height_mm: float = 700.0        # along feed direction
    width_mm: float = 400.0         # across jaw mouth
    thickness_mm: float = 65.0      # mean plate thickness
    backing_thickness_mm: float = 35.0  # mounting backing plate

    # corrugation (tooth) profile
    tooth_pitch_mm: float = 90.0    # peak-to-peak along height
    tooth_depth_mm: float = 22.0    # peak amplitude
    tooth_angle_deg: float = 95.0   # included angle of tooth profile

    # mounting interface
    mount_holes: int = 4
    mount_hole_dia_mm: float = 24.0
    mount_hole_inset_mm: float = 40.0

    # wedge edges (tapered top + bottom for retention)
    top_taper_deg: float = 6.0
    bottom_taper_deg: float = 6.0

    def validate(self) -> None:
        assert 300 <= self.height_mm <= 1600
        assert 200 <= self.width_mm <= 1500
        assert 30 <= self.thickness_mm <= 150
        assert 15 <= self.backing_thickness_mm <= 60
        assert 30 <= self.tooth_pitch_mm <= 200
        assert 5 <= self.tooth_depth_mm <= 60
        assert 60 <= self.tooth_angle_deg <= 130
        assert self.mount_holes in (2, 3, 4, 6)
        assert 12 <= self.mount_hole_dia_mm <= 48


def _corrugated_profile(p: SwingJawParams):
    """2D corrugation profile in the YZ plane (height × thickness). Triangular
    tooth pattern across the plate face, swept by width."""
    import cadquery as cq
    half_pitch = p.tooth_pitch_mm / 2.0
    n_teeth = max(1, int(p.height_mm / p.tooth_pitch_mm))
    points = [(0.0, 0.0)]
    z = 0.0
    for i in range(n_teeth):
        z += half_pitch
        points.append((p.tooth_depth_mm, z))
        z += half_pitch
        points.append((0.0, z))
    points.append((0.0, p.height_mm))
    points.append((-p.thickness_mm, p.height_mm))
    points.append((-p.thickness_mm, 0.0))
    points.append((0.0, 0.0))
    sketch = cq.Workplane("YZ").polyline(points).close()
    return sketch.extrude(p.width_mm)


def build(params: SwingJawParams):
    params.validate()
    p = params

    body = _corrugated_profile(p)

    # mounting holes — through the backing thickness, along centerline
    if p.mount_holes:
        offsets_z = [
            p.mount_hole_inset_mm + i * (p.height_mm - 2 * p.mount_hole_inset_mm)
            / (p.mount_holes - 1) for i in range(p.mount_holes)
        ] if p.mount_holes > 1 else [p.height_mm / 2]
        for z in offsets_z:
            body = (body
                    .faces("<X")
                    .workplane(offset=0.1)
                    .pushPoints([(p.width_mm / 2, z)])
                    .hole(p.mount_hole_dia_mm))

    # tapered top + bottom edges for wedge retention (slight chamfer)
    if p.top_taper_deg > 0 or p.bottom_taper_deg > 0:
        body = body.edges("|X and >Z").chamfer(p.tooth_depth_mm * 0.25)
        body = body.edges("|X and <Z").chamfer(p.tooth_depth_mm * 0.25)

    return body


def export_step(params: SwingJawParams, out_path) -> Path:
    import cadquery as cq
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STEP")
    return out


def export_stl_for_dem(params: SwingJawParams, out_path) -> Path:
    """STL export for DEM solver consumption."""
    import cadquery as cq
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STL",
                        tolerance=0.5, angularTolerance=0.2)
    return out


if __name__ == "__main__":
    import json
    from dataclasses import asdict
    p = SwingJawParams()
    step = export_step(p, "out/swing_jaw_default.step")
    print(json.dumps({"step": str(step), "params": asdict(p)}, indent=2))
