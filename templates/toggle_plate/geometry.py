"""
Toggle plate parametric geometry.

The toggle plate is a sacrificial compression member in a jaw crusher.
It transmits load from the pitman to the back wall and is designed to
shear/buckle under tramp-iron overload to protect the rest of the machine.

Parameters are bounded; defaults are placeholders — replace with
TechMakkers' validated master geometry before production use.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ToggleParams:
    length_mm: float = 520.0       # overall length between seat centers
    width_mm: float = 180.0        # plate width
    thickness_mm: float = 28.0     # primary plate thickness
    seat_radius_mm: float = 35.0   # toggle seat radius (each end)
    web_height_mm: float = 60.0    # stiffening web depth
    web_thickness_mm: float = 12.0 # web thickness
    relief_groove_depth_mm: float = 6.0   # shear-relief groove depth
    relief_groove_width_mm: float = 8.0   # shear-relief groove width

    def validate(self) -> None:
        assert 300 <= self.length_mm <= 900,         "length out of range"
        assert 100 <= self.width_mm <= 300,          "width out of range"
        assert 15 <= self.thickness_mm <= 60,        "thickness out of range"
        assert 20 <= self.seat_radius_mm <= 70,      "seat radius out of range"
        assert 30 <= self.web_height_mm <= 120,      "web height out of range"
        assert 6 <= self.web_thickness_mm <= 25,     "web thickness out of range"
        assert 0 <= self.relief_groove_depth_mm <= self.thickness_mm * 0.4
        assert 0 <= self.relief_groove_width_mm <= self.width_mm * 0.3


def build(params: ToggleParams):
    """Return CadQuery workplane for the toggle plate."""
    import cadquery as cq
    params.validate()
    p = params

    # primary plate
    plate = (
        cq.Workplane("XY")
        .box(p.length_mm, p.width_mm, p.thickness_mm)
        .edges("|Z")
        .fillet(min(p.width_mm * 0.1, 15.0))
    )

    # toggle seats — cylindrical reliefs at each end (curved seating surfaces)
    plate = (
        plate.faces(">Z").workplane()
        .pushPoints([(-p.length_mm / 2 + p.seat_radius_mm, 0),
                     ( p.length_mm / 2 - p.seat_radius_mm, 0)])
        .cylinder(p.thickness_mm, p.seat_radius_mm, combine="cut")
    )

    # central shear-relief groove (transverse) — the controlled-failure feature
    if p.relief_groove_depth_mm > 0:
        plate = (
            plate.faces(">Z").workplane()
            .rect(p.relief_groove_width_mm, p.width_mm)
            .cutBlind(-p.relief_groove_depth_mm)
        )

    # stiffening web on underside (longitudinal)
    web = (
        cq.Workplane("XY")
        .workplane(offset=-p.thickness_mm / 2 - p.web_height_mm / 2)
        .box(p.length_mm - 2 * p.seat_radius_mm,
             p.web_thickness_mm,
             p.web_height_mm)
    )

    return plate.union(web)


def export_step(params: ToggleParams, out_path: str | Path) -> Path:
    """Build geometry and export STEP file. Returns absolute path."""
    import cadquery as cq
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STEP")
    return out


if __name__ == "__main__":
    import json
    p = ToggleParams()
    path = export_step(p, "out/toggle_plate_default.step")
    print(json.dumps({"step": str(path), "params": asdict(p)}, indent=2))
