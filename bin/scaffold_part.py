#!/usr/bin/env python3
"""
Scaffold a new part class.

Usage:
  python bin/scaffold_part.py <part_name> [--criticality sacrificial|structural|safety_critical]
                                          [--material <name>]
                                          [--families single_toggle_jaw fine_jaw]

Creates templates/<part_name>/{geometry.py.stub, metadata.yaml, load_cases.yaml, instances/}
and appends to catalog/parts.yaml.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
CATALOG = ROOT / "catalog" / "parts.yaml"


GEOMETRY_STUB = '''"""{part} parametric geometry — SCAFFOLD.

Replace this stub with real CadQuery code when validated master geometry is available.
"""
from __future__ import annotations
from dataclasses import dataclass

import cadquery as cq


@dataclass
class {ClassName}Params:
    # TODO: define parameters with sensible defaults + bounds
    length_mm: float = 100.0
    width_mm: float = 50.0
    thickness_mm: float = 10.0

    def validate(self) -> None:
        assert self.length_mm > 0
        assert self.width_mm > 0
        assert self.thickness_mm > 0


def build(params: {ClassName}Params) -> cq.Workplane:
    params.validate()
    return cq.Workplane("XY").box(params.length_mm, params.width_mm, params.thickness_mm)


def export_step(params: {ClassName}Params, out_path):
    from pathlib import Path
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(build(params), str(out), exportType="STEP")
    return out
'''


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--criticality", default="structural",
                    choices=["sacrificial", "structural", "safety_critical"])
    ap.add_argument("--material", default="S355J2")
    ap.add_argument("--families", nargs="+",
                    default=["single_toggle_jaw", "fine_jaw"])
    ap.add_argument("--wear-exposed", action="store_true")
    args = ap.parse_args(argv)

    part_dir = TEMPLATES / args.part
    if part_dir.exists():
        print(f"part already exists: {part_dir}", file=sys.stderr)
        return 1

    (part_dir / "instances").mkdir(parents=True)
    class_name = "".join(w.capitalize() for w in args.part.split("_"))
    (part_dir / "geometry.py").write_text(
        GEOMETRY_STUB.format(part=args.part, ClassName=class_name)
    )
    (part_dir / "metadata.yaml").write_text(
        f"part_class: {args.part}\nversion: 0.1.0\nstatus: scaffold\n"
        f"material_default: {args.material}\n"
    )
    (part_dir / "load_cases.yaml").write_text("cases: []\nconstraints: {}\n")

    catalog = yaml.safe_load(CATALOG.read_text()) or {"parts": {}}
    catalog["parts"][args.part] = {
        "criticality": args.criticality,
        "typical_material": args.material,
        "wear_exposed": args.wear_exposed,
        "applies_to_families": args.families,
        "has_geometry": True,
    }
    CATALOG.write_text(yaml.safe_dump(catalog, sort_keys=False))
    print(f"scaffolded part: {part_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
