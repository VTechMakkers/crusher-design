#!/usr/bin/env python3
"""
Scaffold a new (part, model) instance.

Usage:
  python bin/scaffold_instance.py <part> <model> [--material AR400]
                                                 [--from PE_400x600]   (copy params from existing instance)

Creates templates/<part>/instances/<model>.yaml with default or copied params.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("model")
    ap.add_argument("--material")
    ap.add_argument("--from", dest="source", help="copy params from existing instance")
    args = ap.parse_args(argv)

    instances_dir = ROOT / "templates" / args.part / "instances"
    if not instances_dir.exists():
        print(f"unknown part: {args.part} (run scaffold_part first)", file=sys.stderr)
        return 1

    catalog = yaml.safe_load((ROOT / "catalog" / "models.yaml").read_text())
    if args.model not in catalog["models"]:
        print(f"unknown model: {args.model}", file=sys.stderr)
        return 1

    parts = yaml.safe_load((ROOT / "catalog" / "parts.yaml").read_text())["parts"]
    material = args.material or parts[args.part]["typical_material"]

    if args.source:
        src = instances_dir / f"{args.source}.yaml"
        if not src.exists():
            print(f"source instance not found: {src}", file=sys.stderr)
            return 1
        data = yaml.safe_load(src.read_text())
        data["model"] = args.model
        data["material"] = material
        data["validation"] = {"fea_validated": False, "field_validated": False,
                              "units_shipped": 0, "field_failures": 0}
        data["notes"] = f"copied from {args.source}; adjust params for {args.model}"
    else:
        data = {"model": args.model, "part": args.part, "material": material,
                "params": {}, "validation": {"fea_validated": False,
                "field_validated": False, "units_shipped": 0, "field_failures": 0},
                "notes": "fill in validated params"}

    out = instances_dir / f"{args.model}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"scaffolded instance: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
