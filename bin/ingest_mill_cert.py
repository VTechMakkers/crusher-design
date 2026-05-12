#!/usr/bin/env python3
"""
Ingest a steel mill certificate into `knowledge/sources/techmakkers_internal.yaml`.

Each cert is an append-only record with heat number, supplier, date, and
measured properties. The trust resolver in `mcp-servers/knowledge/server.py`
automatically prefers these tier-1 values over handbook (tier-4) data.

Interactive:
  python bin/ingest_mill_cert.py

Scripted:
  python bin/ingest_mill_cert.py --material Mn13 --heat MN13-2026-0142 \
      --supplier "Bharat Forge" --date 2026-04-12 \
      --yield 395 --uts 870 --hardness 210 --elongation 38 \
      --composition '{"C_pct":1.18,"Mn_pct":12.4,"Si_pct":0.55}'
"""
from __future__ import annotations
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "knowledge" / "sources" / "techmakkers_internal.yaml"


def _known_materials(root: Path) -> list[str]:
    mats = yaml.safe_load((root / "knowledge/materials.yaml").read_text())
    return sorted((mats or {}).keys())


def ingest_mill_cert(*,
                      material: str,
                      heat_number: str,
                      supplier: str,
                      date_received: str,
                      properties: dict[str, Any],
                      composition: dict[str, float] | None = None,
                      root: Path = ROOT) -> dict[str, Any]:
    """Append a mill-cert record. Returns {appended, total_certs, duplicate}."""
    known = _known_materials(root)
    if material not in known:
        raise ValueError(f"unknown material {material!r}; known: {known}")
    if not heat_number:
        raise ValueError("heat_number is required (uniquely identifies the batch)")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("properties must be a non-empty dict")

    path = root / "knowledge/sources/techmakkers_internal.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    certs = data.get("mill_certs") or []
    if not isinstance(certs, list):
        certs = []

    duplicate = any((c.get("heat_number") == heat_number
                     and c.get("material") == material) for c in certs)

    entry: dict[str, Any] = {
        "material": material,
        "heat_number": heat_number,
        "supplier": supplier,
        "date_received": date_received,
        "properties": dict(properties),
    }
    if composition:
        entry["composition"] = dict(composition)

    certs.append(entry)
    data["mill_certs"] = certs
    if "fab_measurements" not in data:
        data["fab_measurements"] = []
    if "field_observations" not in data:
        data["field_observations"] = []
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return {"appended": True, "total_certs": len(certs),
            "duplicate_heat_number": duplicate}


def _prompt(label: str, default: Any = None, parse=str) -> Any:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return parse(raw)
        except (ValueError, TypeError):
            print(f"  invalid — expected {parse.__name__}")


def _interactive(root: Path) -> int:
    materials = _known_materials(root)
    print("Known materials:", ", ".join(materials))
    material = _prompt("material", parse=str)
    if material not in materials:
        print(f"unknown material {material}", file=sys.stderr)
        return 1
    supplier = _prompt("supplier (e.g. 'Bharat Forge')", parse=str)
    heat = _prompt("heat_number (e.g. 'MN13-2026-0142')", parse=str)
    date = _prompt("date_received", default=datetime.date.today().isoformat())

    mats_yaml = yaml.safe_load((root / "knowledge/materials.yaml").read_text())
    schema = mats_yaml[material]
    print(f"\nEnter measured values (Enter to skip):")
    props: dict[str, Any] = {}
    for key, handbook_val in schema.items():
        if not isinstance(handbook_val, (int, float)):
            continue
        if key in ("density_kg_m3", "youngs_modulus_GPa", "poisson_ratio",
                   "cost_INR_per_kg", "archard_k_relative"):
            continue  # these aren't on a typical mill cert
        raw = input(f"  {key} (handbook: {handbook_val}) [skip]: ").strip()
        if raw:
            try:
                props[key] = float(raw)
            except ValueError:
                print(f"    skipped — not a number")

    composition: dict[str, float] = {}
    print("\nComposition (mass percent). Enter empty key to finish:")
    while True:
        element = input("  element_pct key (e.g. C_pct): ").strip()
        if not element:
            break
        try:
            composition[element] = float(input(f"  {element} value: ").strip())
        except ValueError:
            print("    invalid number, skipped")

    out = ingest_mill_cert(material=material, heat_number=heat, supplier=supplier,
                            date_received=date, properties=props,
                            composition=composition or None, root=root)
    print(f"\nappended cert. total certs: {out['total_certs']}")
    if out["duplicate_heat_number"]:
        print("note: duplicate heat number — older entry kept, this one appended too")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material")
    ap.add_argument("--heat", dest="heat_number")
    ap.add_argument("--supplier", default="")
    ap.add_argument("--date", dest="date_received",
                    default=datetime.date.today().isoformat())
    ap.add_argument("--yield", dest="yield_MPa", type=float)
    ap.add_argument("--uts", dest="uts_MPa", type=float)
    ap.add_argument("--hardness", dest="hardness_HBW", type=float)
    ap.add_argument("--elongation", dest="elongation_pct", type=float)
    ap.add_argument("--composition", dest="composition_json",
                    help="JSON object of element_pct values")
    args = ap.parse_args(argv)

    if not args.material:
        return _interactive(ROOT)

    properties: dict[str, Any] = {}
    if args.yield_MPa is not None:
        properties["yield_strength_MPa"] = args.yield_MPa
    if args.uts_MPa is not None:
        properties["ultimate_strength_MPa"] = args.uts_MPa
    if args.hardness_HBW is not None:
        properties["hardness_HBW"] = args.hardness_HBW
    if args.elongation_pct is not None:
        properties["elongation_pct"] = args.elongation_pct
    composition = json.loads(args.composition_json) if args.composition_json else None

    out = ingest_mill_cert(
        material=args.material, heat_number=args.heat_number,
        supplier=args.supplier, date_received=args.date_received,
        properties=properties, composition=composition, root=ROOT,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
