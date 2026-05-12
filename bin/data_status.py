#!/usr/bin/env python3
"""
Single-screen report of placeholder vs real data across the codebase.

  Parts × Models coverage
  Material data tier (handbook vs mill cert)
  Field telemetry presence
  Overall % real

Run:
  python bin/data_status.py          # human-readable
  python bin/data_status.py --json   # machine-readable
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _is_real_instance(instance: dict[str, Any]) -> bool:
    """Heuristic: an instance is 'real' if its notes/provenance show drawing
    ingest or it has been field-validated. Placeholder-labeled notes mean fake."""
    notes = (instance.get("notes") or "").upper()
    if "PLACEHOLDER" in notes:
        return False
    if instance.get("provenance", {}).get("source") == "drawing_ingest":
        return True
    val = instance.get("validation") or {}
    if val.get("field_validated") or val.get("units_shipped", 0) > 0:
        return True
    # Has notes but no PLACEHOLDER token and no provenance — leave as real-ish
    return bool(notes)


def assess_data_status(root: Path = ROOT) -> dict[str, Any]:
    catalog_parts = yaml.safe_load((root / "catalog/parts.yaml").read_text())["parts"]
    catalog_models = yaml.safe_load((root / "catalog/models.yaml").read_text())["models"]

    parts_with_geom = [p for p, info in catalog_parts.items()
                        if info.get("has_geometry")]
    parts_without_geom = [p for p in catalog_parts if p not in parts_with_geom]

    instances_real: list[tuple[str, str]] = []
    instances_placeholder: list[tuple[str, str]] = []
    for part in parts_with_geom:
        inst_dir = root / "templates" / part / "instances"
        if not inst_dir.exists():
            continue
        for inst_file in inst_dir.glob("*.yaml"):
            data = yaml.safe_load(inst_file.read_text())
            tag = (part, inst_file.stem)
            (instances_real if _is_real_instance(data) else instances_placeholder).append(tag)

    n_possible_geom = len(parts_with_geom) * len(catalog_models)
    instances_present = len(instances_real) + len(instances_placeholder)
    instances_missing = n_possible_geom - instances_present

    materials_yaml = yaml.safe_load((root / "knowledge/materials.yaml").read_text()) or {}
    internal = yaml.safe_load(
        (root / "knowledge/sources/techmakkers_internal.yaml").read_text()) or {}
    mill_certs = internal.get("mill_certs") or []
    fab_measurements = internal.get("fab_measurements") or []
    field_observations = internal.get("field_observations") or []
    materials_with_certs = sorted({c.get("material") for c in mill_certs
                                     if isinstance(c, dict) and c.get("material")})

    real_score = (len(instances_real) / max(n_possible_geom, 1) * 0.5
                   + (len(materials_with_certs) / max(len(materials_yaml), 1)) * 0.3
                   + min(len(field_observations) / 10.0, 1.0) * 0.2)

    return {
        "parts": {
            "total": len(catalog_parts),
            "with_geometry": len(parts_with_geom),
            "stub_only": len(parts_without_geom),
            "stub_classes": parts_without_geom,
        },
        "models": {
            "total": len(catalog_models),
        },
        "instances": {
            "possible_geometry_combinations": n_possible_geom,
            "present": instances_present,
            "real": len(instances_real),
            "placeholder": len(instances_placeholder),
            "missing": instances_missing,
            "real_pairs": instances_real,
            "placeholder_pairs": instances_placeholder,
        },
        "materials": {
            "in_handbook": len(materials_yaml),
            "with_mill_certs": len(materials_with_certs),
            "certified_list": materials_with_certs,
        },
        "internal_data": {
            "mill_certs": len(mill_certs),
            "fab_measurements": len(fab_measurements),
            "field_observations": len(field_observations),
        },
        "realness_score_0_to_1": round(real_score, 3),
    }


def _format_human(status: dict[str, Any]) -> str:
    p = status["parts"]
    i = status["instances"]
    m = status["materials"]
    d = status["internal_data"]
    out = []
    out.append("crusher-design data status")
    out.append("=" * 50)
    out.append("")
    out.append("PARTS")
    out.append(f"  catalog:           {p['total']} part classes")
    out.append(f"  with geometry.py:  {p['with_geometry']} ({100*p['with_geometry']/max(p['total'],1):.0f}%)")
    out.append(f"  stubs only:        {p['stub_only']}  {p['stub_classes']}")
    out.append("")
    out.append("INSTANCES (part, model)")
    out.append(f"  possible:          {i['possible_geometry_combinations']}")
    out.append(f"  present:           {i['present']}")
    out.append(f"  real (ingested):   {i['real']}")
    out.append(f"  placeholder:       {i['placeholder']}")
    out.append(f"  missing:           {i['missing']}")
    out.append("")
    out.append("MATERIAL DATA")
    out.append(f"  handbook entries:  {m['in_handbook']}")
    out.append(f"  with mill certs:   {m['with_mill_certs']}  {m['certified_list']}")
    out.append("")
    out.append("INTERNAL CORPUS")
    out.append(f"  mill_certs:        {d['mill_certs']}")
    out.append(f"  fab_measurements:  {d['fab_measurements']}")
    out.append(f"  field_observations:{d['field_observations']}")
    out.append("")
    out.append(f"OVERALL REALNESS:    {status['realness_score_0_to_1']:.1%}")
    out.append("")
    if i["placeholder_pairs"]:
        out.append("Placeholder instances awaiting drawing ingest:")
        for part, model in i["placeholder_pairs"]:
            out.append(f"  - {part}  on  {model}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of human format")
    args = ap.parse_args(argv)
    status = assess_data_status(ROOT)
    if args.json:
        print(json.dumps(status, indent=2, default=str))
    else:
        print(_format_human(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
