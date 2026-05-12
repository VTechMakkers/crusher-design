#!/usr/bin/env python3
"""
Ingest production drawing dimensions into a (part, model) instance.

Two modes:
  interactive (default):
    python bin/ingest_drawing.py
      → walks through part / model / each param with prompts and bounds,
        validates against the geometry.py validate() method, runs DFM,
        writes templates/<part>/instances/<model>.yaml with provenance.

  scripted:
    python bin/ingest_drawing.py --part toggle_plate --model PE_400x600 \
        --material AR400 --drawing-ref "TP-PE400-Rev3" \
        --params-json '{"length_mm":540,"thickness_mm":28,...}'

Refuses to write if validate() raises or DFM check fails. Existing
instance file is read first so partial overrides preserve unset fields.
"""
from __future__ import annotations
import argparse
import datetime
import importlib.util
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_geometry_module(part: str, root: Path):
    path = root / "templates" / part / "geometry.py"
    if not path.exists():
        raise FileNotFoundError(f"no geometry.py for part: {part}")
    spec = importlib.util.spec_from_file_location(f"geom_{part}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _params_cls(mod):
    return next(v for k, v in vars(mod).items()
                if k.endswith("Params") and hasattr(v, "__dataclass_fields__"))


def _existing_instance(part: str, model: str, root: Path) -> dict[str, Any] | None:
    path = root / "templates" / part / "instances" / f"{model}.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def _run_dfm(part: str, params: dict[str, Any], root: Path) -> dict[str, Any]:
    """Invoke the dfm server's logic directly (bypassing MCP transport)."""
    spec = importlib.util.spec_from_file_location(
        "dfm_logic", root / "mcp-servers" / "dfm" / "server.py"
    )
    dfm = importlib.util.module_from_spec(spec)
    sys.modules["dfm_logic"] = dfm
    spec.loader.exec_module(dfm)
    if part == "toggle_plate":
        return dfm.check_toggle_plate(params)
    if part in ("swing_jaw_plate", "fixed_jaw_plate"):
        return dfm.check_jaw_plate(params)
    return {"passes": True, "errors": [], "warnings": [], "info": [],
            "total_issues": 0}


def ingest_drawing(*,
                    part: str,
                    model: str,
                    params: dict[str, Any],
                    material: str,
                    drawing_ref: str,
                    revision: str = "",
                    ingested_by: str = "",
                    date: str | None = None,
                    root: Path = ROOT,
                    run_dfm: bool = True) -> dict[str, Any]:
    """Validate + write a part instance from drawing data.

    Returns {written_to, validated, dfm}. Raises ValueError on validation
    or DFM error; never writes a partial / invalid YAML.
    """
    mod = _load_geometry_module(part, root)
    cls = _params_cls(mod)

    valid_keys = {f.name for f in fields(cls)}
    unknown = set(params) - valid_keys
    if unknown:
        raise ValueError(f"unknown params for {part}: {sorted(unknown)}")

    existing = _existing_instance(part, model, root) or {}
    merged = {**(existing.get("params") or {}), **params}

    try:
        instance = cls(**merged)
    except TypeError as e:
        raise ValueError(f"params don't fit {cls.__name__}: {e}") from e
    try:
        instance.validate()
    except AssertionError as e:
        raise ValueError(
            f"params out of bounds for {cls.__name__}: {merged} "
            f"(see validate() in templates/{part}/geometry.py)"
        ) from e

    dfm_result = _run_dfm(part, merged, root) if run_dfm else None
    if run_dfm and not dfm_result["passes"]:
        raise ValueError(f"DFM check failed: {dfm_result['errors']}")

    out_path = root / "templates" / part / "instances" / f"{model}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    when = date or datetime.date.today().isoformat()
    yaml_doc = {
        "model": model,
        "part": part,
        "material": material,
        "params": merged,
        "validation": {
            "fea_validated": False,
            "field_validated": False,
            "units_shipped": (existing.get("validation") or {}).get("units_shipped", 0),
            "field_failures": (existing.get("validation") or {}).get("field_failures", 0),
        },
        "provenance": {
            "drawing_ref": drawing_ref,
            "revision": revision,
            "ingested_by": ingested_by or os.environ.get("USER", ""),
            "ingested_on": when,
            "source": "drawing_ingest",
        },
        "notes": (
            f"Ingested from drawing {drawing_ref}"
            + (f" rev {revision}" if revision else "")
            + f" on {when}."
        ),
    }
    out_path.write_text(yaml.safe_dump(yaml_doc, sort_keys=False))
    return {"written_to": str(out_path), "validated": True, "dfm": dfm_result}


def _prompt(label: str, default: Any = None, parse=str,
             validator=None) -> Any:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            value = parse(raw)
        except (ValueError, TypeError):
            print(f"  invalid input — expected {parse.__name__}")
            continue
        if validator:
            err = validator(value)
            if err:
                print(f"  {err}")
                continue
        return value


def _interactive(root: Path) -> int:
    parts = yaml.safe_load((root / "catalog/parts.yaml").read_text())["parts"]
    part_classes = [p for p, info in parts.items() if info.get("has_geometry")]
    print("Available part classes:", ", ".join(part_classes))
    part = _prompt("part_class", parse=str,
                    validator=lambda v: None if v in part_classes
                    else f"must be one of {part_classes}")

    models = list(yaml.safe_load((root / "catalog/models.yaml").read_text())["models"])
    print(f"Available models ({len(models)}):", ", ".join(models[:6]),
          "..." if len(models) > 6 else "")
    model = _prompt("model", parse=str,
                     validator=lambda v: None if v in models
                     else f"unknown model {v!r}")

    mod = _load_geometry_module(part, root)
    cls = _params_cls(mod)
    existing = _existing_instance(part, model, root) or {}
    existing_params = (existing.get("params") or {})

    print(f"\nEnter dimensions for {part} on {model}. "
          f"Press Enter to keep existing value (in brackets). "
          f"All units mm unless stated.\n")
    params: dict[str, Any] = {}
    for f in fields(cls):
        default = existing_params.get(f.name, getattr(cls(), f.name))
        params[f.name] = _prompt(f.name, default=default,
                                   parse=type(default) if default is not None else float)

    material = _prompt("material",
                        default=existing.get("material",
                                              parts[part].get("typical_material", "")))
    drawing_ref = _prompt("drawing_ref (e.g. TP-PE400-Rev3)", default="")
    revision = _prompt("revision", default="")

    print()
    try:
        out = ingest_drawing(
            part=part, model=model, params=params, material=material,
            drawing_ref=drawing_ref, revision=revision, root=root,
        )
    except ValueError as e:
        print(f"refused to write: {e}", file=sys.stderr)
        return 1

    print(f"wrote {out['written_to']}")
    if out["dfm"]:
        if out["dfm"]["warnings"]:
            print("DFM warnings:")
            for w in out["dfm"]["warnings"]:
                print(f"  - {w}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part")
    ap.add_argument("--model")
    ap.add_argument("--material")
    ap.add_argument("--drawing-ref", default="")
    ap.add_argument("--revision", default="")
    ap.add_argument("--params-json",
                    help="JSON object of param overrides")
    args = ap.parse_args(argv)

    if not args.part:
        return _interactive(ROOT)

    params = json.loads(args.params_json) if args.params_json else {}
    try:
        out = ingest_drawing(
            part=args.part, model=args.model, params=params,
            material=args.material or "",
            drawing_ref=args.drawing_ref, revision=args.revision,
            root=ROOT,
        )
    except ValueError as e:
        print(f"refused to write: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
