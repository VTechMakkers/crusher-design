"""
Design loop driver.

Orchestrates: variant generation -> meshing -> FEA -> metrics -> fitness -> rank.
Works over (part, model) pairs. Logs every run to:
   templates/<part>/instances/<model>.history.jsonl   (append-only per model)

Two run modes:
  1. SAMPLE — sweep parameter grid over a (part, model) baseline (deterministic)
  2. EVOLVE — LLM-proposed mutations from outside (Opus today, Mythos tomorrow)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .fitness import score

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
KNOWLEDGE = ROOT / "knowledge"
CATALOG = ROOT / "catalog"


def load_part_module(part: str):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        f"geom_{part}", TEMPLATES / part / "geometry.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_instance(part: str, model: str) -> dict[str, Any]:
    path = TEMPLATES / part / "instances" / f"{model}.yaml"
    return yaml.safe_load(path.read_text())


def load_load_cases(part: str) -> dict[str, Any]:
    return yaml.safe_load((TEMPLATES / part / "load_cases.yaml").read_text())


def load_materials() -> dict[str, Any]:
    return yaml.safe_load((KNOWLEDGE / "materials.yaml").read_text())


def evaluate(
    *,
    part: str,
    model: str,
    params: dict[str, Any],
    material: str | None = None,
    load_case_name: str | None = None,
    override_load_case: dict[str, Any] | None = None,
    runner: dict[str, Callable] | None = None,
) -> dict[str, Any]:
    """End-to-end evaluation of one variant for one (part, model).
    `runner` keys: 'generate', 'mesh', 'solve', 'extract_metrics'.
    If None, plumbing-only dry-run is used (no FEA executed).
    `override_load_case`: pass an MBD-derived case (or any user case) to
    bypass loading from load_cases.yaml — used by the MBD-driven loop."""
    inst = load_instance(part, model)
    material = material or inst["material"]
    if override_load_case is not None:
        case = override_load_case
        load_case_name = case.get("name", "override")
    else:
        cases = load_load_cases(part)
        if load_case_name is None:
            load_case_name = cases["cases"][0]["name"]
        case = next(c for c in cases["cases"] if c["name"] == load_case_name)

    if runner is None:
        mass = sum(v for v in params.values() if isinstance(v, (int, float))) * 0.01
        # dry-run: pull whichever force key the load case uses (varies per part)
        force_kN = next(
            (case[k] for k in ("seat_force_kN", "crushing_force_kN", "force_kN") if k in case),
            100.0,
        )
        stress = max(force_kN * 1.5, 1.0)
        metrics = {"max_von_mises_MPa": stress, "max_displacement_mm": 0.5, "mass_kg": mass}
    else:
        gen = runner["generate"](part=part, model=model, params=params)
        meshed = runner["mesh"](step_path=gen["step_path"])
        solved = runner["solve"](inp_path=meshed["inp"], material=material, load_case=case)
        metrics = runner["extract_metrics"](frd_path=solved["frd_path"])

    materials = load_materials()
    m = materials[material]
    fitness = score(
        mass_kg=metrics.get("mass_kg", 0.0),
        max_stress_MPa=metrics["max_von_mises_MPa"],
        material_yield_MPa=m["yield_strength_MPa"],
        archard_k_relative=m.get("archard_k_relative", 1.0),
    )

    record = {
        "ts": int(time.time()),
        "part": part,
        "model": model,
        "params": params,
        "material": material,
        "load_case": load_case_name,
        "metrics": metrics,
        "fitness": fitness,
    }
    _append_history(part, model, record)
    return record


def _append_history(part: str, model: str, record: dict[str, Any]) -> None:
    hist = TEMPLATES / part / "instances" / f"{model}.history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        f.write(json.dumps(record) + "\n")


def rank(records: Iterable[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: r["fitness"]["composite"], reverse=True)[:top_k]


def sweep(
    *,
    part: str,
    model: str,
    param_grid: list[dict[str, Any]] | None = None,
    sweeps: dict[str, list[Any]] | None = None,
    runner: dict[str, Callable] | None = None,
) -> list[dict[str, Any]]:
    """Sweep a parameter grid for one (part, model).
    Either provide explicit `param_grid` or a `sweeps` dict of axes to
    cartesian-product over the model's baseline params."""
    inst = load_instance(part, model)
    base = dict(inst["params"])

    if param_grid is None:
        if sweeps is None:
            param_grid = [base]
        else:
            import itertools
            keys = list(sweeps.keys())
            param_grid = []
            for combo in itertools.product(*[sweeps[k] for k in keys]):
                p = dict(base)
                for k, v in zip(keys, combo):
                    p[k] = v
                param_grid.append(p)

    results = [evaluate(part=part, model=model, params=p, runner=runner)
               for p in param_grid]
    return rank(results, top_k=len(results))


def list_catalog() -> dict[str, Any]:
    """Return current state: parts × models with instance coverage."""
    parts = yaml.safe_load((CATALOG / "parts.yaml").read_text())["parts"]
    models = yaml.safe_load((CATALOG / "models.yaml").read_text())["models"]
    coverage: dict[str, dict[str, bool]] = {}
    for part, info in parts.items():
        coverage[part] = {}
        inst_dir = TEMPLATES / part / "instances"
        existing = set(p.stem for p in inst_dir.glob("*.yaml")) if inst_dir.exists() else set()
        for model in models:
            coverage[part][model] = model in existing
    return {"parts": list(parts), "models": list(models), "coverage": coverage}


if __name__ == "__main__":
    # Smoke test: dry-run sweep on PE_400x600 toggle plate
    out = sweep(
        part="toggle_plate",
        model="PE_400x600",
        sweeps={"thickness_mm": [24, 28, 32], "web_height_mm": [50, 60, 70]},
    )
    print(json.dumps(out[:3], indent=2))
