"""
Geometry MCP server.

Exposes parametric part-class generators. Works over (part, model) pairs:
- `templates/<part>/geometry.py` provides the parametric script
- `templates/<part>/instances/<model>.yaml` provides validated params for that model

Run: python -m mcp_servers.geometry.server  (or via Claude Code MCP config)
"""
from __future__ import annotations
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-geometry")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
CATALOG = ROOT / "catalog"
OUT_DIR = ROOT / "runs" / "geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_part_module(part: str):
    geom = TEMPLATES / part / "geometry.py"
    if not geom.exists():
        raise FileNotFoundError(f"no geometry.py for part class: {part}")
    spec = importlib.util.spec_from_file_location(f"geom_{part}", geom)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _params_cls(mod):
    return next(v for k, v in vars(mod).items()
                if k.endswith("Params") and hasattr(v, "__dataclass_fields__"))


@mcp.tool()
def list_models() -> list[str]:
    """All crusher models in the catalog."""
    return list(yaml.safe_load((CATALOG / "models.yaml").read_text())["models"].keys())


@mcp.tool()
def list_parts(only_with_geometry: bool = True) -> list[str]:
    """All part classes in the catalog (optionally only those with geometry.py)."""
    parts = yaml.safe_load((CATALOG / "parts.yaml").read_text())["parts"]
    return [p for p, info in parts.items()
            if (info.get("has_geometry") if only_with_geometry else True)]


@mcp.tool()
def list_instances(part: str) -> list[str]:
    """All crusher models for which this part has a validated instance YAML."""
    d = TEMPLATES / part / "instances"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


@mcp.tool()
def get_instance(part: str, model: str) -> dict[str, Any]:
    """Return the validated configuration for (part, model)."""
    path = TEMPLATES / part / "instances" / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no instance: {part} on {model}")
    return yaml.safe_load(path.read_text())


@mcp.tool()
def get_param_schema(part: str) -> dict[str, Any]:
    """Parameter schema (names, defaults, bounds) for a part class."""
    mod = _load_part_module(part)
    cls = _params_cls(mod)
    inst = cls()
    return {f.name: {
        "type": f.type if isinstance(f.type, str) else f.type.__name__,
        "default": getattr(inst, f.name),
    } for f in fields(cls)}


@mcp.tool()
def generate(part: str, model: str | None = None,
             params: dict[str, Any] | None = None,
             variant_id: str | None = None) -> dict[str, Any]:
    """Generate STEP geometry.
    - If `params` given, use those directly.
    - Else if `model` given, load validated params from instance YAML.
    - Returns {step_path, params, variant_id, part, model}."""
    mod = _load_part_module(part)
    cls = _params_cls(mod)

    if params is None:
        if model is None:
            raise ValueError("provide either `params` or `model`")
        inst = get_instance(part, model)
        params = inst["params"]

    p = cls(**params)
    vid = variant_id or f"v_{abs(hash(tuple(sorted(params.items())))) % 10**8}"
    out_path = OUT_DIR / part / (model or "free") / f"{vid}.step"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mod.export_step(p, out_path)
    return {"step_path": str(out_path), "variant_id": vid,
            "part": part, "model": model, "params": params}


@mcp.tool()
def sweep_params(part: str, base_params: dict[str, Any],
                 sweeps: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate the cartesian product of parameter sweeps over `base_params`.
    Example: sweeps={'thickness_mm': [22, 26, 30], 'web_height_mm': [50, 60]}.
    Does NOT export STEP — returns the parameter dicts only. Pair with `generate`."""
    import itertools
    keys = list(sweeps.keys())
    values_lists = [sweeps[k] for k in keys]
    out = []
    for combo in itertools.product(*values_lists):
        p = dict(base_params)
        for k, v in zip(keys, combo):
            p[k] = v
        out.append(p)
    return out


if __name__ == "__main__":
    mcp.run()
