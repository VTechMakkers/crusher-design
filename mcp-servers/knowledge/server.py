"""
Knowledge MCP server.

Exposes materials, manufacturing rules, standards, and a multi-source
resolver as tools. The resolver chooses which value to trust for a given
property by tier-ranking sources, with optional task-specific overrides.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-knowledge")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"
SOURCES_DIR = KNOWLEDGE / "sources"


@lru_cache(maxsize=8)
def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((KNOWLEDGE / f"{name}.yaml").read_text())


def _load_source_file(filename: str) -> dict[str, Any]:
    path = SOURCES_DIR / filename
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _collect_property_candidates(material: str, prop: str) -> list[dict[str, Any]]:
    """Walk all sources, collect every observed value for (material, prop)."""
    registry = _load("sources")["sources"]
    candidates: list[dict[str, Any]] = []

    # 1) Internal mill certs (tier 1)
    internal = _load_source_file("techmakkers_internal.yaml")
    for cert in internal.get("mill_certs", []) or []:
        if cert.get("material") == material:
            v = cert.get("properties", {}).get(prop)
            if v is not None:
                candidates.append({
                    "value": v, "source_id": "techmakkers_mill_cert",
                    "tier": 1, "date": cert.get("date_received"),
                    "reference": cert.get("heat_number"),
                })

    # 2) Per-source YAML files (typically tier 3-5 external)
    for source_id, meta in registry.items():
        storage = meta.get("storage", "")
        if not storage.startswith("knowledge/sources/"):
            continue
        filename = storage.rsplit("/", 1)[-1]
        data = _load_source_file(filename)
        mats = (data or {}).get("materials", {})
        if material in mats and prop in mats[material]:
            candidates.append({
                "value": mats[material][prop],
                "source_id": source_id,
                "tier": meta["tier"],
                "date": data.get("fetched"),
            })

    return candidates


@mcp.tool()
def resolve_property(material: str, prop: str, task: str | None = None) -> dict[str, Any]:
    """Resolve a material property by trust hierarchy.

    Returns {chosen, alternatives, reason, conflict_warning?}.
    - tier-ranks all candidate values (lower tier = higher trust)
    - applies task-specific source elevation from sources.yaml -> task_overrides
    - if top two candidates differ by >15%, sets conflict_warning
    """
    candidates = _collect_property_candidates(material, prop)
    if not candidates:
        return {"chosen": None, "alternatives": [],
                "reason": f"no source has {material}.{prop}"}

    registry = _load("sources")
    overrides = registry.get("task_overrides", {}).get(task or "", {})
    elevate = set(overrides.get("elevate", []))

    def sort_key(c):
        elevated_bonus = -10 if c["source_id"] in elevate else 0
        return (c["tier"] + elevated_bonus, -(hash(str(c.get("date", ""))) & 0xFFFF))

    ranked = sorted(candidates, key=sort_key)
    chosen = ranked[0]
    alternatives = ranked[1:]

    reason_parts = [f"tier {chosen['tier']} source: {chosen['source_id']}"]
    if chosen["source_id"] in elevate:
        reason_parts.append(f"elevated for task '{task}'")
    if not elevate and task:
        reason_parts.append(f"no task-specific override for '{task}'")

    out = {
        "property": f"{material}.{prop}",
        "chosen": chosen,
        "alternatives": alternatives,
        "reason": "; ".join(reason_parts),
    }

    if len(ranked) >= 2:
        v1, v2 = ranked[0]["value"], ranked[1]["value"]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != 0:
            delta = abs(v1 - v2) / abs(v1)
            if delta > 0.15:
                out["conflict_warning"] = (
                    f"top two sources disagree by {delta*100:.0f}% "
                    f"({ranked[0]['source_id']}={v1} vs {ranked[1]['source_id']}={v2}) — review"
                )
    return out


@mcp.tool()
def list_materials() -> list[str]:
    """All materials known across all sources."""
    seen: set[str] = set()
    seen.update(_load("materials").keys())
    for src in SOURCES_DIR.glob("*.yaml"):
        data = yaml.safe_load(src.read_text()) or {}
        seen.update((data.get("materials") or {}).keys())
    return sorted(seen)


@mcp.tool()
def list_sources() -> dict[str, Any]:
    """Source registry with tiers."""
    return _load("sources")["sources"]


@mcp.tool()
def wear_rank(materials: list[str]) -> list[dict[str, Any]]:
    """Rank materials by relative wear (lower archard_k_relative = better)."""
    rows = []
    for m in materials:
        r = resolve_property(m, "archard_k_relative", task="wear_life")
        if r["chosen"]:
            rows.append({"material": m, "value": r["chosen"]["value"],
                         "source": r["chosen"]["source_id"]})
    return sorted(rows, key=lambda r: r["value"])


@mcp.tool()
def get_manufacturing_rule(process: str, rule: str | None = None) -> Any:
    """Look up a manufacturing constraint by dotted path."""
    node = _load("manufacturing")
    for key in process.split("."):
        node = node[key]
    if rule is not None:
        return node[rule]
    return node


@mcp.tool()
def get_safety_factor(part_criticality: str, load_type: str = "static_load") -> float:
    std = _load("standards")
    return float(std["safety_factors"][load_type][part_criticality])


@mcp.tool()
def list_standards() -> list[str]:
    std = _load("standards")
    return [k for k in std if k != "safety_factors"]


@mcp.tool()
def get_standard(name: str) -> dict[str, Any]:
    return _load("standards")[name]


if __name__ == "__main__":
    mcp.run()
