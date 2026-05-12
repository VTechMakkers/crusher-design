"""
Assembly MCP server.

Exposes whole-machine queries: load assemblies, traverse load paths,
check cross-part consistency, surface material distribution. Pairs with
the system_fitness + pareto modules to give the LLM architect a
machine-level view.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-assembly")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from assembly.crusher_assembly import load, list_assemblies


@mcp.tool()
def list_assemblies_tool() -> list[str]:
    """All assembly definitions available."""
    return list_assemblies(ROOT)


@mcp.tool()
def get_assembly(model: str) -> dict[str, Any]:
    """Full assembly definition by model name."""
    asm = load(model, ROOT)
    return {
        "model": asm.model,
        "family": asm.family,
        "parts": {nid: {"part_class": p.part_class, "instance": p.instance,
                        "role": p.role} for nid, p in asm.parts.items()},
        "n_connections": len(asm.connections),
        "system_targets": asm.system_targets,
        "pareto_objectives": load(model, ROOT)._raw_yaml().get("pareto_objectives", [])
            if hasattr(asm, "_raw_yaml") else
            yaml.safe_load((ROOT / "assembly" / f"{model}.yaml").read_text())
                .get("pareto_objectives", []),
    }


@mcp.tool()
def load_path(model: str, source: str = "swing_jaw",
              sink: str = "main_frame") -> list[list[str]]:
    """All structural paths from source to sink — useful for tracing how
    forces flow from rock contact to ground reaction."""
    asm = load(model, ROOT)
    return asm.load_path(source, sink)


@mcp.tool()
def check_assembly_consistency(model: str) -> dict[str, Any]:
    """Verify cross-part constraints (e.g. paired jaws have matching pitch)."""
    asm = load(model, ROOT)
    return asm.check_consistency()


@mcp.tool()
def assembly_materials(model: str) -> dict[str, list[str]]:
    """Which parts use which materials."""
    asm = load(model, ROOT)
    return asm.material_list()


@mcp.tool()
def neighbors(model: str, node: str, kind: str | None = None) -> list[str]:
    """Neighboring parts in the assembly graph (filter by connection kind)."""
    asm = load(model, ROOT)
    return asm.neighbors(node, kind=kind)


@mcp.tool()
def part_classes_used(model: str) -> list[str]:
    """Distinct part classes referenced by this assembly."""
    asm = load(model, ROOT)
    return sorted(asm.part_classes())


if __name__ == "__main__":
    mcp.run()
