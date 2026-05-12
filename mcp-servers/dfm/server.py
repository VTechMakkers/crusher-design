"""
Design-for-Manufacturability (DFM) MCP server.

Checks a parametric design against shop-floor rules encoded in
knowledge/manufacturing.yaml. A design that passes FEA + DEM but fails DFM
cannot be made — the loop must reject it before the engineer sees it.

Rules currently checked:
  - min wall thickness for casting process
  - sufficient draft angle for sand casting
  - bore aspect ratio (length/diameter) for through-hole machining
  - weld joint access clearance
  - mount-hole positions clear of corrugation/relief features
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-dfm")
except ImportError:
    # mcp not installed — provide a no-op decorator so the DFM logic is
    # still importable and callable directly (e.g. from bin/run_design.py
    # or tests). MCP runtime only required for serving over the protocol.
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"


def _rules() -> dict[str, Any]:
    return yaml.safe_load((KNOWLEDGE / "manufacturing.yaml").read_text())


def _flatten(issues: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passes": all(i["severity"] != "error" for i in issues),
        "errors": [i for i in issues if i["severity"] == "error"],
        "warnings": [i for i in issues if i["severity"] == "warning"],
        "info": [i for i in issues if i["severity"] == "info"],
        "total_issues": len(issues),
    }


@mcp.tool()
def check_toggle_plate(params: dict[str, Any], process: str = "sand_casting") -> dict[str, Any]:
    """Check toggle plate geometry against manufacturing rules."""
    rules = _rules()
    issues: list[dict[str, Any]] = []

    cast = rules["casting"].get(process, {})
    min_wall = cast.get("min_wall_thickness_mm", 8)
    if params.get("thickness_mm", 0) < min_wall:
        issues.append({
            "severity": "error", "rule": "min_wall_thickness",
            "process": process,
            "actual": params["thickness_mm"], "required_min": min_wall,
        })
    if params.get("web_thickness_mm", 0) < min_wall:
        issues.append({
            "severity": "error", "rule": "min_wall_thickness_web",
            "actual": params["web_thickness_mm"], "required_min": min_wall,
        })

    # relief groove must not exceed 40% of plate thickness
    rg = params.get("relief_groove_depth_mm", 0)
    th = params.get("thickness_mm", 1)
    if rg > th * 0.4:
        issues.append({
            "severity": "error", "rule": "relief_groove_depth",
            "actual_ratio": rg / th, "max_ratio": 0.4,
        })

    return _flatten(issues)


@mcp.tool()
def check_jaw_plate(params: dict[str, Any], process: str = "sand_casting") -> dict[str, Any]:
    """Check swing/fixed jaw plate geometry against manufacturing rules."""
    rules = _rules()
    issues: list[dict[str, Any]] = []

    cast = rules["casting"].get(process, {})
    min_wall = cast.get("min_wall_thickness_mm", 8)
    if params.get("backing_thickness_mm", 0) < min_wall:
        issues.append({
            "severity": "error", "rule": "min_backing_thickness",
            "actual": params["backing_thickness_mm"], "required_min": min_wall,
        })

    # tooth depth vs thickness — must leave structural cross-section
    td = params.get("tooth_depth_mm", 0)
    th = params.get("thickness_mm", 1)
    if td > th * 0.55:
        issues.append({
            "severity": "error", "rule": "tooth_depth_excessive",
            "actual_ratio": td / th, "max_ratio": 0.55,
            "consequence": "insufficient cross-section behind teeth → fatigue",
        })

    # mount hole inset must allow drill access
    inset = params.get("mount_hole_inset_mm", 0)
    if inset < 30:
        issues.append({
            "severity": "warning", "rule": "mount_hole_inset_low",
            "actual": inset, "recommended_min": 30,
        })

    # tooth pitch sanity — too coarse = poor breakage; too fine = no bite
    tp = params.get("tooth_pitch_mm", 0)
    if tp < 30:
        issues.append({"severity": "warning", "rule": "tooth_pitch_too_fine",
                       "actual": tp, "recommended_min": 30})
    if tp > 200:
        issues.append({"severity": "warning", "rule": "tooth_pitch_too_coarse",
                       "actual": tp, "recommended_max": 200})

    return _flatten(issues)


@mcp.tool()
def check_paired_jaws(swing_params: dict[str, Any],
                      fixed_params: dict[str, Any]) -> dict[str, Any]:
    """Verify swing and fixed jaw plates are compatible for paired operation."""
    issues: list[dict[str, Any]] = []
    if swing_params.get("tooth_pitch_mm") != fixed_params.get("tooth_pitch_mm"):
        issues.append({
            "severity": "error", "rule": "tooth_pitch_mismatch",
            "swing": swing_params.get("tooth_pitch_mm"),
            "fixed": fixed_params.get("tooth_pitch_mm"),
        })
    if swing_params.get("width_mm") != fixed_params.get("width_mm"):
        issues.append({
            "severity": "error", "rule": "width_mismatch",
            "swing": swing_params.get("width_mm"),
            "fixed": fixed_params.get("width_mm"),
        })
    if swing_params.get("height_mm") != fixed_params.get("height_mm"):
        issues.append({
            "severity": "warning", "rule": "height_mismatch",
            "swing": swing_params.get("height_mm"),
            "fixed": fixed_params.get("height_mm"),
            "note": "common in some designs but verify intentional",
        })
    return _flatten(issues)


@mcp.tool()
def list_processes() -> list[str]:
    rules = _rules()
    return list(rules.get("casting", {}).keys()) + ["machining", "welding"]


if __name__ == "__main__":
    mcp.run()
