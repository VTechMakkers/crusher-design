"""
FEA MCP server.

Production pipeline:
  STEP -> gmsh mesh (CalculiX INP with named node sets)
       -> calculix_deck.build_deck() with material + BC + loads
       -> ccx solver -> .frd
       -> frd_parser.parse() -> {max_von_mises_MPa, max_displacement_mm}

NAFEMS-style benchmarks live under mcp-servers/fea/benchmark.py and are
runnable via `python -m mcp-servers.fea.benchmark`. Test coverage in
tests/test_fea_pipeline.py exercises the pieces that don't need external
binaries; tests/test_fea_benchmark.py runs the full pipeline when
gmsh + ccx + calculix-frd-py are installed.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-fea")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"
OUT_DIR = ROOT / "runs" / "fea"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "mcp-servers"))
from fea import calculix_deck, frd_parser   # noqa: E402
from fea.gmsh_mesher import mesh_step as _mesh_step   # noqa: E402


def _materials() -> dict[str, Any]:
    return yaml.safe_load((KNOWLEDGE / "materials.yaml").read_text())


def _material_for_ccx(name: str) -> calculix_deck.Material:
    mats = _materials()
    if name not in mats:
        raise ValueError(f"unknown material {name!r}; known: {list(mats)}")
    m = mats[name]
    return calculix_deck.Material(
        name=name,
        youngs_modulus_MPa=m["youngs_modulus_GPa"] * 1000.0,
        poisson_ratio=m["poisson_ratio"],
        density_kg_m3=m["density_kg_m3"],
    )


@mcp.tool()
def mesh(step_path: str, mesh_size_mm: float = 8.0,
          variant_id: str | None = None) -> dict[str, Any]:
    """STEP -> CalculiX INP via gmsh."""
    step = Path(step_path).resolve()
    vid = variant_id or step.stem
    work = OUT_DIR / vid
    work.mkdir(parents=True, exist_ok=True)
    out_inp = work / f"{vid}.inp"
    result = _mesh_step(step, out_inp=out_inp, mesh_size_mm=mesh_size_mm)
    return {
        "inp_path": str(result.inp_path),
        "n_nodes": result.n_nodes,
        "n_elements": result.n_elements,
        "physical_groups": list(result.physical_groups.keys()),
    }


@mcp.tool()
def solve(inp_path: str, material: str,
           boundaries: list[dict[str, Any]],
           point_loads: list[dict[str, Any]]) -> dict[str, Any]:
    """Run CalculiX on a meshed INP with structured BC/load spec.

    `boundaries`: list of {nset_name, dof, value?}
    `point_loads`: list of {nset_name, dof, force_N}

    nset_name must reference a Physical Group defined in the gmsh INP.
    """
    if not shutil.which("ccx"):
        return {"_not_implemented": True,
                "reason": "install CalculiX (`ccx`) to run real FEA"}
    inp = Path(inp_path).resolve()
    if not inp.exists():
        raise FileNotFoundError(f"INP not found: {inp}")
    mat = _material_for_ccx(material)
    deck_text = calculix_deck.build_deck(
        mesh_inp_filename=inp.name,
        material=mat,
        boundaries=[calculix_deck.Boundary(**b) for b in boundaries],
        point_loads=[calculix_deck.PointLoad(**p) for p in point_loads],
    )
    work = inp.parent
    deck_path = work / f"{inp.stem}_job.inp"
    deck_path.write_text(deck_text)
    subprocess.run(["ccx", deck_path.stem], cwd=work,
                   check=True, capture_output=True)
    frd_path = work / f"{deck_path.stem}.frd"
    return {"frd_path": str(frd_path), "deck_path": str(deck_path),
            "material": material}


@mcp.tool()
def extract_metrics(frd_path: str) -> dict[str, Any]:
    """Parse a CalculiX .frd file to extract max von Mises stress + max
    displacement. Falls back to a clear `_not_implemented` marker if the
    `calculix-frd-py` library is not installed — see frd_parser.py."""
    return frd_parser.parse(frd_path)


@mcp.tool()
def run_benchmark(name: str = "tension_bar") -> dict[str, Any]:
    """Run a NAFEMS-style benchmark and return the comparison vs analytical."""
    from fea import benchmark
    work = OUT_DIR / "benchmarks" / name
    work.mkdir(parents=True, exist_ok=True)
    fn = getattr(benchmark, name, None)
    if fn is None:
        return {"error": f"unknown benchmark {name!r}",
                "available": ["tension_bar", "cantilever_bending"]}
    result = fn(work_dir=work)
    return {
        "name": result.name, "passed": result.passed,
        "analytical": result.analytical, "fea": result.fea,
        "relative_error": result.relative_error,
        "tolerance": result.tolerance, "notes": result.notes,
    }


if __name__ == "__main__":
    mcp.run()
