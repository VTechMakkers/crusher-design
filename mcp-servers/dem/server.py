"""
DEM (Discrete Element Method) MCP server.

Wraps LIGGGHTS-PUBLIC (LAMMPS GPU package) for crusher chamber simulation.
This is the genuine competitive edge: predict TPH, wear distribution, and
particle breakage from geometry — what Sandvik/Metso do internally and what
Apollo/Propel cannot do.

Pipeline:
  CadQuery STEP -> STL meshes for jaw plates ->
  LIGGGHTS DEM input deck (templated per part class) ->
  liggghts run (GPU) ->
  parse dump files -> {TPH_kg_per_s, wear_contact_map, P80_mm, energy_kWh_per_t}

Requires: liggghts binary on PATH, GPU build with USER-CUDA or KOKKOS package.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("crusher-dem")
except ImportError:
    class _NoMCP:
        def tool(self):
            return lambda fn: fn
        def run(self):
            raise RuntimeError("install `mcp` to run as MCP server")
    mcp = _NoMCP()

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "templates"
KNOWLEDGE = ROOT / "knowledge"
OUT_DIR = ROOT / "runs" / "dem"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"{binary} not found on PATH; install LIGGGHTS-PUBLIC")
    return path


def _load_ore(ore_name: str) -> dict[str, Any]:
    ores_file = KNOWLEDGE / "ores.yaml"
    if not ores_file.exists():
        return {"name": ore_name, "warning": "knowledge/ores.yaml not populated"}
    data = yaml.safe_load(ores_file.read_text())
    return data.get("ores", {}).get(ore_name, {"name": ore_name, "warning": "unknown ore"})


@mcp.tool()
def generate_deck(part: str, model: str, scenario: dict[str, Any]) -> dict[str, str]:
    """Generate a LIGGGHTS input deck for a (part, model) DEM scenario.

    `scenario` schema:
      {ore: 'basalt'|'granite'|..., feed_p80_mm: 250, target_tph: 65,
       throw_mm: 18, frequency_hz: 4.5, css_mm: 80,
       sim_duration_s: 5.0, n_particles: 50000, gpu: true}

    Returns paths to the generated input deck and STL working dir.
    """
    template_path = TEMPLATES / part / "dem_template.lammps"
    if not template_path.exists():
        raise FileNotFoundError(f"no dem_template.lammps for part: {part}")

    tpl = template_path.read_text()
    ore = _load_ore(scenario.get("ore", "basalt"))

    work = OUT_DIR / part / model / scenario.get("id", "default")
    work.mkdir(parents=True, exist_ok=True)

    # Substitution map. The .lammps template uses {{...}} placeholders.
    subs = {
        "{{n_particles}}": str(scenario.get("n_particles", 50000)),
        "{{feed_p80_mm}}": str(scenario.get("feed_p80_mm", 250)),
        "{{throw_mm}}": str(scenario.get("throw_mm", 18)),
        "{{frequency_hz}}": str(scenario.get("frequency_hz", 4.5)),
        "{{css_mm}}": str(scenario.get("css_mm", 80)),
        "{{sim_duration_s}}": str(scenario.get("sim_duration_s", 5.0)),
        "{{ore_density_kg_m3}}": str(ore.get("density_kg_m3", 2900)),
        "{{ore_youngs_GPa}}": str(ore.get("youngs_modulus_GPa", 50)),
        "{{ore_poisson}}": str(ore.get("poisson_ratio", 0.25)),
        "{{ore_friction}}": str(ore.get("friction_coefficient", 0.6)),
        "{{ore_restitution}}": str(ore.get("restitution_coefficient", 0.4)),
        "{{stl_fixed_jaw}}": str(work / "fixed_jaw.stl"),
        "{{stl_swing_jaw}}": str(work / "swing_jaw.stl"),
        "{{output_dir}}": str(work / "out"),
        "{{gpu_clause}}": ("package gpu force/neigh 0 0 1.0\n"
                          "suffix gpu") if scenario.get("gpu", True) else "",
    }
    deck = tpl
    for k, v in subs.items():
        deck = deck.replace(k, v)
    deck_path = work / "in.crusher"
    deck_path.write_text(deck)
    (work / "out").mkdir(exist_ok=True)
    return {"deck_path": str(deck_path), "work_dir": str(work),
            "expected_stls": [subs["{{stl_fixed_jaw}}"], subs["{{stl_swing_jaw}}"]]}


@mcp.tool()
def run(deck_path: str, mpi_ranks: int = 1, gpu: bool = True) -> dict[str, Any]:
    """Run LIGGGHTS on a prepared deck. Returns {work_dir, returncode, stdout_tail}."""
    _require("liggghts")
    deck = Path(deck_path).resolve()
    work = deck.parent
    cmd = (["mpirun", "-np", str(mpi_ranks)] if mpi_ranks > 1 else []) + \
          ["liggghts", "-in", deck.name]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    return {
        "work_dir": str(work),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:] if proc.returncode else "",
    }


@mcp.tool()
def extract_metrics(work_dir: str) -> dict[str, Any]:
    """Parse LIGGGHTS dump + log to extract crusher metrics.

    Requires a VTK reader (e.g. `pyvista` or `meshio`) plus a thermo-log
    parser to compute:
      - throughput_kg_per_s — mass exiting discharge region per sim second
      - wear_score          — Σ(contact_force × sliding_distance) over swing jaw
      - p80_mm              — 80th-percentile particle size at discharge
      - energy_kWh_per_t    — total particle work / mass crushed

    Returns a `_not_implemented` marker rather than fake numbers. Wire in
    pyvista + log scraping before relying on DEM results.
    """
    work = Path(work_dir).resolve()
    log = work / "log.lammps"
    have_log = log.exists()
    dumps_dir = work / "out"
    have_dumps = dumps_dir.exists() and any(dumps_dir.glob("dump_*.vtk"))
    return {
        "_not_implemented": True,
        "reason": "install pyvista + write parser; see module docstring",
        "log_present": have_log,
        "dump_files_present": have_dumps,
        "work_dir": str(work),
    }


@mcp.tool()
def quick_capacity_estimate(model: str, ore: str, css_mm: float) -> dict[str, Any]:
    """Handbook-only capacity estimate (Bond) as a fallback when DEM not available.
    Useful for sanity-checking DEM output."""
    ore_data = _load_ore(ore)
    wi = ore_data.get("bond_work_index_kWh_per_t", 14.0)
    catalog = yaml.safe_load((ROOT / "catalog" / "models.yaml").read_text())
    m = catalog["models"].get(model, {})
    motor_kW = m.get("motor_kW", 30)
    # F.C. Bond simplified: Q = 60 * Mn / (Wi * K) ; rough only
    Q_tph_low = 0.6 * motor_kW / (wi * 0.6)
    Q_tph_high = 1.4 * motor_kW / (wi * 0.4)
    return {
        "tph_range_handbook": [round(Q_tph_low, 1), round(Q_tph_high, 1)],
        "method": "Bond, simplified",
        "ore": ore_data,
        "model": model,
        "motor_kW": motor_kW,
        "caveat": "wide bounds; use DEM for narrow estimate",
    }


if __name__ == "__main__":
    mcp.run()
