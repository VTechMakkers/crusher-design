"""
Full-pipeline NAFEMS-style benchmarks.

Requires gmsh (Python API), CalculiX (`ccx` on PATH), and
calculix-frd-py. The whole module skips cleanly when any of those is
missing; on a properly provisioned machine each test runs in seconds
and gates the pipeline against analytical solutions.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcp-servers"))


def _require_external_tools():
    try:
        import gmsh  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("gmsh Python API not installed (`pip install gmsh`)")
    if not shutil.which("ccx"):
        pytest.skip("CalculiX `ccx` not on PATH (`brew install calculix-ccx`)")
    try:
        import calculix_frd_py  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("calculix-frd-py not installed (`pip install calculix-frd-py`)")


def test_tension_bar_matches_analytical(tmp_path):
    """Uniform tension: no stress concentration, no shear deflection.
    FEA should match σ and δ to within 2-3% with a fine second-order mesh."""
    _require_external_tools()
    from fea.benchmark import tension_bar
    result = tension_bar(work_dir=tmp_path)
    assert result.passed, (
        f"tension benchmark failed:\n"
        f"  analytical: {result.analytical}\n"
        f"  fea:        {result.fea}\n"
        f"  rel_err:    {result.relative_error}\n"
        f"  tolerance:  {result.tolerance}\n"
    )


def test_cantilever_tip_deflection_matches_bernoulli_euler(tmp_path):
    """L/h=10 cantilever — Bernoulli-Euler accurate to ~2%; FEA should
    match tip deflection to within 5% with second-order tets."""
    _require_external_tools()
    from fea.benchmark import cantilever_bending
    result = cantilever_bending(work_dir=tmp_path)
    assert result.passed, (
        f"cantilever benchmark failed:\n"
        f"  analytical: {result.analytical}\n"
        f"  fea:        {result.fea}\n"
        f"  rel_err:    {result.relative_error}\n"
        f"  tolerance:  {result.tolerance}\n"
    )
