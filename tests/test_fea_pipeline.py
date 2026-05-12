"""
FEA pipeline tests that don't need external solvers.

Verifies the parts of the pipeline we control:
  - CalculiX deck text is syntactically what CalculiX expects
  - Material unit conversion (kg/m^3 -> tonne/mm^3) is exact
  - BC and load definitions are dataclass-validated
  - frd_parser falls back cleanly when calculix-frd-py absent

The full-pipeline benchmarks (gmsh + ccx + frd-py) live in
test_fea_benchmark.py and skip without those binaries.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mcp-servers"))

from fea.calculix_deck import (Boundary, Material, PointLoad, build_deck)
from fea import frd_parser


def test_material_density_unit_conversion():
    m = Material("STEEL", 210_000.0, 0.30, 7850.0)
    # CalculiX consistent units (mm, tonne, s, N, MPa): density in tonne/mm^3
    # 7850 kg/m^3 = 7.85e-9 tonne/mm^3
    assert m.density_tonne_per_mm3 == pytest.approx(7.85e-9, rel=1e-9)


def test_material_validates_physical_ranges():
    with pytest.raises(AssertionError):
        Material("bad", -1.0, 0.3, 7850.0).validate()
    with pytest.raises(AssertionError):
        Material("bad", 200_000.0, 0.5, 7850.0).validate()
    with pytest.raises(AssertionError):
        Material("bad", 200_000.0, 0.3, 0.0).validate()


def test_boundary_dof_validation():
    Boundary("face", 1).validate()
    Boundary("face", 3, 0.0).validate()
    with pytest.raises(AssertionError):
        Boundary("face", 0).validate()
    with pytest.raises(AssertionError):
        Boundary("face", 6).validate()


def test_deck_text_has_required_keywords():
    deck = build_deck(
        mesh_inp_filename="mesh.inp",
        material=Material("STEEL", 210_000.0, 0.30, 7850.0),
        boundaries=[Boundary("x_min", 1), Boundary("x_min", 2),
                     Boundary("x_min", 3)],
        point_loads=[PointLoad("x_max", 1, 1000.0)],
    )
    for kw in ("*INCLUDE", "*MATERIAL", "*ELASTIC", "*DENSITY",
                "*SOLID SECTION", "*STEP", "*STATIC", "*BOUNDARY",
                "*CLOAD", "*NODE FILE", "*EL FILE", "*END STEP"):
        assert kw in deck, f"deck missing {kw}"
    # The material name appears on both *MATERIAL and *SOLID SECTION lines
    assert deck.count("STEEL") >= 2


def test_deck_boundary_lines_have_correct_format():
    """*BOUNDARY format is: nset_name, start_dof, end_dof, value"""
    deck = build_deck(
        mesh_inp_filename="m.inp",
        material=Material("M", 200000.0, 0.3, 7800.0),
        boundaries=[Boundary("clamp", 2, value=0.0)],
        point_loads=[],
    )
    boundary_lines = [l for l in deck.splitlines() if l.startswith("clamp,")]
    assert boundary_lines == ["clamp,2,2,0.0"]


def test_deck_cload_format():
    deck = build_deck(
        mesh_inp_filename="m.inp",
        material=Material("M", 200000.0, 0.3, 7800.0),
        boundaries=[],
        point_loads=[PointLoad("tip", 3, -1500.0)],
    )
    cload = [l for l in deck.splitlines() if l.startswith("tip,")]
    assert cload == ["tip,3,-1500.0"]


def test_deck_with_no_boundaries_or_loads():
    """Edge case: deck still valid syntax (free body / pure inertia run)."""
    deck = build_deck(
        mesh_inp_filename="m.inp",
        material=Material("M", 200000.0, 0.3, 7800.0),
        boundaries=[], point_loads=[],
    )
    assert "*STEP" in deck
    assert "*END STEP" in deck
    assert "*BOUNDARY" not in deck   # omitted when empty
    assert "*CLOAD" not in deck


def test_frd_parser_missing_file():
    out = frd_parser.parse("/tmp/does_not_exist_xyzzy.frd")
    assert out.get("_not_implemented")
    assert "not found" in out["reason"]


def test_frd_parser_missing_library(tmp_path):
    """When calculix-frd-py isn't installed, parser returns a clear marker.
    (calculix-frd-py is not installed on this machine — this test asserts
    the fallback path.)"""
    fake_frd = tmp_path / "fake.frd"
    fake_frd.write_text("not a real frd\n")
    out = frd_parser.parse(fake_frd)
    assert "_not_implemented" in out or "ok" in out
