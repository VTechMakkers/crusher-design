"""
Smoke tests — verify the architecture is internally consistent.
Run with:  python -m pytest tests/
"""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---- catalog ----

def test_catalog_has_models():
    models = yaml.safe_load((ROOT / "catalog/models.yaml").read_text())["models"]
    assert len(models) >= 8, "catalog should list at least the PE/PEX line"
    for name, m in models.items():
        assert "gape_mm" in m and len(m["gape_mm"]) == 2
        assert "motor_kW" in m


def test_catalog_parts_are_consistent():
    parts = yaml.safe_load((ROOT / "catalog/parts.yaml").read_text())["parts"]
    assert "toggle_plate" in parts
    assert "swing_jaw_plate" in parts
    assert "fixed_jaw_plate" in parts
    for name, info in parts.items():
        assert info["criticality"] in {"sacrificial", "structural", "safety_critical"}
        if info.get("has_geometry"):
            assert (ROOT / f"templates/{name}/geometry.py").exists(), \
                f"{name} claims has_geometry but geometry.py missing"


# ---- knowledge ----

def test_knowledge_files_parse():
    for f in ("materials.yaml", "manufacturing.yaml", "standards.yaml",
              "ores.yaml", "sources.yaml"):
        yaml.safe_load((ROOT / "knowledge" / f).read_text())


def test_sources_registry_is_complete():
    reg = yaml.safe_load((ROOT / "knowledge/sources.yaml").read_text())
    assert "sources" in reg
    for sid, meta in reg["sources"].items():
        assert "tier" in meta
        assert 1 <= meta["tier"] <= 6


# ---- templates ----

@pytest.mark.parametrize("part", ["toggle_plate", "swing_jaw_plate", "fixed_jaw_plate"])
def test_template_has_required_files(part):
    base = ROOT / "templates" / part
    assert (base / "geometry.py").exists()
    assert (base / "metadata.yaml").exists()
    assert (base / "load_cases.yaml").exists()
    assert (base / "instances").is_dir()


def test_swing_and_fixed_jaw_pitch_consistency_pe400():
    """If both jaw plates have a PE_400x600 instance, tooth pitch must match."""
    swing = yaml.safe_load(
        (ROOT / "templates/swing_jaw_plate/instances/PE_400x600.yaml").read_text())
    fixed = yaml.safe_load(
        (ROOT / "templates/fixed_jaw_plate/instances/PE_400x600.yaml").read_text())
    assert swing["params"]["tooth_pitch_mm"] == fixed["params"]["tooth_pitch_mm"]
    assert swing["params"]["width_mm"] == fixed["params"]["width_mm"]


# ---- fitness ----

def test_fitness_basic():
    from loop.fitness import score
    out = score(mass_kg=42.0, max_stress_MPa=300.0,
                material_yield_MPa=850.0, archard_k_relative=0.35)
    assert 0.0 < out["composite"] <= 1.0
    assert out["safety_factor_value"] == pytest.approx(850.0 / 300.0)


def test_dem_fitness_basic():
    from loop.dem_fitness import dem_score, combined_score
    out = dem_score(
        throughput_kg_per_s=18.0, target_tph=65.0,
        p80_mm=44.0, p80_target_mm=44.0,
        wear_uniformity_index=0.7, energy_kWh_per_t=1.6,
    )
    assert 0.0 <= out["dem_composite"] <= 1.0

    structural = {"composite": 0.6}
    dem = {"dem_composite": 0.7}
    combined = combined_score(structural, dem, structural_weight=0.5)
    assert combined["total"] == pytest.approx(0.65)


# ---- loop ----

def test_loop_dry_run_toggle_plate():
    from loop import design_loop
    out = design_loop.sweep(
        part="toggle_plate", model="PE_400x600",
        sweeps={"thickness_mm": [26, 28], "web_height_mm": [55, 60]},
    )
    assert len(out) == 4
    for record in out:
        assert "fitness" in record
        assert record["fitness"]["composite"] > 0


def test_loop_list_catalog():
    from loop import design_loop
    cat = design_loop.list_catalog()
    assert "parts" in cat
    assert "models" in cat
    assert "coverage" in cat
    assert cat["coverage"]["toggle_plate"]["PE_400x600"] is True
    assert cat["coverage"]["swing_jaw_plate"]["PE_400x600"] is True


# ---- DFM ----

def test_dfm_toggle_plate_rejects_thin_wall():
    spec = importlib.util.spec_from_file_location(
        "dfm", ROOT / "mcp-servers/dfm/server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dfm"] = mod
    spec.loader.exec_module(mod)
    result = mod.check_toggle_plate({"thickness_mm": 5,
                                      "web_thickness_mm": 12,
                                      "relief_groove_depth_mm": 6})
    assert not result["passes"]
    assert any(e["rule"] == "min_wall_thickness" for e in result["errors"])


def test_dfm_paired_jaws_pitch_mismatch():
    spec = importlib.util.spec_from_file_location(
        "dfm", ROOT / "mcp-servers/dfm/server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dfm"] = mod
    spec.loader.exec_module(mod)
    result = mod.check_paired_jaws(
        {"tooth_pitch_mm": 90, "width_mm": 400, "height_mm": 700},
        {"tooth_pitch_mm": 85, "width_mm": 400, "height_mm": 700},
    )
    assert not result["passes"]


# ---- end-to-end driver ----

def test_run_design_dry_run():
    """Driver should run end-to-end in dry-run mode and emit ranked top-K."""
    result = subprocess.run(
        [sys.executable, "bin/run_design.py", "swing_jaw_plate", "PE_400x600",
         "--ore", "basalt", "--css", "80", "--tph", "65",
         "--sweep", "tooth_pitch_mm=80,90,100",
         "--top-k", "2", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["part"] == "swing_jaw_plate"
    assert out["candidates_total"] == 3
    assert out["dfm_passed"] >= 1
    assert len(out["top_k"]) >= 1
    assert out["ranking_mode"] == "composite"
    for rec in out["top_k"]:
        assert "fitness" in rec
        assert "dem" in rec
        assert "combined" in rec


def test_run_design_dry_run_pareto():
    """--pareto flag selects from the Pareto front with crowding distance."""
    result = subprocess.run(
        [sys.executable, "bin/run_design.py", "swing_jaw_plate", "PE_400x600",
         "--sweep", "tooth_pitch_mm=80,90,100;tooth_depth_mm=18,22,26",
         "--top-k", "3", "--dry-run", "--pareto"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["ranking_mode"] == "pareto"
    assert out["candidates_total"] == 9
    assert 1 <= len(out["top_k"]) <= 3
