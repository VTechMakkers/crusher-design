"""
Ingest CLI tests.

Verifies the ingest functions directly (not the prompt loop), the file
writes are atomic + idempotent, validation gates work, and the status
dashboard reports the expected counts.
"""
from __future__ import annotations
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_bin(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bin" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sandbox_root(tmp_path):
    """Copy the repo's minimum-needed files into a tmp dir so tests don't
    mutate real instance YAMLs or techmakkers_internal.yaml."""
    for sub in ("catalog", "templates", "knowledge", "mcp-servers", "assembly"):
        shutil.copytree(ROOT / sub, tmp_path / sub)
    # bin not needed; we'll import directly
    return tmp_path


def test_ingest_drawing_writes_real_instance(sandbox_root):
    ingest = _load_bin("ingest_drawing")
    out = ingest.ingest_drawing(
        part="toggle_plate", model="PE_400x600",
        params={"length_mm": 540, "width_mm": 180, "thickness_mm": 30,
                "seat_radius_mm": 35, "web_height_mm": 60, "web_thickness_mm": 12,
                "relief_groove_depth_mm": 8, "relief_groove_width_mm": 8},
        material="AR400",
        drawing_ref="TP-PE400-Rev3",
        revision="3", ingested_by="vt", date="2026-05-11",
        root=sandbox_root,
    )
    assert out["validated"]
    written = Path(out["written_to"])
    data = yaml.safe_load(written.read_text())
    assert data["model"] == "PE_400x600"
    assert data["material"] == "AR400"
    assert data["params"]["thickness_mm"] == 30
    assert data["provenance"]["drawing_ref"] == "TP-PE400-Rev3"
    assert "PLACEHOLDER" not in data["notes"].upper()


def test_ingest_drawing_rejects_out_of_bounds(sandbox_root):
    ingest = _load_bin("ingest_drawing")
    with pytest.raises(ValueError):
        ingest.ingest_drawing(
            part="toggle_plate", model="PE_400x600",
            params={"thickness_mm": 5},   # below validate() lower bound
            material="AR400", drawing_ref="invalid_test",
            root=sandbox_root,
        )


def test_ingest_drawing_rejects_unknown_param(sandbox_root):
    ingest = _load_bin("ingest_drawing")
    with pytest.raises(ValueError, match="unknown params"):
        ingest.ingest_drawing(
            part="toggle_plate", model="PE_400x600",
            params={"banana_mm": 99},
            material="AR400", drawing_ref="x",
            root=sandbox_root,
        )


def test_ingest_drawing_rejects_dfm_failure(sandbox_root):
    """Web thickness 6mm passes validate() (>= 6) but fails the DFM rule
    that requires >= 8mm min wall thickness for sand casting."""
    ingest = _load_bin("ingest_drawing")
    with pytest.raises(ValueError, match="DFM"):
        ingest.ingest_drawing(
            part="toggle_plate", model="PE_400x600",
            params={"length_mm": 540, "width_mm": 180, "thickness_mm": 30,
                    "seat_radius_mm": 35, "web_height_mm": 60,
                    "web_thickness_mm": 6,    # passes validate() but < DFM min wall (8)
                    "relief_groove_depth_mm": 8,
                    "relief_groove_width_mm": 8},
            material="AR400", drawing_ref="x",
            root=sandbox_root,
        )


def test_ingest_mill_cert_appends(sandbox_root):
    ingest = _load_bin("ingest_mill_cert")
    out = ingest.ingest_mill_cert(
        material="Mn13", heat_number="MN13-2026-0042",
        supplier="Test Foundry", date_received="2026-04-12",
        properties={"yield_strength_MPa": 395, "ultimate_strength_MPa": 870,
                     "hardness_HBW_as_cast": 210, "elongation_pct": 38},
        composition={"C_pct": 1.18, "Mn_pct": 12.4, "Si_pct": 0.55},
        root=sandbox_root,
    )
    assert out["appended"]
    assert out["total_certs"] == 1
    data = yaml.safe_load(
        (sandbox_root / "knowledge/sources/techmakkers_internal.yaml").read_text())
    certs = data["mill_certs"]
    assert len(certs) == 1
    assert certs[0]["heat_number"] == "MN13-2026-0042"
    assert certs[0]["properties"]["yield_strength_MPa"] == 395
    assert certs[0]["composition"]["C_pct"] == 1.18


def test_ingest_mill_cert_rejects_unknown_material(sandbox_root):
    ingest = _load_bin("ingest_mill_cert")
    with pytest.raises(ValueError, match="unknown material"):
        ingest.ingest_mill_cert(
            material="Unobtanium", heat_number="X1",
            supplier="x", date_received="2026-01-01",
            properties={"yield_strength_MPa": 1000},
            root=sandbox_root,
        )


def test_ingest_mill_cert_detects_duplicate_heat_number(sandbox_root):
    ingest = _load_bin("ingest_mill_cert")
    args = dict(material="Mn13", heat_number="DUP-001", supplier="x",
                 date_received="2026-01-01",
                 properties={"yield_strength_MPa": 395},
                 root=sandbox_root)
    ingest.ingest_mill_cert(**args)
    second = ingest.ingest_mill_cert(**args)
    assert second["duplicate_heat_number"]
    assert second["total_certs"] == 2  # appended anyway (history is immutable)


def test_data_status_counts_real_vs_placeholder(sandbox_root):
    status_mod = _load_bin("data_status")
    before = status_mod.assess_data_status(sandbox_root)
    placeholder_pairs_before = before["instances"]["placeholder"]
    real_before = before["instances"]["real"]

    # ingest one toggle plate as real
    ingest = _load_bin("ingest_drawing")
    ingest.ingest_drawing(
        part="toggle_plate", model="PE_400x600",
        params={"length_mm": 540, "width_mm": 180, "thickness_mm": 30,
                "seat_radius_mm": 35, "web_height_mm": 60, "web_thickness_mm": 12,
                "relief_groove_depth_mm": 8, "relief_groove_width_mm": 8},
        material="AR400", drawing_ref="TP-PE400-Rev3",
        root=sandbox_root,
    )

    after = status_mod.assess_data_status(sandbox_root)
    assert after["instances"]["real"] == real_before + 1
    assert after["instances"]["placeholder"] == placeholder_pairs_before - 1
    assert after["realness_score_0_to_1"] > before["realness_score_0_to_1"]


def test_data_status_counts_mill_certs(sandbox_root):
    status_mod = _load_bin("data_status")
    before = status_mod.assess_data_status(sandbox_root)
    ingest = _load_bin("ingest_mill_cert")
    ingest.ingest_mill_cert(
        material="Mn13", heat_number="STAT-1", supplier="x",
        date_received="2026-01-01",
        properties={"yield_strength_MPa": 400},
        root=sandbox_root,
    )
    after = status_mod.assess_data_status(sandbox_root)
    assert after["internal_data"]["mill_certs"] == before["internal_data"]["mill_certs"] + 1
    assert "Mn13" in after["materials"]["certified_list"]


def test_cli_data_status_runs():
    """Bare CLI invocation against the real repo prints a status report."""
    proc = subprocess.run(
        [sys.executable, "bin/data_status.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PARTS" in proc.stdout
    assert "INSTANCES" in proc.stdout
    assert "REALNESS" in proc.stdout


def test_cli_data_status_json():
    proc = subprocess.run(
        [sys.executable, "bin/data_status.py", "--json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    import json
    out = json.loads(proc.stdout)
    assert "parts" in out
    assert "instances" in out
    assert 0.0 <= out["realness_score_0_to_1"] <= 1.0
