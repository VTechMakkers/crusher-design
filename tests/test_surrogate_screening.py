"""
Surrogate-screening tests.

Splits into:
  - candidate enumeration + constraint filtering + feature overrides
    (no torch required)
  - full screen_with_surrogate pipeline (torch required, skipped cleanly
    when not installed)
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assembly.crusher_assembly import load as load_assembly
from loop.surrogate_screening import enumerate_candidates
from loop.assembly_features import extract


# -------- non-torch tests --------------------------------------------------

def test_enumerate_candidates_respects_paired_jaw_constraint():
    """3x3 raw combinations of jaw pitches; matching-pitch constraint keeps 3."""
    asm = load_assembly("PE_400x600", ROOT)
    cands = enumerate_candidates(
        assembly=asm,
        sweep_config={
            "swing_jaw_plate": {"tooth_pitch_mm": [80, 90, 100]},
            "fixed_jaw_plate": {"tooth_pitch_mm": [80, 90, 100]},
        },
    )
    assert len(cands) == 3
    for c in cands:
        assert c["swing_jaw"]["tooth_pitch_mm"] == c["fixed_jaw"]["tooth_pitch_mm"]


def test_enumerate_candidates_with_single_part_class():
    """Only swing_jaw swept; no cross-part constraint to apply → full sweep kept."""
    asm = load_assembly("PE_400x600", ROOT)
    cands = enumerate_candidates(
        assembly=asm,
        sweep_config={
            "swing_jaw_plate": {"tooth_pitch_mm": [85, 90, 95],
                                  "tooth_depth_mm": [20, 22]},
        },
    )
    assert len(cands) == 6  # 3 × 2


def test_feature_extraction_with_param_overrides_differs_from_baseline():
    asm = load_assembly("PE_400x600", ROOT)
    base = extract(asm, root=ROOT)
    overridden = extract(asm, root=ROOT,
                         param_overrides={"swing_jaw": {"tooth_pitch_mm": 999}})
    swing_idx = overridden.node_ids.index("swing_jaw")
    assert base.node_features[swing_idx] != overridden.node_features[swing_idx]
    # Other nodes should be unaffected
    toggle_idx = overridden.node_ids.index("toggle")
    assert base.node_features[toggle_idx] == overridden.node_features[toggle_idx]


def test_feature_extraction_overrides_dont_mutate_disk_state():
    """Calling extract with overrides must not write to instance YAMLs."""
    asm = load_assembly("PE_400x600", ROOT)
    on_disk_before = (ROOT / "templates/swing_jaw_plate/instances/PE_400x600.yaml").read_text()
    _ = extract(asm, root=ROOT,
                param_overrides={"swing_jaw": {"tooth_pitch_mm": 12345}})
    on_disk_after = (ROOT / "templates/swing_jaw_plate/instances/PE_400x600.yaml").read_text()
    assert on_disk_before == on_disk_after


# -------- torch-dependent end-to-end tests ---------------------------------

def _require_torch():
    return pytest.importorskip("torch", reason="torch not installed")


def test_screen_with_surrogate_end_to_end(tmp_path):
    """Train a tiny surrogate, save it, run the screening pipeline,
    verify the final Pareto front is well-formed."""
    _require_torch()
    from loop.assembly_features import load_schema
    from loop.synthetic_data import generate, TARGET_KEYS
    from loop.system_surrogate import SystemKPISurrogate
    from loop.surrogate_screening import screen_with_surrogate

    schema = load_schema(ROOT)
    samples = generate(n_samples=80, model="PE_400x600", root=ROOT, seed=7)
    sur = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS, hidden=32)
    sur.fit(samples, epochs=80, batch_size=8, val_split=0.2)
    ckpt = tmp_path / "sur.pt"
    sur.save(ckpt)

    result = screen_with_surrogate(
        model="PE_400x600",
        sweep_config={
            "swing_jaw_plate": {"tooth_pitch_mm": [85, 90, 95],
                                  "tooth_depth_mm": [20, 22, 24]},
            "fixed_jaw_plate": {"tooth_pitch_mm": [85, 90, 95]},
            "toggle_plate": {"thickness_mm": [24, 28]},
        },
        surrogate_path=ckpt,
        n_screened_to_keep=5,
        final_top_k=3,
        ore="basalt",
        use_mbd=False,
        root=ROOT,
    )
    assert result["model"] == "PE_400x600"
    assert result["n_candidates_generated"] > 0
    assert result["phase1_kept"] <= 5
    assert result["phase2_evaluated"] == result["phase1_kept"]
    assert 0 <= result["pareto_front_size"] <= 3
    for entry in result["pareto_front"]:
        assert entry["pareto_rank"] == 0
        assert entry["system_metrics"]["total_weight_kg"] > 0


def test_cli_with_surrogate_flag(tmp_path):
    """End-to-end CLI: train then screen."""
    _require_torch()
    sweep_yaml = tmp_path / "sweeps.yaml"
    sweep_yaml.write_text(
        "swing_jaw_plate:\n  tooth_pitch_mm: [85, 90, 95]\n"
        "fixed_jaw_plate:\n  tooth_pitch_mm: [85, 90, 95]\n"
    )
    ckpt = tmp_path / "sur.pt"
    # train
    proc = subprocess.run(
        [sys.executable, "bin/train_system_surrogate.py",
         "--n-samples", "60", "--epochs", "50",
         "--out", str(ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    # screen
    proc = subprocess.run(
        [sys.executable, "bin/run_assembly.py", "PE_400x600",
         "--sweep-config", str(sweep_yaml),
         "--surrogate", str(ckpt),
         "--n-screened-to-keep", "3", "--top-k", "2", "--no-mbd"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["model"] == "PE_400x600"
    assert "pareto_front" in out


def test_screening_required_sweep_config():
    """--surrogate without --sweep-config must fail cleanly."""
    proc = subprocess.run(
        [sys.executable, "bin/run_assembly.py", "PE_400x600",
         "--surrogate", "nonexistent.pt"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "--sweep-config" in proc.stderr
