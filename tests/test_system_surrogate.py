"""
System-level surrogate tests.

Splits into two layers:
  - Feature extraction + synthetic data shape (no torch required)
  - Model construction, training, save/load round-trip (torch required;
    skipped cleanly when torch is not installed)
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.assembly_features import (AssemblyFeatures,
                                     load_schema, extract_for_model)
from loop.synthetic_data import (generate, compute_kpis_from_features,
                                  TARGET_KEYS)


# -------- feature extraction (no torch) -------------------------------------

def test_schema_dimensions_are_consistent():
    schema = load_schema(ROOT)
    assert schema.node_dim == (len(schema.part_classes) + len(schema.materials)
                                + 1 + len(schema.criticality_vocab)
                                + schema.numeric_param_pad)
    assert schema.edge_dim == len(schema.edge_kinds)
    assert schema.global_dim == 5


def test_extract_assembly_features_shape():
    feats = extract_for_model("PE_400x600", root=ROOT)
    assert len(feats.node_features) == len(feats.node_ids)
    for row in feats.node_features:
        assert len(row) == feats.schema.node_dim
    for row in feats.edge_features:
        assert len(row) == feats.schema.edge_dim
    assert len(feats.global_features) == feats.schema.global_dim
    # edges are added in both directions
    assert len(feats.edge_features) == len(feats.edge_index)


def test_extract_is_deterministic():
    a = extract_for_model("PE_400x600", root=ROOT)
    b = extract_for_model("PE_400x600", root=ROOT)
    assert a.node_features == b.node_features
    assert a.edge_features == b.edge_features
    assert a.global_features == b.global_features


def test_synthetic_data_shape_and_targets():
    samples = generate(n_samples=10, model="PE_400x600", root=ROOT, seed=1)
    assert len(samples) == 10
    for feats, kpis in samples:
        assert isinstance(feats, AssemblyFeatures)
        for k in TARGET_KEYS:
            assert k in kpis
            assert kpis[k] > 0.0, f"{k} non-positive: {kpis[k]}"


def test_synthetic_kpis_respond_to_input_force():
    """Bearing life and wear life should both decrease as crushing force rises.
    This verifies the synthetic physics function has the expected monotonicity."""
    base = extract_for_model("PE_400x600", root=ROOT)
    low = compute_kpis_from_features(base, crushing_force_N=400_000.0)
    high = compute_kpis_from_features(base, crushing_force_N=1_200_000.0)
    assert low["bearing_L10_life_hours"] > high["bearing_L10_life_hours"]
    assert low["wear_part_life_hours"] > high["wear_part_life_hours"]


# -------- model + training (torch required) --------------------------------

def _require_torch():
    return pytest.importorskip("torch", reason="torch not installed")


def test_surrogate_can_be_constructed():
    _require_torch()
    from loop.system_surrogate import SystemKPISurrogate
    schema = load_schema(ROOT)
    sur = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS,
                              hidden=32)
    n_params = sum(p.numel() for p in sur._net.parameters())
    assert 1_000 < n_params < 1_000_000


def test_surrogate_overfits_tiny_dataset():
    """If the architecture can't overfit 20 deterministic samples, something
    is broken at the model level — train loss should drop substantially."""
    _require_torch()
    from loop.system_surrogate import SystemKPISurrogate
    schema = load_schema(ROOT)
    samples = generate(n_samples=20, model="PE_400x600", root=ROOT, seed=0)
    sur = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS,
                              hidden=64)
    out = sur.fit(samples, epochs=120, batch_size=8, val_split=0.0)
    assert out["final_train_loss"] < 0.5, (
        f"surrogate failed to overfit: final train loss {out['final_train_loss']}"
    )


def test_surrogate_save_load_roundtrip():
    _require_torch()
    from loop.system_surrogate import SystemKPISurrogate
    schema = load_schema(ROOT)
    samples = generate(n_samples=30, model="PE_400x600", root=ROOT, seed=2)
    sur = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS, hidden=32)
    sur.fit(samples, epochs=50, batch_size=8, val_split=0.2)

    feats = samples[0][0]
    pred_before = sur.predict(feats)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sys_v1.pt"
        sur.save(path)
        assert path.exists()
        loaded = SystemKPISurrogate.load(path)
        pred_after = loaded.predict(feats)

    for k in TARGET_KEYS:
        assert pred_before[k] == pytest.approx(pred_after[k], rel=1e-5)


def test_surrogate_screens_candidates_for_top_k():
    """Practical use: rank a batch of candidates by a chosen KPI."""
    _require_torch()
    from loop.system_surrogate import SystemKPISurrogate
    schema = load_schema(ROOT)
    samples = generate(n_samples=80, model="PE_400x600", root=ROOT, seed=3)
    sur = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS, hidden=64)
    sur.fit(samples, epochs=100, batch_size=8, val_split=0.2)

    candidates = [s[0] for s in samples[:20]]
    preds = sur.predict_batch(candidates)
    tphs = [p["tph_at_design_css"] for p in preds]
    assert max(tphs) - min(tphs) > 0.5, "all predictions collapsed to one value"
