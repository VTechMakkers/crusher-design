"""
Feature extraction for the assembly-level surrogate.

Converts a CrusherAssembly + per-part instance YAMLs + the mechanism
block into numerical tensors the surrogate consumes:

  node features  — per-part: one-hot part class, one-hot material,
                   wear flag, one-hot criticality, padded numerical params
  edge features  — per-connection: one-hot connection kind
  global features — mechanism: eccentric throw, pitman length, toggle
                   length, rpm, motor power

Feature dimensions are derived from the on-disk catalog so the schema
stays consistent as new part classes / materials are added. The schema is
saved alongside any trained model so inference reproduces training-time
layout exactly.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly.crusher_assembly import CrusherAssembly, load as load_assembly


NUMERIC_PARAM_PAD = 20
EDGE_KINDS = ("constraint", "kinematic", "structural")
CRITICALITY_VOCAB = ("safety_critical", "sacrificial", "structural")


@dataclass(frozen=True)
class FeatureSchema:
    part_classes: tuple[str, ...]
    materials: tuple[str, ...]
    numeric_param_pad: int
    edge_kinds: tuple[str, ...]
    criticality_vocab: tuple[str, ...]

    @property
    def node_dim(self) -> int:
        return (len(self.part_classes)
                + len(self.materials)
                + 1                              # wear-exposed flag
                + len(self.criticality_vocab)
                + self.numeric_param_pad)

    @property
    def edge_dim(self) -> int:
        return len(self.edge_kinds)

    @property
    def global_dim(self) -> int:
        return 5


@dataclass(frozen=True)
class AssemblyFeatures:
    schema: FeatureSchema
    node_features: list[list[float]]
    node_ids: list[str]
    edge_index: list[tuple[int, int]]
    edge_features: list[list[float]]
    global_features: list[float]


def load_schema(root: Path = ROOT) -> FeatureSchema:
    parts = yaml.safe_load((root / "catalog/parts.yaml").read_text())["parts"]
    materials: set[str] = set()
    for info in parts.values():
        m = info.get("typical_material")
        if m:
            materials.add(m)
    mats_yaml = root / "knowledge/materials.yaml"
    if mats_yaml.exists():
        materials.update((yaml.safe_load(mats_yaml.read_text()) or {}).keys())
    return FeatureSchema(
        part_classes=tuple(sorted(parts.keys())),
        materials=tuple(sorted(materials)),
        numeric_param_pad=NUMERIC_PARAM_PAD,
        edge_kinds=EDGE_KINDS,
        criticality_vocab=CRITICALITY_VOCAB,
    )


def _one_hot(value: Any, vocab: tuple[str, ...]) -> list[float]:
    return [1.0 if v == value else 0.0 for v in vocab]


def _param_vector(params: dict[str, Any] | None, pad: int) -> list[float]:
    if not params:
        return [0.0] * pad
    vals: list[float] = []
    for v in params.values():
        if isinstance(v, bool):
            vals.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            vals.append(0.0)
    if len(vals) >= pad:
        return vals[:pad]
    return vals + [0.0] * (pad - len(vals))


def extract(assembly: CrusherAssembly,
            schema: FeatureSchema | None = None,
            root: Path = ROOT,
            param_overrides: dict[str, dict[str, Any]] | None = None
            ) -> AssemblyFeatures:
    """Build the feature representation for an assembly.

    Missing per-part instances (the `_pending` placeholders in the assembly
    graph) contribute their default material from `catalog/parts.yaml` and
    zero numerical params — the surrogate sees them as "present but
    unspecified", which is the correct inductive signal.

    `param_overrides`: per-node parameter overrides, applied on top of the
    on-disk baseline. Used by surrogate screening to evaluate hypothetical
    variants without modifying the instance YAML files.
    """
    schema = schema or load_schema(root)
    parts_catalog = yaml.safe_load((root / "catalog/parts.yaml").read_text())["parts"]

    node_features: list[list[float]] = []
    node_ids: list[str] = []
    for node_id, part_ref in assembly.parts.items():
        info = parts_catalog.get(part_ref.part_class, {})
        try:
            instance = assembly.load_part_instance(node_id)
        except FileNotFoundError:
            instance = None

        material = (instance or {}).get("material") or info.get("typical_material", "")
        params = dict((instance or {}).get("params") or {})
        if param_overrides and node_id in param_overrides:
            params.update(param_overrides[node_id])
        feats: list[float] = []
        feats.extend(_one_hot(part_ref.part_class, schema.part_classes))
        feats.extend(_one_hot(material, schema.materials))
        feats.append(1.0 if info.get("wear_exposed") else 0.0)
        feats.extend(_one_hot(info.get("criticality", ""), schema.criticality_vocab))
        feats.extend(_param_vector(params, schema.numeric_param_pad))
        if len(feats) != schema.node_dim:
            raise RuntimeError(
                f"node feature length {len(feats)} != schema node_dim {schema.node_dim}"
            )
        node_features.append(feats)
        node_ids.append(node_id)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for conn in assembly.connections:
        if conn.a not in id_to_idx or conn.b not in id_to_idx:
            continue
        src, dst = id_to_idx[conn.a], id_to_idx[conn.b]
        kind_oh = _one_hot(conn.kind, schema.edge_kinds)
        edge_index.append((src, dst))
        edge_features.append(kind_oh)
        edge_index.append((dst, src))
        edge_features.append(kind_oh)

    asm_yaml = yaml.safe_load((root / f"assembly/{assembly.model}.yaml").read_text())
    mech = asm_yaml.get("mechanism") or {}
    global_features = [
        float(mech.get("eccentric_throw_mm", 14.0)),
        float(mech.get("pitman_length_mm", 600.0)),
        float(mech.get("toggle_length_mm", 300.0)),
        float(mech.get("rpm", 280.0)),
        float(mech.get("motor_kW", 30.0)),
    ]

    return AssemblyFeatures(
        schema=schema,
        node_features=node_features,
        node_ids=node_ids,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
    )


def extract_for_model(model: str, root: Path = ROOT,
                      param_overrides: dict[str, dict[str, Any]] | None = None
                      ) -> AssemblyFeatures:
    return extract(load_assembly(model, root), root=root,
                   param_overrides=param_overrides)
