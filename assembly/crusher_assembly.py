"""
Assembly-level representation of a whole crusher.

A crusher is a graph of parts connected by:
  - STRUCTURAL connections (load path: rock → swing jaw → pitman → toggle → frame)
  - KINEMATIC connections (motion path: motor → eccentric → pitman → swing jaw)
  - CONSTRAINT relations (geometry consistency: swing jaw pitch == fixed jaw pitch)

This module loads an assembly YAML, exposes:
  - graph traversal (which parts share loads, which share motion)
  - constraint checking (verify part instances are mutually consistent)
  - aggregate KPIs hooks (mass, cost, power requirement at assembly level)

NetworkX is the only optional dependency; we provide an in-house fallback
for the small graphs we need (15–25 nodes).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
CATALOG = ROOT / "catalog"


@dataclass
class PartRef:
    """A part instance referenced from an assembly."""
    part_class: str            # e.g. "swing_jaw_plate"
    instance: str              # e.g. "PE_400x600" (instance YAML stem)
    role: str                  # human-readable role in assembly


@dataclass
class Connection:
    """An edge between two parts."""
    a: str                     # node id (assembly-local name, e.g. "swing_jaw")
    b: str
    kind: str                  # "structural" | "kinematic" | "constraint"
    detail: str = ""


@dataclass
class CrusherAssembly:
    model: str
    family: str
    parts: dict[str, PartRef]                # node_id -> PartRef
    connections: list[Connection]
    system_targets: dict[str, Any]           # TPH, motor_kW, CSS_range, etc.
    constraints: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "CrusherAssembly":
        data = yaml.safe_load(path.read_text())
        parts = {nid: PartRef(**v) for nid, v in data["parts"].items()}
        conns = [Connection(**c) for c in data.get("connections", [])]
        return cls(model=data["model"], family=data["family"],
                   parts=parts, connections=conns,
                   system_targets=data.get("system_targets", {}),
                   constraints=data.get("constraints", []))

    def neighbors(self, node: str, kind: str | None = None) -> list[str]:
        out: list[str] = []
        for c in self.connections:
            if kind is not None and c.kind != kind:
                continue
            if c.a == node:
                out.append(c.b)
            elif c.b == node:
                out.append(c.a)
        return out

    def load_path(self, source: str = "swing_jaw", sink: str = "main_frame") -> list[list[str]]:
        """All structural paths from source node to sink. BFS, return paths."""
        seen: set[tuple[str, ...]] = set()
        paths: list[list[str]] = []
        stack: list[list[str]] = [[source]]
        while stack:
            path = stack.pop()
            node = path[-1]
            if node == sink:
                t = tuple(path)
                if t not in seen:
                    seen.add(t)
                    paths.append(path)
                continue
            for nbr in self.neighbors(node, kind="structural"):
                if nbr not in path:
                    stack.append(path + [nbr])
        return paths

    def load_part_instance(self, node_id: str) -> dict[str, Any]:
        ref = self.parts[node_id]
        inst_path = TEMPLATES / ref.part_class / "instances" / f"{ref.instance}.yaml"
        return yaml.safe_load(inst_path.read_text())

    def check_consistency(self) -> dict[str, Any]:
        """Run all declared cross-part constraints. Returns issues + skipped.
        Constraints referencing parts whose instance YAML doesn't exist yet
        (e.g. `_pending` placeholders) are reported as skipped, not failures."""
        issues: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        def try_load(node_id: str):
            try:
                return self.load_part_instance(node_id), None
            except FileNotFoundError as e:
                return None, str(e)

        for con in self.constraints:
            kind = con["kind"]
            if kind == "params_equal":
                a_inst, a_err = try_load(con["a"])
                b_inst, b_err = try_load(con["b"])
                if a_err or b_err:
                    skipped.append({"kind": kind, "param": con["param"],
                                    "reason": "pending instance",
                                    "a_err": a_err, "b_err": b_err})
                    continue
                a_val = a_inst["params"].get(con["param"])
                b_val = b_inst["params"].get(con["param"])
                if a_val != b_val:
                    issues.append({"kind": kind, "param": con["param"],
                                   "a": con["a"], "a_val": a_val,
                                   "b": con["b"], "b_val": b_val})
            elif kind == "params_match_set":
                inst_vals: dict[str, Any] = {}
                missing: list[str] = []
                for nid in con["nodes"]:
                    inst, err = try_load(nid)
                    if err:
                        missing.append(nid)
                    else:
                        inst_vals[nid] = inst["params"].get(con["param"])
                if missing:
                    skipped.append({"kind": kind, "param": con["param"],
                                    "reason": "pending instances",
                                    "missing": missing})
                    continue
                if len(set(inst_vals.values())) > 1:
                    issues.append({"kind": kind, "param": con["param"],
                                   "values": inst_vals})
        return {"passes": not issues, "issues": issues, "skipped": skipped}

    def part_classes(self) -> set[str]:
        return {p.part_class for p in self.parts.values()}

    def material_list(self) -> dict[str, list[str]]:
        """Map material -> list of node_ids that use it."""
        out: dict[str, list[str]] = {}
        for nid, ref in self.parts.items():
            try:
                inst = self.load_part_instance(nid)
            except FileNotFoundError:
                continue
            mat = inst.get("material", "unknown")
            out.setdefault(mat, []).append(nid)
        return out


def load(model: str, root: Path = ROOT) -> CrusherAssembly:
    """Load an assembly definition by model name."""
    path = root / "assembly" / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no assembly for {model} at {path}")
    return CrusherAssembly.from_yaml(path)


def list_assemblies(root: Path = ROOT) -> list[str]:
    return sorted(p.stem for p in (root / "assembly").glob("*.yaml")
                  if p.stem != "schema")
