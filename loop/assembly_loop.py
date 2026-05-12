"""
Whole-machine optimisation orchestrator.

Runs the per-part design loop across every part in an assembly that has
a parametric template, applies cross-part constraints (paired-jaw pitch,
shared bearing bore, etc.), aggregates per-part metrics into 6-D system
KPIs via system_fitness, and Pareto-ranks complete crushers.

Output: a ranked Pareto front of whole crushers. Each entry is a dict
{node_id -> part_variant} plus the system-level KPI vector. Customer /
engineer picks one point on the front based on which trade-off they want.

Combinatorial control:
  - per_part_top_k caps the variants kept after the per-part sweep
  - cross-part constraints filter combinations that violate paired-part
    parameter equality (e.g. swing/fixed jaw pitch must match)
  - the remaining valid combinations are aggregated and Pareto-ranked
  - select_diverse via crowding distance picks the final top-K
"""
from __future__ import annotations
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assembly.crusher_assembly import CrusherAssembly, load as load_assembly
from loop import design_loop, system_fitness
from loop.pareto import Candidate, rank_all, select_diverse, select_pareto_top_k
from loop.mbd_to_fea import derive_load_case_for_model


@dataclass
class PartVariant:
    """One part variant with per-part metrics."""
    node_id: str
    part_class: str
    model: str
    params: dict[str, Any]
    material: str
    metrics: dict[str, float]
    fitness: dict[str, float]
    has_geometry: bool


@dataclass
class AssemblyVariant:
    """One whole-crusher candidate."""
    model: str
    part_variants: dict[str, PartVariant]
    system_metrics: system_fitness.SystemMetrics
    pareto_rank: int | None = None


# ---------------------------------------------------------------------------
# Catalog + ore lookup helpers

def _catalog_parts(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "catalog/parts.yaml").read_text())["parts"]


def _materials(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "knowledge/materials.yaml").read_text()) or {}


def _ore_data(ore: str, root: Path) -> dict[str, Any]:
    data = yaml.safe_load((root / "knowledge/ores.yaml").read_text())
    return data.get("ores", {}).get(ore, {})


def _pareto_objective_signs(model: str, root: Path) -> dict[str, str]:
    asm_yaml = yaml.safe_load((root / f"assembly/{model}.yaml").read_text())
    return {obj["key"]: obj["sign"]
             for obj in asm_yaml.get("pareto_objectives", [])}


# ---------------------------------------------------------------------------
# Per-part sweep

def sweep_part(*, assembly: CrusherAssembly, node_id: str,
                sweep_params: dict[str, list[Any]] | None,
                use_mbd: bool, peak_crushing_force_N: float,
                top_k_per_part: int,
                root: Path = ROOT) -> list[PartVariant]:
    """Generate per-part variants, evaluate via design_loop, return top-K by composite."""
    part_ref = assembly.parts[node_id]
    part_class = part_ref.part_class
    catalog = _catalog_parts(root)
    info = catalog.get(part_class, {})

    if not info.get("has_geometry"):
        return [PartVariant(
            node_id=node_id, part_class=part_class, model=assembly.model,
            params={}, material=info.get("typical_material", ""),
            metrics={"mass_kg": 0.0, "max_von_mises_MPa": 0.0},
            fitness={"composite": 0.0},
            has_geometry=False,
        )]

    try:
        instance = assembly.load_part_instance(node_id)
    except FileNotFoundError:
        return []

    base_params = dict(instance["params"])
    if sweep_params:
        keys = list(sweep_params.keys())
        candidates = []
        for combo in itertools.product(*[sweep_params[k] for k in keys]):
            p = dict(base_params)
            for k, v in zip(keys, combo):
                p[k] = v
            candidates.append(p)
    else:
        candidates = [base_params]

    mbd_case = None
    if use_mbd:
        try:
            mbd_case = derive_load_case_for_model(
                part_class=part_class, model=assembly.model,
                peak_crushing_force_N=peak_crushing_force_N, root=root,
            )
        except ValueError:
            mbd_case = None   # MBD bridge has no mapping for this part class

    results: list[PartVariant] = []
    for params in candidates:
        rec = design_loop.evaluate(
            part=part_class, model=assembly.model, params=params,
            material=instance["material"], runner=None,
            override_load_case=mbd_case,
        )
        results.append(PartVariant(
            node_id=node_id, part_class=part_class, model=assembly.model,
            params=params, material=instance["material"],
            metrics=rec["metrics"], fitness=rec["fitness"],
            has_geometry=True,
        ))

    results.sort(key=lambda v: -v.fitness["composite"])
    return results[:top_k_per_part]


# ---------------------------------------------------------------------------
# Cross-part constraint enforcement

def constraints_satisfied(assembly: CrusherAssembly,
                            params_by_node: dict[str, dict[str, Any]]) -> bool:
    """Evaluate cross-part constraints against in-memory per-node parameters.

    Constraints referencing nodes outside `params_by_node` (e.g. swept-only
    subset) are skipped, not failed — same convention as the static
    consistency checker.
    """
    for con in assembly.constraints:
        kind = con["kind"]
        if kind == "params_equal":
            a, b = con["a"], con["b"]
            if a not in params_by_node or b not in params_by_node:
                continue
            pa = params_by_node[a].get(con["param"])
            pb = params_by_node[b].get(con["param"])
            if pa is None or pb is None:
                continue
            if pa != pb:
                return False
        elif kind == "params_match_set":
            param = con["param"]
            vals = [params_by_node[n].get(param)
                     for n in con["nodes"] if n in params_by_node]
            vals = [v for v in vals if v is not None]
            if len(set(vals)) > 1:
                return False
    return True


# ---------------------------------------------------------------------------
# System-level aggregation

def _aggregable_part_results(part_variants: dict[str, PartVariant],
                              root: Path) -> dict[str, dict[str, Any]]:
    catalog = _catalog_parts(root)
    materials = _materials(root)
    out: dict[str, dict[str, Any]] = {}
    for node_id, pv in part_variants.items():
        info = catalog.get(pv.part_class, {})
        mat = materials.get(pv.material, {})
        mass = pv.metrics.get("mass_kg", 0.0)
        cost_per_kg = mat.get("cost_INR_per_kg", 100.0)
        out[node_id] = {
            "mass_kg": mass,
            "max_stress_MPa": pv.metrics.get("max_von_mises_MPa",
                                              pv.metrics.get("max_stress_MPa", 0.0)),
            "material": pv.material,
            "wear_exposed": info.get("wear_exposed", False),
            "archard_k_relative": mat.get("archard_k_relative", 1.0),
            "cost_INR": mass * cost_per_kg,
        }
    return out


def aggregate_to_system(*, part_variants: dict[str, PartVariant],
                          assembly: CrusherAssembly,
                          ore_data: dict[str, Any],
                          root: Path = ROOT) -> system_fitness.SystemMetrics:
    asm_yaml = yaml.safe_load((root / f"assembly/{assembly.model}.yaml").read_text())
    targets = asm_yaml.get("system_targets", {})
    part_results = _aggregable_part_results(part_variants, root)
    return system_fitness.aggregate(
        part_results=part_results,
        assembly_targets=targets,
        ore=ore_data,
    )


# ---------------------------------------------------------------------------
# Top-level driver

def run_assembly(*, model: str,
                  ore: str = "basalt",
                  peak_crushing_force_N: float = 850_000.0,
                  sweep_config: dict[str, dict[str, list[Any]]] | None = None,
                  per_part_top_k: int = 3,
                  top_k: int = 5,
                  use_mbd: bool = True,
                  algorithm: str = "default",
                  root: Path = ROOT) -> dict[str, Any]:
    """Run the whole-machine optimisation pipeline.

    sweep_config: {part_class -> {param_name -> [values]}}. Parts without an
    entry use baseline params only. Parts present in the assembly graph but
    not in sweep_config still appear in the resulting assembly (with their
    baseline params).
    """
    assembly = load_assembly(model, root)
    ore_data = _ore_data(ore, root)
    catalog = _catalog_parts(root)

    per_part: dict[str, list[PartVariant]] = {}
    for node_id in assembly.parts:
        part_ref = assembly.parts[node_id]
        sweep_params = (sweep_config or {}).get(part_ref.part_class)
        variants = sweep_part(
            assembly=assembly, node_id=node_id,
            sweep_params=sweep_params,
            use_mbd=use_mbd, peak_crushing_force_N=peak_crushing_force_N,
            top_k_per_part=per_part_top_k, root=root,
        )
        if variants:
            per_part[node_id] = variants

    node_ids = list(per_part.keys())
    variants_per_node = [per_part[n] for n in node_ids]

    assembly_variants: list[AssemblyVariant] = []
    n_combinations = 1
    for vs in variants_per_node:
        n_combinations *= len(vs)

    for combo in itertools.product(*variants_per_node):
        part_variants = {nid: pv for nid, pv in zip(node_ids, combo)}
        params_by_node = {nid: pv.params for nid, pv in part_variants.items()
                           if pv.has_geometry}
        if not constraints_satisfied(assembly, params_by_node):
            continue
        sys_metrics = aggregate_to_system(
            part_variants=part_variants, assembly=assembly,
            ore_data=ore_data, root=root,
        )
        assembly_variants.append(AssemblyVariant(
            model=model, part_variants=part_variants,
            system_metrics=sys_metrics,
        ))

    pareto_front: list[AssemblyVariant] = []
    if assembly_variants:
        signs = _pareto_objective_signs(model, root)
        candidates = []
        for i, av in enumerate(assembly_variants):
            vec = av.system_metrics.to_objective_vector(signs)
            candidates.append(Candidate(id=str(i), objectives=vec, payload=av))
        # Tag pareto_rank on every variant using the built-in ranker
        # (so all_valid_variants gets ranks even when pymoo selection is used)
        ranks = rank_all(candidates)
        for c in candidates:
            c.payload.pareto_rank = ranks[c.id]
        diverse = select_pareto_top_k(candidates, top_k, algorithm=algorithm)
        pareto_front = [c.payload for c in diverse]

    return {
        "model": model,
        "ore": ore,
        "peak_crushing_force_N": peak_crushing_force_N,
        "n_parts_considered": len(per_part),
        "n_combinations": n_combinations,
        "n_valid_assemblies": len(assembly_variants),
        "pareto_front_size": len(pareto_front),
        "pareto_front": [_serialize(av) for av in pareto_front],
        "all_valid_variants": [_serialize(av) for av in assembly_variants],
    }


def _serialize(av: AssemblyVariant) -> dict[str, Any]:
    return {
        "model": av.model,
        "pareto_rank": av.pareto_rank,
        "system_metrics": av.system_metrics.as_dict(),
        "part_variants": {
            node_id: {
                "part_class": pv.part_class,
                "material": pv.material,
                "params": pv.params,
                "fitness_composite": pv.fitness.get("composite", 0.0),
                "max_stress_MPa": pv.metrics.get("max_von_mises_MPa",
                                              pv.metrics.get("max_stress_MPa", 0.0)),
            } for node_id, pv in av.part_variants.items()
        },
    }
