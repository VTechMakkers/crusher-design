"""
Surrogate-screened whole-machine search.

Two-stage pipeline:

  Phase 1 (cheap, ~1 ms per candidate)
    - enumerate candidate assemblies from sweep_config
    - filter by cross-part constraints early (surrogate cannot reason about them)
    - extract AssemblyFeatures with per-candidate parameter overrides
    - predict 6-D system KPIs via the trained system_surrogate
    - rank predicted KPIs by Pareto + crowding-distance diversity
    - keep top-N predicted

  Phase 2 (expensive, ~minutes per candidate in production)
    - for each of the top-N: run the real per-part design_loop.evaluate
      and system_fitness aggregation
    - Pareto-rank the real-evaluated subset
    - return final top-K

When real solvers (CalculiX, LIGGGHTS) are wired in, Phase 2 becomes the
bottleneck. Surrogate screening lets the search visit 10^3–10^4 candidates
in seconds, then spend the expensive solver budget only on the ones the
surrogate ranks well — typically a 50–200x throughput multiplier vs full
brute-force sweeps.
"""
from __future__ import annotations
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assembly.crusher_assembly import CrusherAssembly, load as load_assembly
from loop import assembly_loop
from loop.assembly_features import extract as extract_features, load_schema
from loop.pareto import Candidate, rank_all, select_diverse
from loop.synthetic_data import TARGET_KEYS


@dataclass
class ScreeningResult:
    candidate_params: dict[str, dict[str, Any]]   # node_id -> params
    predicted_kpis: dict[str, float]


def enumerate_candidates(*, assembly: CrusherAssembly,
                          sweep_config: dict[str, dict[str, list[Any]]]
                          ) -> list[dict[str, dict[str, Any]]]:
    """Enumerate the constraint-satisfying combinations of swept part params.

    Returns a list of {node_id -> params}. Only nodes whose part_class
    appears in `sweep_config` are included; unsweept nodes retain their
    baseline params at evaluation time.
    """
    # Map part_class -> list of node_ids of that class in this assembly
    class_to_nodes: dict[str, list[str]] = {}
    for node_id, part_ref in assembly.parts.items():
        class_to_nodes.setdefault(part_ref.part_class, []).append(node_id)

    # Build per-node sweeps. A node inherits its part_class's sweep.
    per_node_sweeps: dict[str, list[dict[str, Any]]] = {}
    for part_class, sweep_params in sweep_config.items():
        if part_class not in class_to_nodes:
            continue
        keys = list(sweep_params.keys())
        combos = [dict(zip(keys, vals))
                   for vals in itertools.product(*[sweep_params[k] for k in keys])]
        for node_id in class_to_nodes[part_class]:
            per_node_sweeps[node_id] = combos

    node_ids = list(per_node_sweeps.keys())
    out: list[dict[str, dict[str, Any]]] = []
    for combo in itertools.product(*[per_node_sweeps[n] for n in node_ids]):
        cand = {nid: params for nid, params in zip(node_ids, combo)}
        if assembly_loop.constraints_satisfied(assembly, cand):
            out.append(cand)
    return out


def predict_for_candidates(*, assembly: CrusherAssembly,
                            candidates: list[dict[str, dict[str, Any]]],
                            surrogate: Any,
                            root: Path = ROOT) -> list[ScreeningResult]:
    """Run the surrogate on every candidate. Returns list of (candidate, KPIs)."""
    schema = load_schema(root)
    out: list[ScreeningResult] = []
    for cand in candidates:
        feats = extract_features(assembly, schema=schema, root=root,
                                  param_overrides=cand)
        kpis = surrogate.predict(feats)
        out.append(ScreeningResult(candidate_params=cand, predicted_kpis=kpis))
    return out


def _pareto_signs(model: str, root: Path) -> dict[str, str]:
    asm_yaml = yaml.safe_load((root / f"assembly/{model}.yaml").read_text())
    return {obj["key"]: obj["sign"]
             for obj in asm_yaml.get("pareto_objectives", [])}


def select_top_n_by_predicted_pareto(*, model: str,
                                      results: list[ScreeningResult],
                                      n: int,
                                      root: Path = ROOT
                                      ) -> list[ScreeningResult]:
    """Pareto-rank surrogate predictions; pick diverse top-N from rank-0 set,
    fill from rank-1 onwards if rank-0 is smaller than n."""
    if not results:
        return []
    signs = _pareto_signs(model, root)
    candidates = []
    for i, r in enumerate(results):
        vec = []
        for k in TARGET_KEYS:
            v = r.predicted_kpis[k]
            vec.append(v if signs.get(k) == "maximize" else -v)
        candidates.append(Candidate(id=str(i), objectives=vec, payload=r))
    ranks = rank_all(candidates)
    front = [c for c in candidates if ranks[c.id] == 0]
    chosen = select_diverse(front, n) if len(front) > n else front
    if len(chosen) < n:
        extras = sorted([c for c in candidates if ranks[c.id] > 0],
                        key=lambda c: ranks[c.id])[:n - len(chosen)]
        chosen = list(chosen) + list(extras)
    return [c.payload for c in chosen]


def screen_with_surrogate(*,
                           model: str,
                           sweep_config: dict[str, dict[str, list[Any]]],
                           surrogate_path: Path,
                           n_screened_to_keep: int = 20,
                           final_top_k: int = 5,
                           ore: str = "basalt",
                           peak_crushing_force_N: float = 850_000.0,
                           use_mbd: bool = True,
                           root: Path = ROOT) -> dict[str, Any]:
    """End-to-end screening pipeline."""
    from loop.system_surrogate import SystemKPISurrogate
    surrogate = SystemKPISurrogate.load(surrogate_path)

    assembly = load_assembly(model, root)
    candidates = enumerate_candidates(assembly=assembly, sweep_config=sweep_config)
    if not candidates:
        return {"model": model, "n_candidates": 0, "pareto_front": [],
                "phase1_kept": 0, "phase2_evaluated": 0}

    phase1 = predict_for_candidates(assembly=assembly, candidates=candidates,
                                      surrogate=surrogate, root=root)
    kept = select_top_n_by_predicted_pareto(
        model=model, results=phase1,
        n=min(n_screened_to_keep, len(phase1)), root=root,
    )

    # Phase 2: run the real (per-part evaluate + aggregate) path on kept candidates
    real_variants: list[assembly_loop.AssemblyVariant] = []
    ore_data = yaml.safe_load((root / "knowledge/ores.yaml").read_text()) \
                  .get("ores", {}).get(ore, {})
    for screen in kept:
        av = _evaluate_candidate(
            assembly=assembly, candidate=screen.candidate_params,
            ore_data=ore_data, use_mbd=use_mbd,
            peak_crushing_force_N=peak_crushing_force_N, root=root,
        )
        real_variants.append(av)

    # Final Pareto rank over real-evaluated subset
    if real_variants:
        signs = _pareto_signs(model, root)
        cands = []
        for i, av in enumerate(real_variants):
            cands.append(Candidate(id=str(i),
                                    objectives=av.system_metrics.to_objective_vector(signs),
                                    payload=av))
        ranks = rank_all(cands)
        for c in cands:
            c.payload.pareto_rank = ranks[c.id]
        front = [c for c in cands if ranks[c.id] == 0]
        diverse = (select_diverse(front, final_top_k)
                    if len(front) > final_top_k else front)
        final_front = [c.payload for c in diverse]
    else:
        final_front = []

    return {
        "model": model,
        "n_candidates_generated": len(candidates),
        "phase1_kept": len(kept),
        "phase2_evaluated": len(real_variants),
        "pareto_front_size": len(final_front),
        "pareto_front": [assembly_loop._serialize(av) for av in final_front],
        "phase1_predictions": [
            {"candidate": s.candidate_params, "predicted": s.predicted_kpis}
            for s in kept
        ],
    }


def _evaluate_candidate(*, assembly: CrusherAssembly,
                         candidate: dict[str, dict[str, Any]],
                         ore_data: dict[str, Any],
                         use_mbd: bool, peak_crushing_force_N: float,
                         root: Path) -> assembly_loop.AssemblyVariant:
    """Run per-part evaluate + aggregate for one fully-specified candidate.
    For nodes not in `candidate`, baseline instance params are used; for
    nodes without an instance YAML, placeholder zero-metric stubs are used."""
    from loop import design_loop
    from loop.mbd_to_fea import derive_load_case_for_model
    from assembly.crusher_assembly import CrusherAssembly as _CA

    catalog = assembly_loop._catalog_parts(root)
    part_variants: dict[str, assembly_loop.PartVariant] = {}
    for node_id, part_ref in assembly.parts.items():
        info = catalog.get(part_ref.part_class, {})
        if not info.get("has_geometry"):
            part_variants[node_id] = assembly_loop.PartVariant(
                node_id=node_id, part_class=part_ref.part_class,
                model=assembly.model, params={},
                material=info.get("typical_material", ""),
                metrics={"mass_kg": 0.0, "max_von_mises_MPa": 0.0},
                fitness={"composite": 0.0}, has_geometry=False,
            )
            continue
        try:
            instance = assembly.load_part_instance(node_id)
        except FileNotFoundError:
            continue
        params = dict(instance["params"])
        if node_id in candidate:
            params.update(candidate[node_id])
        mbd_case = None
        if use_mbd:
            try:
                mbd_case = derive_load_case_for_model(
                    part_class=part_ref.part_class, model=assembly.model,
                    peak_crushing_force_N=peak_crushing_force_N, root=root,
                )
            except ValueError:
                pass
        rec = design_loop.evaluate(
            part=part_ref.part_class, model=assembly.model, params=params,
            material=instance["material"], runner=None,
            override_load_case=mbd_case,
        )
        part_variants[node_id] = assembly_loop.PartVariant(
            node_id=node_id, part_class=part_ref.part_class, model=assembly.model,
            params=params, material=instance["material"],
            metrics=rec["metrics"], fitness=rec["fitness"], has_geometry=True,
        )

    sys_metrics = assembly_loop.aggregate_to_system(
        part_variants=part_variants, assembly=assembly,
        ore_data=ore_data, root=root,
    )
    return assembly_loop.AssemblyVariant(
        model=assembly.model, part_variants=part_variants,
        system_metrics=sys_metrics,
    )
