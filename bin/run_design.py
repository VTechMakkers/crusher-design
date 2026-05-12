#!/usr/bin/env python3
"""
End-to-end design driver.

Usage:
  python bin/run_design.py <part> <model> [--ore basalt] [--css 80] [--tph 65]
                                          [--sweep thickness_mm=24,28,32]
                                          [--top-k 3] [--dry-run]

Runs the full design loop:
  1. Load (part, model) instance
  2. Generate variants (deterministic sweep OR LLM-proposed)
  3. For each: geometry export, FEA (or skip if --dry-run), DEM, fitness
  4. DFM gate (reject un-makeable)
  5. Rank, log to history, return top-K

Without GPU + LIGGGHTS installed, defaults to --dry-run mode that exercises
the orchestration plumbing with placeholder solver outputs.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop import design_loop
from loop.dem_fitness import dem_score, combined_score


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_sweep_arg(s: str) -> dict[str, list[Any]]:
    """Parse --sweep "thickness_mm=24,28,32;width_mm=160,200" into dict."""
    out: dict[str, list[Any]] = {}
    if not s:
        return out
    for clause in s.split(";"):
        key, values = clause.split("=", 1)
        items = []
        for v in values.split(","):
            v = v.strip()
            try:
                items.append(int(v) if v.isdigit() else float(v))
            except ValueError:
                items.append(v)
        out[key.strip()] = items
    return out


def dfm_gate(part: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run DFM check by importing the dfm server module directly (no MCP needed)."""
    dfm = load_module(ROOT / "mcp-servers/dfm/server.py", "dfm_local")
    if part == "toggle_plate":
        return dfm.check_toggle_plate(params)
    if part in ("swing_jaw_plate", "fixed_jaw_plate"):
        return dfm.check_jaw_plate(params)
    return {"passes": True, "errors": [], "warnings": [], "info": [], "total_issues": 0}


def run(part: str, model: str, ore: str, css_mm: float, target_tph: float,
        sweep: dict[str, list[Any]], top_k: int, dry_run: bool,
        use_mbd: bool = False,
        peak_crushing_force_N: float = 850_000.0,
        pareto: bool = False) -> dict[str, Any]:
    inst = design_loop.load_instance(part, model)
    base = dict(inst["params"])

    mbd_case: dict[str, Any] | None = None
    if use_mbd:
        from loop.mbd_to_fea import derive_load_case_for_model
        mbd_case = derive_load_case_for_model(
            part_class=part, model=model,
            peak_crushing_force_N=peak_crushing_force_N,
            root=ROOT,
        )

    # Build candidate grid
    if sweep:
        import itertools
        keys = list(sweep.keys())
        candidates = []
        for combo in itertools.product(*[sweep[k] for k in keys]):
            p = dict(base)
            for k, v in zip(keys, combo):
                p[k] = v
            candidates.append(p)
    else:
        candidates = [base]

    # DFM gate
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for p in candidates:
        gate = dfm_gate(part, p)
        if gate["passes"]:
            passed.append(p)
        else:
            rejected.append({"params": p, "errors": gate["errors"]})

    # Evaluate (dry-run uses plumbing-only path in loop.design_loop)
    runner = None  # dry-run; integrate MCP clients here when ready
    results = []
    for p in passed:
        rec = design_loop.evaluate(part=part, model=model, params=p,
                                   runner=runner, load_case_name=None,
                                   override_load_case=mbd_case)
        # In dry-run, also fake some DEM metrics to exercise dem_fitness
        if dry_run:
            dem = dem_score(
                throughput_kg_per_s=target_tph * 1000.0 / 3600.0 * 0.95,
                target_tph=target_tph,
                p80_mm=css_mm * 0.55,
                p80_target_mm=css_mm * 0.55,
                wear_uniformity_index=0.72,
                energy_kWh_per_t=1.6,
            )
            rec["dem"] = dem
            rec["combined"] = combined_score(rec["fitness"], dem,
                                              structural_weight=0.5)
        results.append(rec)

    if pareto:
        from loop.pareto import Candidate, rank_all, select_diverse
        cand_objs: list = []
        for i, r in enumerate(results):
            f = r["fitness"]
            # All maximise: higher score = better on each axis.
            vec = [f["mass_score"], f["safety_factor_score"],
                   f["wear_score"], f["manufacturability_score"]]
            if "dem" in r:
                vec.append(r["dem"]["throughput_score"])
                vec.append(r["dem"]["wear_uniformity_score"])
            cand_objs.append(Candidate(id=str(i), objectives=vec, payload=r))
        ranks = rank_all(cand_objs)
        pareto_front = [c for c in cand_objs if ranks[c.id] == 0]
        selected = select_diverse(pareto_front, top_k) if pareto_front else []
        if len(selected) < top_k:
            extras = sorted([c for c in cand_objs if ranks[c.id] > 0],
                            key=lambda c: ranks[c.id])[:top_k - len(selected)]
            selected = list(selected) + list(extras)
        top = [c.payload for c in selected]
    else:
        def sort_key(r):
            if "combined" in r:
                return -r["combined"]["total"]
            return -r["fitness"]["composite"]
        results.sort(key=sort_key)
        top = results[:top_k]

    return {
        "part": part, "model": model, "ore": ore,
        "css_mm": css_mm, "target_tph": target_tph,
        "candidates_total": len(candidates),
        "dfm_passed": len(passed),
        "dfm_rejected": len(rejected),
        "rejected": rejected,
        "top_k": top,
        "ranking_mode": "pareto" if pareto else "composite",
        "dry_run": dry_run,
        "mbd_load_case": mbd_case,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("model")
    ap.add_argument("--ore", default="basalt")
    ap.add_argument("--css", type=float, default=80.0)
    ap.add_argument("--tph", type=float, default=65.0)
    ap.add_argument("--sweep", default="")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="skip real FEA/DEM, exercise plumbing only")
    ap.add_argument("--use-mbd", action="store_true",
                    help="derive FEA loads from MBD reactions instead of "
                         "yaml load_cases (recommended for non-jaw parts)")
    ap.add_argument("--peak-crushing-force-N", type=float, default=850_000.0,
                    help="peak jaw-face crushing force in newtons (MBD input)")
    ap.add_argument("--pareto", action="store_true",
                    help="select top-K from the Pareto-optimal set with "
                         "crowding-distance diversity (vs scalar composite)")
    args = ap.parse_args(argv)

    sweep = parse_sweep_arg(args.sweep)
    out = run(args.part, args.model, args.ore, args.css, args.tph,
              sweep, args.top_k, args.dry_run,
              use_mbd=args.use_mbd,
              peak_crushing_force_N=args.peak_crushing_force_N,
              pareto=args.pareto)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
