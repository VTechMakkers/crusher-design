#!/usr/bin/env python3
"""
Whole-machine optimisation driver.

Runs the assembly-level design loop: per-part sweeps + cross-part
constraint enforcement + system aggregation + Pareto rank across complete
crushers. Returns a diverse Pareto front of validated whole-machine
designs — the customer picks one trade-off point.

Usage:
  python bin/run_assembly.py PE_400x600 [options]
  python bin/run_assembly.py PE_400x600 --sweep-config sweeps.yaml --top-k 5
  python bin/run_assembly.py PE_400x600 --ore granite --peak-crushing-force-N 950000

sweep-config schema (YAML):
  toggle_plate:
    thickness_mm: [24, 28, 32]
  swing_jaw_plate:
    tooth_pitch_mm: [80, 90, 100]
    tooth_depth_mm: [18, 22, 26]
  fixed_jaw_plate:
    tooth_pitch_mm: [80, 90, 100]   # must include matching pitches to swing
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.assembly_loop import run_assembly


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="assembly model (e.g. PE_400x600)")
    ap.add_argument("--ore", default="basalt",
                    help="ore type from knowledge/ores.yaml")
    ap.add_argument("--peak-crushing-force-N", type=float, default=850_000.0,
                    help="peak jaw-face crushing force in newtons")
    ap.add_argument("--sweep-config",
                    help="path to YAML mapping part_class -> {param: [values]}")
    ap.add_argument("--per-part-top-k", type=int, default=3,
                    help="how many variants to keep per part after sweep")
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many Pareto-front whole crushers to return")
    ap.add_argument("--no-mbd", action="store_true",
                    help="skip MBD-derived loads, use yaml load_cases")
    ap.add_argument("--surrogate",
                    help="path to trained surrogate checkpoint — switches to "
                         "two-stage surrogate-screened search")
    ap.add_argument("--n-screened-to-keep", type=int, default=20,
                    help="when --surrogate set: top-N from surrogate Pareto "
                         "to forward to real evaluation")
    ap.add_argument("--algorithm", default="default",
                    choices=["default", "nsga2", "nsga3"],
                    help="Pareto algorithm: 'default' (built-in NSGA-II-style), "
                         "'nsga2' (pymoo), 'nsga3' (pymoo, recommended for "
                         "≥4 objectives — the system_fitness output has 6)")
    args = ap.parse_args(argv)

    sweep_config: dict[str, dict[str, list[Any]]] | None = None
    if args.sweep_config:
        sweep_config = yaml.safe_load(Path(args.sweep_config).read_text())

    if args.surrogate:
        if sweep_config is None:
            print("--surrogate requires --sweep-config", file=sys.stderr)
            return 2
        from loop.surrogate_screening import screen_with_surrogate
        result = screen_with_surrogate(
            model=args.model, sweep_config=sweep_config,
            surrogate_path=Path(args.surrogate),
            n_screened_to_keep=args.n_screened_to_keep,
            final_top_k=args.top_k, ore=args.ore,
            peak_crushing_force_N=args.peak_crushing_force_N,
            use_mbd=not args.no_mbd, root=ROOT,
        )
    else:
        result = run_assembly(
            model=args.model, ore=args.ore,
            peak_crushing_force_N=args.peak_crushing_force_N,
            sweep_config=sweep_config,
            per_part_top_k=args.per_part_top_k,
            top_k=args.top_k,
            use_mbd=not args.no_mbd,
            algorithm=args.algorithm,
            root=ROOT,
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
