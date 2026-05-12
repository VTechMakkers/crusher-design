#!/usr/bin/env python3
"""
Run global sensitivity analysis on a (part, model) design loop.

Outputs Sobol first-order + total-order indices for the chosen KPI,
ranked by total-order share of variance. Tells the engineer which
parameters dominate the KPI — so the next drawing-ingest session focuses
on measuring those dimensions most carefully.

Usage:
  python bin/run_sensitivity.py toggle_plate PE_400x600 [--kpi composite]
                                                          [--samples 1024]
                                                          [--bounds bounds.yaml]
                                                          [--json]

Example output (default KPI = composite fitness):

  Sobol sensitivity of composite
    base samples: 1024, total evaluations: 18432, invalid: 0

    param           S1 (first)  S1±95%CI    ST (total)  interaction
    thickness_mm    +0.412     ±0.024       +0.493     0.081
    web_height_mm   +0.218     ±0.019       +0.301     0.083
    ...
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.sensitivity import analyze_part


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part")
    ap.add_argument("model")
    ap.add_argument("--kpi", default="composite",
                    help="target KPI to decompose")
    ap.add_argument("--samples", type=int, default=1024,
                    help="Saltelli base sample count N "
                         "(total evaluations = N(2k+2) for k parameters)")
    ap.add_argument("--bounds",
                    help="YAML file: {param_name: [lo, hi]} per parameter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON output instead of human format")
    args = ap.parse_args(argv)

    bounds = None
    if args.bounds:
        raw = yaml.safe_load(Path(args.bounds).read_text())
        bounds = {k: tuple(v) for k, v in raw.items()}

    result = analyze_part(
        part=args.part, model=args.model, target_kpi=args.kpi,
        parameter_bounds=bounds,
        n_samples_base=args.samples, seed=args.seed,
    )

    if args.json:
        print(json.dumps({
            "target_kpi": result.target_kpi,
            "parameter_names": list(result.parameter_names),
            "first_order": result.first_order,
            "first_order_ci": result.first_order_ci,
            "total_order": result.total_order,
            "total_order_ci": result.total_order_ci,
            "interaction_strength": result.interaction_strength(),
            "n_samples_base": result.n_samples_base,
            "n_evaluations": result.n_evaluations,
            "n_invalid_samples": result.n_invalid_samples,
            "bounds": {n: list(v) for n, v in result.bounds.items()},
        }, indent=2, default=str))
    else:
        for line in result.summary_lines():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
