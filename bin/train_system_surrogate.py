#!/usr/bin/env python3
"""
Train the system-level KPI surrogate.

Loads or generates training data, fits the `SystemKPISurrogate`, saves the
checkpoint to disk. Use the saved file with `run_assembly.py --surrogate
path/to/file.pt` to enable surrogate-screened search.

Today: bootstrap on `loop.synthetic_data.generate` (ISO-281-shaped synthetic
KPIs). When real run history accumulates in `templates/<part>/instances/
<model>.history.jsonl`, replace the data source with a real loader.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop.assembly_features import load_schema
from loop.synthetic_data import generate, TARGET_KEYS


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="PE_400x600",
                    help="assembly model to train surrogate for")
    ap.add_argument("--n-samples", type=int, default=400,
                    help="synthetic samples (bootstrap mode)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/surrogates/system_v1.pt",
                    help="output checkpoint path (relative to repo root)")
    args = ap.parse_args(argv)

    try:
        from loop.system_surrogate import SystemKPISurrogate
    except ImportError as e:
        print(f"torch not installed: {e}", file=sys.stderr)
        return 1

    schema = load_schema(ROOT)
    print(f"generating {args.n_samples} synthetic samples for {args.model}...")
    samples = generate(n_samples=args.n_samples, model=args.model,
                        root=ROOT, seed=args.seed)
    print(f"  schema: node_dim={schema.node_dim}, edge_dim={schema.edge_dim}, "
          f"global_dim={schema.global_dim}")

    surrogate = SystemKPISurrogate(schema=schema, target_keys=TARGET_KEYS,
                                     hidden=args.hidden)
    print(f"training: hidden={args.hidden}, epochs={args.epochs}, lr={args.lr}")
    history = surrogate.fit(samples, epochs=args.epochs, lr=args.lr,
                              batch_size=16, val_split=0.2, seed=args.seed)
    print(f"  final train loss: {history['final_train_loss']:.4f}")
    print(f"  final val loss:   {history['final_val_loss']:.4f}")
    print(f"  n_train: {history['n_train']}, n_val: {history['n_val']}")

    out_path = ROOT / args.out
    surrogate.save(out_path)
    print(f"saved checkpoint to {out_path}")
    print(json.dumps({"out": str(out_path), "history": history}, indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
