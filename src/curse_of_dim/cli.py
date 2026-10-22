"""Command-line orchestration for the dimensionality lab."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curse-of-dim",
        description="Reproducible experiments on high-dimensional geometry, KNN and generalization.",
    )
    parser.add_argument(
        "--experiment", "-e", action="append", default=None,
        help="Experiment name (repeatable, comma-separated, or 'all'). See --list.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    # Headless-safe even on minimal Linux/CI runners.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .experiments import EXPERIMENTS, run_selected

    if args.list:
        for name, fn in EXPERIMENTS.items():
            print(f"{name:22s} {fn.__doc__ or ''}")
        return 0

    tokens = [part.strip() for arg in (args.experiment or ["all"]) for part in arg.split(",")]
    unknown = set(tokens) - set(EXPERIMENTS) - {"all"}
    if unknown:
        make_parser().error(f"Unknown experiment(s): {', '.join(sorted(unknown))}")
    names = ([n for n in EXPERIMENTS if n != "autoencoder_manifold"]
             if "all" in tokens else list(dict.fromkeys(tokens)))
    metrics = run_selected(names, out=args.out)
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Finished {len(names)} experiment(s); metrics: {metrics_path.resolve()}")
    return 0
