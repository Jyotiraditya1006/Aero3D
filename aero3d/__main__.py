from __future__ import annotations

import argparse
from pathlib import Path

from aero3d.pipeline import run_pipeline
from aero3d.synthetic import generate_synthetic_mission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aero3D single-pass drone video reconstruction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("synth", help="Generate a synthetic single-pass mission")
    p_syn.add_argument("--out", default="data/synthetic")
    p_syn.add_argument("--seconds", type=float, default=8.0)

    p_run = sub.add_parser("reconstruct", help="Run reconstruction")
    p_run.add_argument("--video", required=True)
    p_run.add_argument("--telemetry", required=True)
    p_run.add_argument("--out", default="outputs/run")
    p_run.add_argument("--config", default=None)

    p_demo = sub.add_parser("demo", help="Generate synthetic data and reconstruct")
    p_demo.add_argument("--out", default="outputs/demo")

    args = parser.parse_args(argv)

    if args.cmd == "synth":
        info = generate_synthetic_mission(args.out, seconds=args.seconds)
        print(info)
        return 0
    if args.cmd == "reconstruct":
        result = run_pipeline(args.video, args.telemetry, args.out, args.config, progress=_print_progress)
        print(result.metrics)
        print(f"Wrote artefacts to {result.output_dir}")
        return 0
    if args.cmd == "demo":
        out = Path(args.out)
        info = generate_synthetic_mission(out / "mission")
        result = run_pipeline(info["video"], info["telemetry"], out / "model", progress=_print_progress)
        print(result.metrics)
        print(f"Wrote artefacts to {result.output_dir}")
        return 0
    return 1


def _print_progress(msg: str, frac: float) -> None:
    print(f"[{frac:5.1%}] {msg}")


if __name__ == "__main__":
    raise SystemExit(main())
