from __future__ import annotations

import argparse
from pathlib import Path

from video_to_3d.metrics import load_npz, run_no_gt_evaluation, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction coherence without ground truth.")
    parser.add_argument("--predictions", type=Path, required=True, help="VGGT-Omega predictions.npz.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictions = load_npz(args.predictions)
    metrics = run_no_gt_evaluation(predictions)
    save_json(metrics, args.out)
    print(f"[done] wrote {args.out}")
    print(f"[done] coherence score: {metrics['quality_gates']['coherence_score_0_to_1']:.3f}")


if __name__ == "__main__":
    main()

