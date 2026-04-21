import argparse
import sys
from pathlib import Path

TIMEMIXER_ROOT = Path(__file__).resolve().parent
MODELING_ROOT = TIMEMIXER_ROOT.parent
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, TIMEMIXER_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from official_benchmark import SUPPORTED_INPUT_VARIANTS, default_time_series_root, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official TimeMixer runner for fusion/text/image/structured windows")
    parser.add_argument("--mode", choices=["official", "debug"], default="official")
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--input-variant", choices=SUPPORTED_INPUT_VARIANTS, default="fusion")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--smoke", action="store_true", help="Skip tuning and run a single-seed official smoke benchmark.")
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(MODELING_ROOT / "results" / "official"),
    )
    parser.add_argument("--fusion-method", type=str, default="late.gru_gate")
    parser.add_argument("--debug-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "l1", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.02)
    parser.add_argument("--target-mode", choices=["level", "residual"], default="level")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")
    parser.add_argument("--forecast-horizon-days", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--down-sampling-layers", type=int, default=2)
    parser.add_argument("--moving-avg", type=int, default=3)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.root_path is None:
        args.root_path = default_time_series_root(args.input_variant, args.frequency)
    run_experiment(args)
