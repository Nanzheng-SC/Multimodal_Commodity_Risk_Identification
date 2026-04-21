from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


MODELING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODELING_ROOT.parent
for path in (PROJECT_ROOT, MODELING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baselines.har.run_har_benchmark import har_windows_for_frequency, run_har_benchmark
from baselines.lstm.run_lstm_benchmark import run_lstm_benchmark
from TimeMixer.official_benchmark import default_time_series_root, run_official as run_timemixer_official
from TimeMixer.official_benchmark import current_fusion_method
from common.comparison import (
    load_official_record,
    save_leaderboard,
    save_metric_barplots,
    save_prediction_overlay,
    save_stability_plot,
)
from common.paths import final_model_comparison_dir
from common.reporting import clean_directory
from project_shared.frequency import default_window_lengths, expand_frequency_choice, normalize_frequency


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official modeling benchmarks for HAR, LSTM, and TimeMixer.")
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated subset of har,lstm,timemixer or 'all'.",
    )
    parser.add_argument("--frequency", choices=["daily", "monthly", "both"], default="monthly")
    return parser


def better_official_dir(candidate_a: Path, candidate_b: Path) -> Path:
    record_a = load_official_record("candidate_a", candidate_a)
    record_b = load_official_record("candidate_b", candidate_b)
    key_a = (
        record_a["diagnostics_rank"],
        record_a["deployed_final_valid_rmse"],
        record_a["deployed_test_rmse"],
        record_a["test_rmse_std"],
        -record_a["direction_acc_mean"],
        record_a["window_length"],
    )
    key_b = (
        record_b["diagnostics_rank"],
        record_b["deployed_final_valid_rmse"],
        record_b["deployed_test_rmse"],
        record_b["test_rmse_std"],
        -record_b["direction_acc_mean"],
        record_b["window_length"],
    )
    return candidate_a if key_a <= key_b else candidate_b


def run_all(window_length: int | None, models: set[str], frequency: str) -> dict[str, Path]:
    frequency = normalize_frequency(frequency)
    outputs: dict[str, Path] = {}
    default_windows = default_window_lengths(frequency)

    if "har" in models:
        outputs["HAR"] = run_har_benchmark(
            window_length=max(har_windows_for_frequency(frequency)),
            input_variant="fusion",
            frequency=frequency,
        )
    if "lstm" in models:
        candidate_dirs = [
            run_lstm_benchmark(
                window_length=candidate_window,
                input_variant="fusion",
                fusion_method="late.gru_gate",
                frequency=frequency,
            )
            for candidate_window in default_windows
        ]
        best_dir = candidate_dirs[0]
        for candidate_dir in candidate_dirs[1:]:
            best_dir = better_official_dir(best_dir, candidate_dir)
        outputs["LSTM-window"] = best_dir
    if "timemixer" in models:
        base_window = int(window_length or default_windows[1 if len(default_windows) > 1 else 0])
        args = SimpleNamespace(
            mode="official",
            window_length=base_window,
            input_variant="fusion",
            frequency=frequency,
            root_path=default_time_series_root("fusion", frequency),
            results_dir=None,
            fusion_method=current_fusion_method(frequency),
            debug_seed=42,
            batch_size=16,
            max_epochs=40,
            patience=8,
            learning_rate=5e-4,
            weight_decay=1e-4,
            loss="huber",
            huber_delta=0.02,
            target_mode="residual",
            target_transform="none",
            grad_clip=1.0,
        )
        outputs["TimeMixer"] = run_timemixer_official(args)
    return outputs


def build_comparison(outputs: dict[str, Path], frequency: str) -> Path:
    output_dir = final_model_comparison_dir(frequency)
    clean_directory(output_dir)
    records = []
    model_dirs = {}
    for display_name, model_dir in outputs.items():
        records.append(load_official_record(display_name, model_dir))
        model_dirs[display_name] = model_dir

    leaderboard = save_leaderboard(records, output_dir)
    save_metric_barplots(leaderboard, output_dir)
    save_stability_plot(leaderboard, output_dir)
    save_prediction_overlay(model_dirs, output_dir)
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    if args.models == "all":
        model_set = {"har", "lstm", "timemixer"}
    else:
        model_set = {item.strip().lower() for item in args.models.split(",") if item.strip()}

    for frequency in expand_frequency_choice(args.frequency):
        outputs = run_all(args.window_length, model_set, frequency)
        comp_dir = build_comparison(outputs, frequency)
        print(f"[{frequency}] Comparison output directory: {comp_dir.resolve()}")


if __name__ == "__main__":
    main()
