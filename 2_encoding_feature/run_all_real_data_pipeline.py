#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified encoding pipeline entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structured.process_structured import process_structured_data
from text_bert.run_text_pipeline import run_text_pipeline
from image_clip.run_image_pipeline import run_image_pipeline
from fusion.run_fusion_pipeline import run_fusion_pipeline
from time_series.run_time_series_pipeline import run_all_window_lengths
from project_shared.frequency import expand_frequency_choice


STAGE_ORDER = ["structured", "text", "image", "fusion", "time_series"]


def run_stage(label, fn, **kwargs):
    print(f"\n--- {label} ---")
    start_time = time.time()
    fn(**kwargs)
    duration = time.time() - start_time
    print(f"{label} completed in {duration:.2f} seconds")


def run_all_pipeline(stop_after: str = "time_series", frequency: str = "both"):
    print("=== Starting multimodal encoding pipeline ===")
    frequencies = expand_frequency_choice(frequency)
    for current_frequency in frequencies:
        print(f"\n### Frequency = {current_frequency} ###")
        run_stage("Step 1: Process structured data", process_structured_data, frequency=current_frequency)
        if stop_after == "structured":
            continue

        run_stage("Step 2: Text encoding and aggregation", run_text_pipeline, frequency=current_frequency)
        if stop_after == "text":
            continue

        run_stage("Step 3: Image encoding and aggregation", run_image_pipeline, frequency=current_frequency)
        if stop_after == "image":
            continue

        run_stage("Step 4: Multimodal fusion", run_fusion_pipeline, frequency=current_frequency)
        if stop_after == "fusion":
            continue

        run_stage("Step 5: Time-series window construction", run_all_window_lengths, frequency=current_frequency)

    print("\n=== Multimodal encoding pipeline completed ===")
    print("Output directory: 2_encoding_feature/outputs/{daily,monthly}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the dual-frequency encoding pipeline.")
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default="time_series",
        help="Stop after the specified stage.",
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "monthly", "both"],
        default="both",
        help="Frequency branch to execute.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_pipeline(stop_after=args.stop_after, frequency=args.frequency)
