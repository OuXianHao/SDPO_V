#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SDPO-V Margin Calibration Utility.

Estimates the natural distribution of delta_t = s_good - s_bad over valid
(reward==1) samples, then recommends a conservative global margin.

Usage:
    Run your normal SDPO training with these config overrides:
        algorithm.sdpo_v_enabled=True
        algorithm.sdpo_v_calibration=True

    The calibration statistics (count, mean, std, p10-p90) will be printed
    to stdout with the prefix [sdpo_v_calibration].

    After collecting stats from 100-300 valid samples, use the recommended
    margin (p25) as:
        algorithm.sdpo_v_margin=<recommended_value>
        algorithm.sdpo_v_calibration=False

Alternatively, this script can aggregate calibration logs from a training run:

    python -m verl.utils.sdpo_v_calibration --log-file <training_log.txt>

It will parse all [sdpo_v_calibration] lines, aggregate the delta statistics,
and produce a summary with a recommended margin.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Optional


def parse_calibration_lines(lines: list[str]) -> list[dict[str, float]]:
    """Parse [sdpo_v_calibration] log lines into list of stat dicts."""
    pattern = re.compile(r"\[sdpo_v_calibration\]\s*(\{.*\})")
    results = []
    for line in lines:
        m = pattern.search(line)
        if m:
            try:
                # The dict is printed with Python repr, convert to JSON-compatible
                dict_str = m.group(1).replace("'", '"')
                stats = json.loads(dict_str)
                if stats.get("calib_count", 0) > 0:
                    results.append(stats)
            except (json.JSONDecodeError, ValueError):
                continue
    return results


def aggregate_calibration_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate multiple calibration snapshots into a summary."""
    if not stats_list:
        return {"error": "No valid calibration entries found."}

    total_count = sum(s.get("calib_count", 0) for s in stats_list)
    num_snapshots = len(stats_list)

    # Weighted mean of means
    weighted_mean = sum(
        s["calib_mean"] * s["calib_count"] for s in stats_list
    ) / max(total_count, 1)

    # Collect all percentiles (simple average across snapshots)
    keys = ["calib_p10", "calib_p25", "calib_p50", "calib_p75", "calib_p90", "calib_std"]
    avg_stats = {}
    for key in keys:
        values = [s[key] for s in stats_list if key in s]
        avg_stats[key] = sum(values) / max(len(values), 1)

    recommended_margin = avg_stats.get("calib_p25", 0.0)

    return {
        "total_token_count": total_count,
        "num_snapshots": num_snapshots,
        "mean_delta": round(weighted_mean, 6),
        "avg_std": round(avg_stats.get("calib_std", 0.0), 6),
        "avg_p10": round(avg_stats.get("calib_p10", 0.0), 6),
        "avg_p25": round(avg_stats.get("calib_p25", 0.0), 6),
        "avg_p50": round(avg_stats.get("calib_p50", 0.0), 6),
        "avg_p75": round(avg_stats.get("calib_p75", 0.0), 6),
        "avg_p90": round(avg_stats.get("calib_p90", 0.0), 6),
        "recommended_margin_p25": round(recommended_margin, 6),
    }


def main(log_file: Optional[str] = None):
    parser = argparse.ArgumentParser(description="SDPO-V Margin Calibration Aggregator")
    parser.add_argument(
        "--log-file",
        type=str,
        default=log_file,
        help="Path to training log file containing [sdpo_v_calibration] lines.",
    )
    args = parser.parse_args()

    if args.log_file is None:
        print("Usage: python -m verl.utils.sdpo_v_calibration --log-file <training_log.txt>")
        print()
        print("To collect calibration data, run training with:")
        print("  algorithm.sdpo_v_enabled=True")
        print("  algorithm.sdpo_v_calibration=True")
        print()
        print("Then pass the training log to this script.")
        sys.exit(0)

    with open(args.log_file, "r") as f:
        lines = f.readlines()

    stats_list = parse_calibration_lines(lines)
    if not stats_list:
        print("No [sdpo_v_calibration] entries found in the log file.")
        sys.exit(1)

    summary = aggregate_calibration_stats(stats_list)
    print("=" * 60)
    print("SDPO-V Margin Calibration Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print(f"\nRecommended margin (p25): {summary['recommended_margin_p25']}")
    print(f"\nTo use this margin, set:")
    print(f"  algorithm.sdpo_v_margin={summary['recommended_margin_p25']}")
    print(f"  algorithm.sdpo_v_calibration=False")


if __name__ == "__main__":
    main()
