#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity check script for RLSD / SDPO-V new 32-frame dataset.

This script checks:
1. JSON files under ground_jsons_trans_improve_trans
2. Enough filtering
3. Answer 0-based index -> letter conversion
4. frame_dir resolution under frames_32
5. exactly 32 jpg frames
6. frame sorting by sampled index before "_"
7. ground frame index validity
8. normalized examples for future dataset loader
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean


ANSWER_LETTERS = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        default="/data/xhou/datasets/ground_jsons_trans_improve_trans",
        help="Directory containing one JSON file per sample.",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="/data/xhou/datasets/frames_32",
        help="Root directory containing 32-frame folders.",
    )
    parser.add_argument(
        "--expected_frames",
        type=int,
        default=32,
        help="Expected number of uniformly sampled frames.",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=3,
        help="Number of normalized examples to print.",
    )
    return parser.parse_args()


def is_enough_sample(obj):
    """
    Keep sample if:
    - is_enough == "Enough"
    OR
    - is_enough_trans is True
    """
    return obj.get("is_enough") == "Enough" or obj.get("is_enough_trans") is True


def answer_index_to_letter(answer):
    """
    Convert 0-based answer index to A/B/C/D/E/F.

    Example:
    0 -> A
    1 -> B
    2 -> C
    """
    if isinstance(answer, bool):
        return None

    if isinstance(answer, int):
        idx = answer
    elif isinstance(answer, str) and answer.strip().isdigit():
        idx = int(answer.strip())
    else:
        return None

    if 0 <= idx < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[idx]
    return None


def get_video_stem(video_name):
    """
    Convert:
    LqwbcxG4Si8.120t300.1.mp4 -> LqwbcxG4Si8.120t300.1
    """
    if not video_name:
        return None
    return Path(video_name).stem


def frame_sort_key(path):
    """
    Sort frame names by the first integer before "_".

    Example:
    0_0.jpg -> 0
    12_43.jpg -> 12
    """
    name = Path(path).name
    m = re.match(r"^(\d+)_", name)
    if m:
        return int(m.group(1))

    # fallback: extract first number anywhere
    nums = re.findall(r"\d+", name)
    if nums:
        return int(nums[0])

    return 10**9


def get_sorted_frames(frame_dir):
    frame_dir = Path(frame_dir)
    frames = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg")) + list(frame_dir.glob("*.png"))
    frames = sorted(frames, key=frame_sort_key)
    return frames


def get_ground_frame_indices(obj):
    """
    Prefer ground_frames_trans_improve.
    Fallback to ground_frames_trans.
    Finally fallback to parsing string like:
    "Frame 4, Frame 12, Frame 13"
    """
    indices = obj.get("ground_frames_trans_improve", None)

    if not indices:
        indices = obj.get("ground_frames_trans", None)

    if isinstance(indices, list):
        clean = []
        for x in indices:
            if isinstance(x, bool):
                continue
            if isinstance(x, int):
                clean.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                clean.append(int(x.strip()))
        return clean

    # fallback: parse ground_frames string
    s = obj.get("ground_frames", "")
    if isinstance(s, str):
        nums = re.findall(r"\d+", s)
        return [int(x) for x in nums]

    return []


def normalize_sample(obj, json_path, frames_root, expected_frames):
    question = obj.get("Question")
    answer_choices = obj.get("Answer_Choices")
    answer_raw = obj.get("Answer")
    answer_letter = answer_index_to_letter(answer_raw)

    video_name = obj.get("video_name")
    video_stem = get_video_stem(video_name)
    frame_dir = str(Path(frames_root) / video_stem) if video_stem else None

    ground_frame_indices = get_ground_frame_indices(obj)

    options = None
    if isinstance(answer_choices, list):
        options = []
        for i, choice in enumerate(answer_choices):
            if i < len(ANSWER_LETTERS):
                options.append(f"{ANSWER_LETTERS[i]}. {choice}")

    normalized = {
        "json_path": str(json_path),
        "question": question,
        "options": options,
        "answer": answer_letter,
        "answer_index": answer_raw,
        "video_name": video_name,
        "video_stem": video_stem,
        "frame_dir": frame_dir,
        "ground_frame_indices": ground_frame_indices,
        "question_id": obj.get("question_id"),
        "type": obj.get("Type"),
        "category": obj.get("category"),
    }

    errors = []

    if not question:
        errors.append("missing_question")

    if not isinstance(answer_choices, list) or len(answer_choices) == 0:
        errors.append("missing_or_empty_answer_choices")

    if answer_letter is None:
        errors.append("invalid_answer")

    if not video_name:
        errors.append("missing_video_name")

    if not video_stem:
        errors.append("missing_video_stem")

    if not frame_dir or not Path(frame_dir).exists():
        errors.append("missing_frame_dir")
        frame_count = 0
    else:
        frames = get_sorted_frames(frame_dir)
        frame_count = len(frames)

        if frame_count != expected_frames:
            errors.append(f"bad_frame_count_{frame_count}")

        # Check whether frame indices are exactly 0..31 after sorting.
        parsed_indices = []
        for p in frames:
            m = re.match(r"^(\d+)_", p.name)
            if m:
                parsed_indices.append(int(m.group(1)))

        if len(parsed_indices) == expected_frames:
            expected = list(range(expected_frames))
            if parsed_indices != expected:
                errors.append("frame_indices_not_0_to_31_or_bad_sort")

    valid_ground = []
    invalid_ground = []
    for idx in ground_frame_indices:
        if isinstance(idx, int) and 0 <= idx < expected_frames:
            valid_ground.append(idx)
        else:
            invalid_ground.append(idx)

    normalized["valid_ground_frame_indices"] = valid_ground
    normalized["invalid_ground_frame_indices"] = invalid_ground
    normalized["frame_count"] = frame_count

    if len(valid_ground) == 0:
        errors.append("empty_valid_ground_frames")

    if len(invalid_ground) > 0:
        errors.append("invalid_ground_frame_indices")

    return normalized, errors


def main():
    args = parse_args()

    json_dir = Path(args.json_dir)
    frames_root = Path(args.frames_root)

    if not json_dir.exists():
        raise FileNotFoundError(f"json_dir does not exist: {json_dir}")

    if not frames_root.exists():
        raise FileNotFoundError(f"frames_root does not exist: {frames_root}")

    json_files = sorted(json_dir.glob("*.json"))

    total = len(json_files)
    enough_count = 0
    normalized_count = 0

    error_counter = Counter()
    answer_counter = Counter()
    category_counter = Counter()
    type_counter = Counter()
    key_frame_counts = []

    examples = []
    bad_examples = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            error_counter["json_load_error"] += 1
            if len(bad_examples) < 5:
                bad_examples.append({
                    "json_path": str(json_path),
                    "errors": [f"json_load_error: {repr(e)}"],
                })
            continue

        if not is_enough_sample(obj):
            continue

        enough_count += 1

        normalized, errors = normalize_sample(
            obj=obj,
            json_path=json_path,
            frames_root=frames_root,
            expected_frames=args.expected_frames,
        )

        for err in errors:
            error_counter[err] += 1

        if errors:
            if len(bad_examples) < 5:
                bad_examples.append({
                    "json_path": str(json_path),
                    "errors": errors,
                    "sample": normalized,
                })
            continue

        normalized_count += 1

        answer_counter[normalized["answer"]] += 1
        category_counter[normalized["category"]] += 1
        type_counter[normalized["type"]] += 1
        key_frame_counts.append(len(normalized["valid_ground_frame_indices"]))

        if len(examples) < args.print_examples:
            examples.append(normalized)

    print("=" * 80)
    print("RLSD / SDPO-V new 32-frame dataset sanity check")
    print("=" * 80)
    print(f"json_dir:        {json_dir}")
    print(f"frames_root:     {frames_root}")
    print(f"expected_frames: {args.expected_frames}")
    print("-" * 80)

    print(f"total_json_files:        {total}")
    print(f"kept_enough_samples:     {enough_count}")
    print(f"valid_normalized_samples:{normalized_count}")

    if enough_count > 0:
        print(f"valid_ratio_in_enough:   {normalized_count / enough_count:.4f}")

    print("-" * 80)
    print("error_counter:")
    if error_counter:
        for k, v in error_counter.most_common():
            print(f"  {k}: {v}")
    else:
        print("  No errors found.")

    print("-" * 80)
    print("answer_distribution:")
    for k in sorted(answer_counter.keys()):
        print(f"  {k}: {answer_counter[k]}")

    print("-" * 80)
    print("key_frame_stats:")
    if key_frame_counts:
        print(f"  avg_key_frames: {mean(key_frame_counts):.3f}")
        print(f"  min_key_frames: {min(key_frame_counts)}")
        print(f"  max_key_frames: {max(key_frame_counts)}")
    else:
        print("  No valid key-frame stats.")

    print("-" * 80)
    print("top_categories:")
    for k, v in category_counter.most_common(10):
        print(f"  {k}: {v}")

    print("-" * 80)
    print("top_types:")
    for k, v in type_counter.most_common(10):
        print(f"  {k}: {v}")

    print("-" * 80)
    print(f"normalized_examples first {len(examples)}:")
    for i, ex in enumerate(examples):
        print(f"\nExample {i + 1}:")
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    if bad_examples:
        print("-" * 80)
        print(f"bad_examples first {len(bad_examples)}:")
        for i, ex in enumerate(bad_examples):
            print(f"\nBad example {i + 1}:")
            print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("=" * 80)

    if error_counter:
        print("Finished with warnings/errors. Please inspect error_counter and bad_examples.")
    else:
        print("Finished successfully. Dataset looks ready for loader adaptation.")


if __name__ == "__main__":
    main()