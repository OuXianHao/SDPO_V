#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

ANSWER_LETTERS = "ABCDEF"


def is_enough_sample(obj):
    return obj.get("is_enough") == "Enough" or obj.get("is_enough_trans") is True


def answer_index_to_letter(answer):
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


def frame_sort_key(path):
    name = Path(path).name
    m = re.match(r"^(\d+)_", name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 10**9


def get_sorted_frames(frame_dir):
    frame_dir = Path(frame_dir)
    frames = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg")) + list(frame_dir.glob("*.png"))
    return sorted(frames, key=frame_sort_key)


def get_ground_frame_indices(obj):
    indices = obj.get("ground_frames_trans_improve") or obj.get("ground_frames_trans")

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

    s = obj.get("ground_frames", "")
    if isinstance(s, str):
        return [int(x) for x in re.findall(r"\d+", s)]

    return []


def normalize_one(obj, json_path, frames_root, expected_frames):
    if not is_enough_sample(obj):
        return None, "not_enough"

    question = obj.get("Question")
    choices = obj.get("Answer_Choices")
    answer_index = obj.get("Answer")
    answer = answer_index_to_letter(answer_index)
    video_name = obj.get("video_name")

    if not question:
        return None, "missing_question"
    if not isinstance(choices, list) or len(choices) == 0:
        return None, "bad_choices"
    if answer is None:
        return None, "bad_answer"
    if not video_name:
        return None, "missing_video_name"

    video_stem = Path(video_name).stem
    frame_dir = Path(frames_root) / video_stem

    if not frame_dir.exists():
        return None, "missing_frame_dir"

    frame_paths = get_sorted_frames(frame_dir)
    if len(frame_paths) != expected_frames:
        return None, f"bad_frame_count_{len(frame_paths)}"

    parsed_frame_indices = []
    for p in frame_paths:
        m = re.match(r"^(\d+)_", p.name)
        if not m:
            return None, "bad_frame_name"
        parsed_frame_indices.append(int(m.group(1)))

    if parsed_frame_indices != list(range(expected_frames)):
        return None, "bad_frame_indices"

    raw_ground_indices = get_ground_frame_indices(obj)
    valid_ground_indices = sorted({
        int(x) for x in raw_ground_indices
        if isinstance(x, int) and 0 <= x < expected_frames
    })

    if len(valid_ground_indices) == 0:
        return None, "empty_valid_ground_frames"

    options = []
    for i, choice in enumerate(choices):
        if i >= len(ANSWER_LETTERS):
            break
        options.append(f"{ANSWER_LETTERS[i]}. {choice}")

    sample = {
        "question": question,
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "video_name": video_name,
        "video_stem": video_stem,
        "frame_dir": str(frame_dir),
        "frame_paths": [str(p) for p in frame_paths],
        "ground_frame_indices": valid_ground_indices,
        "question_id": obj.get("question_id") or Path(json_path).stem,
        "type": obj.get("Type"),
        "category": obj.get("category"),
        "source_json": str(json_path),
    }

    return sample, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="/data/xhou/datasets/ground_jsons_trans_improve_trans")
    parser.add_argument("--frames_root", default="/data/xhou/datasets/frames_32")
    parser.add_argument("--output", default="/data/xhou/datasets/rlsd_v_frames32_train.jsonl")
    parser.add_argument("--expected_frames", type=int, default=32)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    error_counts = {}

    with open(out_path, "w", encoding="utf-8") as fout:
        for json_path in sorted(json_dir.glob("*.json")):
            total += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                error = "json_load_error"
                error_counts[error] = error_counts.get(error, 0) + 1
                continue

            sample, error = normalize_one(
                obj=obj,
                json_path=json_path,
                frames_root=args.frames_root,
                expected_frames=args.expected_frames,
            )

            if error is not None:
                error_counts[error] = error_counts.get(error, 0) + 1
                continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 80)
    print(f"total_json: {total}")
    print(f"kept:       {kept}")
    print(f"output:     {out_path}")
    print("-" * 80)
    print("error_counts:")
    for k, v in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()