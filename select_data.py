#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from collections import Counter
from typing import Any, Dict, List, Optional


INPUT_JSON = "/ssd5/zhzhu/datasets/Video-R1-data/Video-R1-260k.json"
DATASET_ROOT = "/ssd5/zhzhu/datasets/Video-R1-data"
RAW_OUT = "/ssd5/zhzhu/datasets/Video-R1-data/video_mcq_raw.json"
CLEAN_OUT = "/ssd5/zhzhu/datasets/Video-R1-data/video_mcq_clean.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def parse_gold_option(solution: Any) -> Optional[str]:
    """
    尽量从 solution 里解析出标准单选答案 A/B/C/D/E/F。
    支持常见格式，例如：
      <answer>A</answer>
      A
      a
      The answer is A
      <answer> B </answer>
    若无法稳定解析，则返回 None。
    """
    s = normalize_text(solution)
    if not s:
        return None

    valid_choices = {"A", "B", "C", "D", "E", "F"}

    # 优先匹配 <answer>...</answer>
    m = re.search(r"<answer>\s*([A-Fa-f])\s*</answer>", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 其次匹配全文只有单个 A/B/C/D/E/F
    m = re.fullmatch(r"\s*([A-Fa-f])\s*", s)
    if m:
        return m.group(1).upper()

    # 再宽松一点：文本中出现独立的 A/B/C/D/E/F
    matches = re.findall(r"\b([A-Fa-f])\b", s)
    matches = [x.upper() for x in matches]
    matches = [x for x in matches if x in valid_choices]

    if len(matches) == 1:
        return matches[0]

    return None


def resolve_video_path(sample_path: str, dataset_root: Optional[str]) -> str:
    """
    将样本中的 path 转成可检查的真实路径。
    若 sample_path 是相对路径且给了 dataset_root，则拼接。
    若是绝对路径，则直接返回。
    """
    p = normalize_text(sample_path)
    if not p:
        return ""

    if os.path.isabs(p):
        return os.path.normpath(p)

    if dataset_root:
        return os.path.normpath(os.path.join(dataset_root, p))

    return os.path.normpath(p)


def main() -> None:
    data = load_json(INPUT_JSON)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON to be a list, but got: {type(data)}")

    total_count = len(data)

    raw_subset: List[Dict[str, Any]] = []
    clean_subset: List[Dict[str, Any]] = []

    data_source_counter = Counter()
    gold_counter = Counter()

    num_video = 0
    num_video_mcq = 0
    num_parseable_gold = 0
    num_path_exists = 0
    num_clean = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        data_type = normalize_text(item.get("data_type")).lower()
        problem_type = normalize_text(item.get("problem_type")).lower()

        if data_type == "video":
            num_video += 1

        if not (data_type == "video" and problem_type == "multiple choice"):
            continue

        num_video_mcq += 1

        sample = dict(item)
        gold = parse_gold_option(sample.get("solution"))
        resolved_path = resolve_video_path(sample.get("path", ""), DATASET_ROOT)
        path_exists = bool(resolved_path) and os.path.exists(resolved_path)

        if gold is not None:
            num_parseable_gold += 1
            gold_counter[gold] += 1

        if path_exists:
            num_path_exists += 1

        data_source = normalize_text(sample.get("data_source"))
        if data_source:
            data_source_counter[data_source] += 1

        sample["gold_option"] = gold
        sample["resolved_video_path"] = resolved_path
        sample["video_path_exists"] = path_exists

        raw_subset.append(sample)

        # clean 版本：要求答案可解析 + 路径存在
        if gold is not None and path_exists:
            clean_subset.append(sample)
            num_clean += 1

    save_json(raw_subset, RAW_OUT)
    save_json(clean_subset, CLEAN_OUT)

    print("=" * 80)
    print("Filter summary")
    print("=" * 80)
    print(f"Input file                : {INPUT_JSON}")
    print(f"Dataset root              : {DATASET_ROOT}")
    print(f"Total samples             : {total_count}")
    print(f"Video samples             : {num_video}")
    print(f"Video + MCQ samples       : {num_video_mcq}")
    print(f"Parseable gold option     : {num_parseable_gold}")
    print(f"Existing video path       : {num_path_exists}")
    print(f"Clean subset size         : {num_clean}")
    print(f"Raw output                : {RAW_OUT}")
    print(f"Clean output              : {CLEAN_OUT}")

    print("\nGold option distribution (within parseable samples):")
    if gold_counter:
        for k in ["A", "B", "C", "D", "E", "F"]:
            print(f"  {k}: {gold_counter.get(k, 0)}")
    else:
        print("  No parseable gold labels found.")

    print("\nTop 20 data sources in raw subset:")
    for ds, cnt in data_source_counter.most_common(20):
        print(f"  {ds}: {cnt}")

    print("=" * 80)


if __name__ == "__main__":
    main()