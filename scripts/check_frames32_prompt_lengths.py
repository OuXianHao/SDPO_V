#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from jinja2 import Template
from transformers import AutoProcessor, AutoTokenizer


def build_base_content(sample):
    question = str(sample.get("question", "")).strip()
    options = sample.get("options", [])
    if not isinstance(options, list):
        options = []

    options_text = "\n".join(str(x) for x in options)
    prompt_body = f"Question:\n{question}\n\nOptions:\n{options_text}"
    wrapped_prompt = f"<video>\n{prompt_body}"
    return wrapped_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="/data/xhou/datasets/rlsd_v_frames32_train.jsonl")
    parser.add_argument("--model_path", default="/data/xhou/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--format_prompt", default="/data/xhou/frameworks/SDPO_V/examples/format_prompt/r1v.jinja")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--thresholds", type=str, default="2048,3072,4096,6144,8192,12288,16384")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    with open(args.format_prompt, "r", encoding="utf-8") as f:
        template = Template(f.read().strip())

    lengths = []
    records = []

    jsonl_path = Path(args.jsonl)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.max_samples > 0 and idx >= args.max_samples:
                break

            sample = json.loads(line)
            base_content = build_base_content(sample)

            # This matches dataset.py behavior:
            # _build_messages first applies format_prompt.render(content=...)
            formatted_content = template.render(content=base_content)

            # Then _build_messages splits <video> into video message + text.
            content_list = []
            for i, content in enumerate(formatted_content.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})

            messages = [{"role": "user", "content": content_list}]

            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            token_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            length = len(token_ids)
            lengths.append(length)

            records.append({
                "idx": idx,
                "question_id": sample.get("question_id"),
                "video_stem": sample.get("video_stem"),
                "answer": sample.get("answer"),
                "length": length,
                "question": sample.get("question", "")[:120],
            })

    arr = np.array(lengths, dtype=np.int64)
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]

    print("=" * 80)
    print("Prompt length statistics")
    print("=" * 80)
    print(f"jsonl:          {args.jsonl}")
    print(f"model_path:     {args.model_path}")
    print(f"format_prompt:  {args.format_prompt}")
    print(f"num_samples:    {len(arr)}")
    print("-" * 80)

    print(f"min:   {int(arr.min())}")
    print(f"max:   {int(arr.max())}")
    print(f"mean:  {float(arr.mean()):.2f}")
    print(f"std:   {float(arr.std()):.2f}")

    for p in [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]:
        print(f"p{p:<4}: {float(np.percentile(arr, p)):.1f}")

    print("-" * 80)
    print("Threshold coverage:")
    for th in thresholds:
        n_over = int((arr > th).sum())
        ratio_over = n_over / max(len(arr), 1)
        print(f"> {th:5d}: {n_over:6d} samples | {ratio_over * 100:6.3f}%")

    print("-" * 80)
    print("Top 20 longest prompts:")
    for r in sorted(records, key=lambda x: x["length"], reverse=True)[:20]:
        print(
            f"len={r['length']:5d} | qid={r['question_id']} | "
            f"answer={r['answer']} | video={r['video_stem']} | q={r['question']}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()