#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Qwen3-VL style vLLM model on TempCompass multi-choice split.

Data:
  /data/xhou/datasets/TempCompass/multi-choice/test-00000-of-00001.parquet
  /data/xhou/datasets/TempCompass/videos/{video_id}.mp4

Output:
  JSONL with per-sample predictions
  JSON summary with overall and per-dim accuracy / format rate

Example:
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_tempcompass_qwen3vl_vllm.py \
  --model /data/xhou/checkpoints/qwen3vl8b_newdata_frames32_pure_dapo_run1_step115/global_step_230/actor/huggingface \
  --data-root /data/xhou/datasets/TempCompass \
  --output /data/xhou/eval_results/tempcompass_multichoice_pure_dapo_step230_test10.jsonl \
  --summary-output /data/xhou/eval_results/tempcompass_multichoice_pure_dapo_step230_test10_summary.json \
  --num-frames 64 \
  --max-samples 10 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --batch-size 1 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-tokens 256 \
  --format-weight 0.3
"""

import argparse
import json
import os
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================
# R1V prompt template
# =========================

R1V_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Respond concisely. Your thinking should be brief and focused — identify the core logic, skip trivial steps, and avoid verbose or redundant thinking.
Place the reasoning before the final answer, enclosed in <thinking> and </thinking> tags.

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Inside the <answer> tag, output only one uppercase letter corresponding to the correct option, for example A, B, C, D, E, or F.
Do not output any text outside the <thinking> and <answer> tags.
Do not output any text after </answer>.

User: {content}
Assistant:"""


# =========================
# R1V reward-compatible parser
# copied/adapted from r1v.py logic
# =========================

_VALID_OPTION_SET = {"A", "B", "C", "D", "E", "F"}


def normalize_response(response: str) -> str:
    if response is None:
        return ""
    response = str(response).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s*<\s*", "<", response)
    response = re.sub(r"\s*>\s*", ">", response)
    response = re.sub(r"\s*/\s*", "/", response)
    return response.strip()


def extract_all_answer_contents(response: str) -> list[str]:
    if not response:
        return []
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return [m.strip() for m in matches if m and m.strip()]


def extract_answer_content(response: str) -> Optional[str]:
    answers = extract_all_answer_contents(response)
    if not answers:
        return None
    return answers[-1]


def extract_option_letter_strict(text: str) -> Optional[str]:
    if not text:
        return None
    ch = str(text).strip().upper()
    if ch in _VALID_OPTION_SET:
        return ch
    return None


def format_reward(response: str) -> float:
    response = normalize_response(response)

    thinking_open = len(re.findall(r"<thinking>", response, flags=re.IGNORECASE))
    thinking_close = len(re.findall(r"</thinking>", response, flags=re.IGNORECASE))
    answer_open = len(re.findall(r"<answer>", response, flags=re.IGNORECASE))
    answer_close = len(re.findall(r"</answer>", response, flags=re.IGNORECASE))

    if thinking_open != 1 or thinking_close != 1:
        return 0.0
    if answer_open != 1 or answer_close != 1:
        return 0.0

    match = re.search(
        r"^\s*<thinking>(.*?)</thinking>\s*<answer>(.*?)</answer>\s*$",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return 0.0

    answer_content = match.group(2).strip()
    if not answer_content:
        return 0.0

    if extract_option_letter_strict(answer_content) is None:
        return 0.0

    return 1.0


def accuracy_reward(response: str, ground_truth_letter: str) -> float:
    response = normalize_response(response)
    gt = str(ground_truth_letter).strip().upper()

    if format_reward(response) == 0.0:
        return 0.0

    given_answer = extract_answer_content(response)
    pred = extract_option_letter_strict(given_answer) if given_answer is not None else None

    if gt not in _VALID_OPTION_SET:
        return 0.0

    return 1.0 if pred == gt else 0.0


def compute_score(response: str, ground_truth_letter: str, format_weight: float = 0.3) -> dict[str, float]:
    fmt = format_reward(response)
    acc = accuracy_reward(response, ground_truth_letter)
    return {
        "overall": (1.0 - format_weight) * acc + format_weight * fmt,
        "format": fmt,
        "accuracy": acc,
    }


def extract_pred_letter_for_logging(response: str) -> Optional[str]:
    """
    Strict prediction extraction for logging.
    Only returns a letter when the last <answer>...</answer> contains exactly one A-F.
    """
    response = normalize_response(response)
    ans = extract_answer_content(response)
    return extract_option_letter_strict(ans) if ans is not None else None


def extract_gt_letter_from_tempcompass_answer(answer: Any) -> str:
    """
    TempCompass multi-choice answer is like:
      A. dunking a basketball
      B. moving from left to right
    Convert it to a single gold letter.
    """
    s = str(answer).strip()
    if not s:
        return ""
    ch = s[0].upper()
    return ch if ch in _VALID_OPTION_SET else ""


# =========================
# Video loading
# =========================

def sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int64)
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
    # If video has fewer frames, repeat indices to requested num_frames.
    return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)


def read_video_frames(video_path: Path, num_frames: int):
    """
    Read fixed number of frames and return (frames, metadata) for Qwen3-VL vLLM.

    Qwen3-VL in vLLM expects each video item to carry metadata.
    """
    video_path = Path(video_path)

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)

        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = 1.0

        inds = sample_indices(total, num_frames)
        frames = vr.get_batch(inds).asnumpy()  # [T, H, W, C], RGB uint8

        duration = float(total / fps) if fps and fps > 0 else 0.0

        metadata = VideoMetadata(
            total_num_frames=int(total),
            fps=float(fps),
            duration=float(duration),
            video_backend="decord",
            frames_indices=[int(x) for x in inds.tolist()],
        )

        return frames, metadata

    except Exception as e_decord:
        try:
            import torch
            import torchvision

            video, _, info = torchvision.io.read_video(
                str(video_path),
                pts_unit="sec",
                output_format="TCHW",
            )

            total = int(video.shape[0])
            fps = float(info.get("video_fps", 1.0) or 1.0)

            inds = sample_indices(total, num_frames)
            frames = video[torch.as_tensor(inds)].permute(0, 2, 3, 1).cpu().numpy()
            frames = frames.astype(np.uint8)

            duration = float(total / fps) if fps and fps > 0 else 0.0

            metadata = VideoMetadata(
                total_num_frames=int(total),
                fps=float(fps),
                duration=float(duration),
                video_backend="torchvision",
                frames_indices=[int(x) for x in inds.tolist()],
            )

            return frames, metadata

        except Exception as e_torchvision:
            raise RuntimeError(
                f"Failed to read video: {video_path}\n"
                f"decord error: {repr(e_decord)}\n"
                f"torchvision error: {repr(e_torchvision)}"
            )

# =========================
# Prompt / inputs
# =========================
def _get_image_patch_size(processor) -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", 14)

    if isinstance(patch_size, dict):
        patch_size = patch_size.get("height", patch_size.get("width", 14))

    try:
        return int(patch_size)
    except Exception:
        return 14



def build_vllm_input_with_processor(
    processor,
    question: str,
    video_path: Path,
    num_frames: int,
) -> dict[str, Any]:
    """
    Correct Qwen3-VL/vLLM path:
      messages
      -> processor.apply_chat_template()
      -> process_vision_info(..., return_video_metadata=True, return_video_kwargs=True)
      -> vLLM input
    """

    system_text = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Respond concisely. Your thinking should be brief and focused — identify the core logic, skip trivial steps, and avoid verbose or redundant thinking.
Place the reasoning before the final answer, enclosed in <thinking> and </thinking> tags.

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Inside the <answer> tag, output only one uppercase letter corresponding to the correct option, for example A, B, C, D, E, or F.
Do not output any text outside the <thinking> and <answer> tags.
Do not output any text after </answer>."""

    messages = [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": int(num_frames),
                },
                {
                    "type": "text",
                    "text": str(question).strip(),
                },
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    patch_size = _get_image_patch_size(processor)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=patch_size,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    if video_kwargs is None:
        video_kwargs = {}

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
# =========================
# Summary
# =========================

def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    accs = [float(r.get("accuracy_score", 0.0)) for r in records]
    fmts = [float(r.get("format_score", 0.0)) for r in records]
    overs = [float(r.get("overall_score", 0.0)) for r in records]

    summary: dict[str, Any] = {
        "count": n,
        "accuracy": safe_mean(accs),
        "format_rate": safe_mean(fmts),
        "overall_mean": safe_mean(overs),
        "correct": int(sum(accs)),
        "format_correct": int(sum(fmts)),
        "answer_distribution": dict(Counter(r.get("gt_letter", "") for r in records)),
        "pred_distribution": dict(Counter(str(r.get("pred_letter")) for r in records)),
        "by_dim": {},
    }

    by_dim_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dim_records[str(r.get("dim", "unknown"))].append(r)

    for dim, rs in sorted(by_dim_records.items()):
        d_accs = [float(r.get("accuracy_score", 0.0)) for r in rs]
        d_fmts = [float(r.get("format_score", 0.0)) for r in rs]
        d_overs = [float(r.get("overall_score", 0.0)) for r in rs]
        summary["by_dim"][dim] = {
            "count": len(rs),
            "accuracy": safe_mean(d_accs),
            "format_rate": safe_mean(d_fmts),
            "overall_mean": safe_mean(d_overs),
            "correct": int(sum(d_accs)),
            "format_correct": int(sum(d_fmts)),
        }

    return summary


def load_existing_records(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    records = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def write_summary(summary_path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    summary["args"] = vars(args)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Updated summary: {summary_path}")
    print(
        f"accuracy={summary['accuracy']:.4f}, "
        f"format_rate={summary['format_rate']:.4f}, "
        f"overall_mean={summary['overall_mean']:.4f}, "
        f"count={summary['count']}"
    )


# =========================
# Main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="/data/xhou/datasets/TempCompass")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary-output", type=str, required=True)

    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)

    parser.add_argument("--format-weight", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    data_file = data_root / "multi-choice" / "test-00000-of-00001.parquet"
    video_root = data_root / "videos"

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_file}")
    df = pd.read_parquet(data_file)

    # Add stable sample index.
    df = df.reset_index(drop=True)
    df["sample_index"] = list(range(len(df)))

    if args.start_index > 0:
        df = df[df["sample_index"] >= args.start_index]

    if args.max_samples is not None:
        df = df.head(args.max_samples)

    print(f"Total eval samples after slicing: {len(df)}")
    print("Dim distribution:")
    print(df["dim"].value_counts())

    existing_records: list[dict[str, Any]] = []
    done_indices = set()

    if args.resume and output_path.exists():
        existing_records = load_existing_records(output_path)
        done_indices = {int(r["sample_index"]) for r in existing_records if "sample_index" in r}
        print(f"Resume enabled. Existing records: {len(existing_records)}")
        print(f"Done sample indices: {len(done_indices)}")

    pending_rows = []
    for _, row in df.iterrows():
        idx = int(row["sample_index"])
        if idx in done_indices:
            continue
        pending_rows.append(row)

    print(f"Pending samples: {len(pending_rows)}")

    if not pending_rows:
        write_summary(summary_path, existing_records, args)
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(f"Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"video": 1},
    )

    all_records = list(existing_records)

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as fout:
        for batch_start in tqdm(range(0, len(pending_rows), args.batch_size), desc="Evaluating TempCompass"):
            batch_rows = pending_rows[batch_start: batch_start + args.batch_size]

            vllm_inputs = []
            meta = []

            for row in batch_rows:
                sample_index = int(row["sample_index"])
                video_id = str(row["video_id"])
                question = str(row["question"]).strip()
                raw_answer = str(row["answer"]).strip()
                gt_letter = extract_gt_letter_from_tempcompass_answer(raw_answer)
                dim = str(row["dim"])

                video_path = video_root / f"{video_id}.mp4"

                record_base = {
                    "sample_index": sample_index,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "question": question,
                    "raw_answer": raw_answer,
                    "gt_letter": gt_letter,
                    "dim": dim,
                }

                if not video_path.exists():
                    record = {
                        **record_base,
                        "error": f"missing video: {video_path}",
                        "response": "",
                        "pred_letter": None,
                        "format_score": 0.0,
                        "accuracy_score": 0.0,
                        "overall_score": 0.0,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    all_records.append(record)
                    continue

                if gt_letter not in _VALID_OPTION_SET:
                    record = {
                        **record_base,
                        "error": f"invalid gt answer: {raw_answer}",
                        "response": "",
                        "pred_letter": None,
                        "format_score": 0.0,
                        "accuracy_score": 0.0,
                        "overall_score": 0.0,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    all_records.append(record)
                    continue

                try:
                    vllm_inputs.append(
                        build_vllm_input_with_processor(
                            processor=processor,
                            question=question,
                            video_path=video_path,
                            num_frames=args.num_frames,
                        )
                    )
                    meta.append(record_base)
                except Exception as e:
                    record = {
                        **record_base,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                        "response": "",
                        "pred_letter": None,
                        "format_score": 0.0,
                        "accuracy_score": 0.0,
                        "overall_score": 0.0,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    all_records.append(record)

            if not vllm_inputs:
                write_summary(summary_path, all_records, args)
                continue

            # try:
            #     outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            # except Exception as e:
            #     # If one batch crashes, record errors for the whole batch and continue.
            #     err = repr(e)
            #     tb = traceback.format_exc()
            #     for m in meta:
            #         record = {
            #             **m,
            #             "error": err,
            #             "traceback": tb,
            #             "response": "",
            #             "pred_letter": None,
            #             "format_score": 0.0,
            #             "accuracy_score": 0.0,
            #             "overall_score": 0.0,
            #         }
            #         fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            #         fout.flush()
            #         all_records.append(record)
            #     write_summary(summary_path, all_records, args)
            #     continue
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            for m, out in zip(meta, outputs):
                response = out.outputs[0].text if out.outputs else ""
                pred_letter = extract_pred_letter_for_logging(response)

                score = compute_score(
                    response=response,
                    ground_truth_letter=m["gt_letter"],
                    format_weight=args.format_weight,
                )

                record = {
                    **m,
                    "response": response,
                    "pred_letter": pred_letter,
                    "format_score": score["format"],
                    "accuracy_score": score["accuracy"],
                    "overall_score": score["overall"],
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                all_records.append(record)

            write_summary(summary_path, all_records, args)

    print("\n===== Final Summary =====")
    write_summary(summary_path, all_records, args)


if __name__ == "__main__":
    main()