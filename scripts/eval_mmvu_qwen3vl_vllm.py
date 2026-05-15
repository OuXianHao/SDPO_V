#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
from decord import VideoReader, cpu
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

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


_VALID_OPTION_SET = {"A", "B", "C", "D", "E", "F"}


def _normalize_response(response: str) -> str:
    if response is None:
        return ""
    response = str(response).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s*<\s*", "<", response)
    response = re.sub(r"\s*>\s*", ">", response)
    response = re.sub(r"\s*/\s*", "/", response)
    return response.strip()


def _extract_all_answer_contents(response: str) -> list[str]:
    if not response:
        return []
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m and m.strip()]


def _extract_answer_content(response: str) -> str | None:
    answers = _extract_all_answer_contents(response)
    if not answers:
        return None
    return answers[-1]


def _extract_option_letter_strict(text: str) -> Optional[str]:
    if not text:
        return None
    ch = str(text).strip().upper()
    if ch in _VALID_OPTION_SET:
        return ch
    return None


def _extract_gold_option_letter(text: str) -> Optional[str]:
    if not text:
        return None
    text = str(text).strip()

    m = re.search(r"<answer>\s*([A-Fa-f])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    ch = text.strip().upper()
    if ch in _VALID_OPTION_SET:
        return ch

    return None


def format_reward(response: str) -> float:
    response = _normalize_response(response)

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

    if _extract_option_letter_strict(answer_content) is None:
        return 0.0

    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)
    gt_raw = _normalize_response(ground_truth)

    if format_reward(response) == 0.0:
        return 0.0

    given_answer = _extract_answer_content(response)
    pred = _extract_option_letter_strict(given_answer) if given_answer is not None else None
    gt = _extract_gold_option_letter(gt_raw)

    if gt not in _VALID_OPTION_SET:
        return 0.0

    return 1.0 if pred == gt else 0.0


def compute_score(response: str, ground_truth: str, format_weight: float = 0.2) -> dict[str, float]:
    fmt = format_reward(response)
    acc = accuracy_reward(response, ground_truth)
    return {
        "overall": (1 - format_weight) * acc + format_weight * fmt,
        "format": fmt,
        "accuracy": acc,
    }


def parse_choices(choices: Any) -> dict:
    if isinstance(choices, dict):
        return choices

    if isinstance(choices, str):
        s = choices.strip()
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}

def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x

def build_content(row: pd.Series) -> str:
    question = str(row["question"]).strip()
    choices = parse_choices(row["choices"])
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    lines = [video_placeholder, f"Question: {question}", "Choices:"]
    for key in ["A", "B", "C", "D", "E", "F"]:
        val = choices.get(key, None)
        if val is not None and str(val).strip() != "":
            lines.append(f"{key}. {str(val).strip()}")

    return "\n".join(lines).strip()


def build_prompt(row: pd.Series) -> str:
    content = build_content(row)
    return PROMPT_TEMPLATE.format(content=content)
def build_video_inputs(vid: str, num_frames: int, max_pixels: int, processor):
    """
    Qwen3-VL + vLLM offline 必须返回带 metadata 的 video_inputs。
    返回：
      video_inputs: list[(video_tensor, metadata)]
      video_kwargs: dict
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": vid,
                    "num_frames": num_frames,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": "dummy",
                },
            ],
        }
    ]

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is None:
        raise RuntimeError(f"process_vision_info failed to load video: {vid}")

    # 关键：Qwen3-VL 下 video_inputs 的每个 item 应该是 (video_tensor, metadata)
    first = video_inputs[0]
    if not isinstance(first, tuple) or len(first) != 2 or first[1] is None:
        raise RuntimeError(
            f"Qwen3-VL video metadata missing. "
            f"type(video_inputs[0])={type(first)}, value={repr(first)[:300]}"
        )

    # 固定帧数时，避免 fps 和 num_frames 同时冲突
    video_kwargs = dict(video_kwargs or {})
    video_kwargs["do_resize"] = False
    video_kwargs["num_frames"] = num_frames
    video_kwargs["fps"] = None

    return video_inputs, video_kwargs

def resolve_video_path(row_video_path: str, data_root: str, video_root: str | None = None) -> str:
    row_video_path = str(row_video_path).strip()

    # 推荐路径：data_root + parquet 里的 video_path，例如:
    # /data/xhou/datasets/MMVU + videos/Chemistry/0.mp4
    p1 = Path(data_root) / row_video_path
    if p1.exists():
        return str(p1)

    # 如果用户传了 video_root，则去掉开头的 videos/
    if video_root is not None:
        rel = row_video_path
        if rel.startswith("videos/"):
            rel = rel[len("videos/"):]
        p2 = Path(video_root) / rel
        if p2.exists():
            return str(p2)

    # 兜底返回 p1，后面统一报 missing
    return str(p1)

def load_video_fixed_frames(video_path: str, num_frames: int = 64) -> np.ndarray:
    """
    Uniformly sample exactly num_frames frames from a video.

    Return:
        np.ndarray with shape [num_frames, H, W, 3], RGB uint8.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total <= 0:
        raise ValueError(f"empty video: {video_path}")

    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=np.int64)
    else:
        # 视频帧数不足 64 时，重复采样补齐到 64 帧
        indices = np.linspace(0, total - 1, num_frames, dtype=np.int64)

    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3], RGB
    return frames

def load_done_ids(output_path: str) -> set[str]:
    done = set()
    if not output_path or not os.path.exists(output_path):
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    done.add(str(obj["id"]))
            except Exception:
                continue
    return done


def write_summary(records: list[dict], summary_path: str):
    if not records:
        summary = {
            "count": 0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "overall_mean": 0.0,
        }
    else:
        n = len(records)
        acc = sum(float(x["accuracy_score"]) for x in records) / n
        fmt = sum(float(x["format_score"]) for x in records) / n
        overall = sum(float(x["overall_score"]) for x in records) / n
        correct = sum(int(float(x["accuracy_score"]) == 1.0) for x in records)
        valid_fmt = sum(int(float(x["format_score"]) == 1.0) for x in records)

        summary = {
            "count": n,
            "correct": correct,
            "valid_format": valid_fmt,
            "accuracy": acc,
            "format_rate": fmt,
            "overall_mean": overall,
        }

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="/data/xhou/datasets/MMVU/data/validation_multiple_choice.parquet")
    parser.add_argument("--data-root", default="/data/xhou/datasets/MMVU")
    parser.add_argument("--video-root", default=None)

    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--format-weight", type=float, default=0.2)

    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=64)

    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if "question_type" in df.columns:
        df = df[df["question_type"] == "multiple-choice"].reset_index(drop=True)

    if args.max_samples is not None:
        df = df.head(args.max_samples).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    done_ids = load_done_ids(args.output) if args.resume else set()
    print(f"Loaded data: {args.data}")
    print(f"Total samples after filter: {len(df)}")
    print(f"Resume done ids: {len(done_ids)}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs={
            "do_resize": False,
        },

        # Long-video multimodal 更稳：关闭容易触发 masked scatter 对齐问题的优化
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=None,
    )

    all_new_records = []

    rows = [row for _, row in df.iterrows() if str(row["id"]) not in done_ids]

    with open(args.output, "a" if args.resume else "w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="Evaluating MMVU"):
            batch_rows = rows[start:start + args.batch_size]

            requests = []
            meta = []

            for row in batch_rows:
                vid = resolve_video_path(
                    row_video_path=row["video_path"],
                    data_root=args.data_root,
                    video_root=args.video_root,
                )

                if not os.path.exists(vid):
                    record = {
                        "id": str(row["id"]),
                        "error": f"missing_video: {vid}",
                        "video_path": str(row["video_path"]),
                        "resolved_video_path": vid,
                        "answer": str(row["answer"]),
                        "prediction": None,
                        "response": "",
                        "format_score": 0.0,
                        "accuracy_score": 0.0,
                        "overall_score": 0.0,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    all_new_records.append(record)
                    continue

                prompt = build_prompt(row)
                
                video_inputs, video_kwargs = build_video_inputs(
                    vid=vid,
                    num_frames=args.num_frames,
                    max_pixels=args.max_pixels,
                    processor=processor,
                )

                requests.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "video": video_inputs,
                    },
                    "mm_processor_kwargs": video_kwargs,
                })

                meta.append({
                    "row": row,
                    "prompt": prompt,
                    "resolved_video_path": vid,
                })

            if not requests:
                continue

            outputs = llm.generate(requests, sampling_params)

            for m, out in zip(meta, outputs):
                row = m["row"]
                response = out.outputs[0].text if out.outputs else ""
                score = compute_score(response, str(row["answer"]), format_weight=args.format_weight)

                answer_content = _extract_answer_content(_normalize_response(response))
                pred = _extract_option_letter_strict(answer_content) if answer_content is not None else None

                record = {
                    "id": str(row["id"]),
                    "question_type": str(row.get("question_type", "")),
                    "question": str(row["question"]),
                    "choices": to_jsonable(parse_choices(row["choices"])),
                    "answer": str(row["answer"]),
                    "prediction": pred,
                    "video_path": str(row["video_path"]),
                    "resolved_video_path": m["resolved_video_path"],
                    "response": response,
                    "format_score": score["format"],
                    "accuracy_score": score["accuracy"],
                    "overall_score": score["overall"],
                }

                # 保存 subject 等 metadata，方便后面按学科统计
                try:
                    metadata = row.get("metadata", None)
                    if isinstance(metadata, dict):
                        metadata = to_jsonable(metadata)
                        record["metadata"] = metadata
                        record["subject"] = metadata.get("subject", None)
                    else:
                        record["metadata"] = str(metadata)
                except Exception:
                    pass

                fout.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
                fout.flush()
                all_new_records.append(record)

    # 重新读完整 output，支持 resume 后汇总历史 + 本次结果
    final_records = []
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_records.append(json.loads(line))

    write_summary(final_records, args.summary_output)


if __name__ == "__main__":
    main()
