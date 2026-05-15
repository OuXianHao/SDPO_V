#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

# 避免 vLLM 多进程 fork 后 CUDA re-init 报错
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


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

VALID_OPTION_SET = {"A", "B", "C", "D", "E", "F"}
LETTERS = "ABCDEF"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def normalize_response(response: str) -> str:
    if response is None:
        return ""
    response = str(response).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s*<\s*", "<", response)
    response = re.sub(r"\s*>\s*", ">", response)
    response = re.sub(r"\s*/\s*", "/", response)
    return response.strip()


def format_score(response: str) -> float:
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

    answer_content = match.group(2).strip().upper()
    if answer_content not in VALID_OPTION_SET:
        return 0.0

    return 1.0


def extract_pred_letter(response: str) -> Optional[str]:
    response = normalize_response(response)

    if format_score(response) == 0.0:
        return None

    answers = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if not answers:
        return None

    pred = answers[-1].strip().upper()
    if pred in VALID_OPTION_SET:
        return pred
    return None


def accuracy_score(response: str, ground_truth: str) -> float:
    pred = extract_pred_letter(response)
    if pred is None:
        return 0.0
    return 1.0 if pred == ground_truth else 0.0


def build_mvbench_content(question: str, candidates: List[str]) -> str:
    lines = [
        "<video>",
        f"Question: {question}",
        "",
        "Options:",
    ]
    for i, cand in enumerate(candidates):
        lines.append(f"{LETTERS[i]}. {cand}")
    return "\n".join(lines)


def build_prompt(question: str, candidates: List[str]) -> str:
    content = build_mvbench_content(question, candidates)
    return PROMPT_TEMPLATE.format(content=content.strip())


def get_gold_letter(item: Dict[str, Any]) -> str:
    if "ground_truth" in item and str(item["ground_truth"]).strip().upper() in VALID_OPTION_SET:
        return str(item["ground_truth"]).strip().upper()

    candidates = item["candidates"]
    answer = item["answer"]
    if answer not in candidates:
        raise ValueError(f"answer not in candidates: answer={answer}, candidates={candidates}")

    idx = candidates.index(answer)
    if idx >= len(LETTERS):
        raise ValueError(f"too many candidates: {len(candidates)}")

    return LETTERS[idx]


def shell_quote_path(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def frames_to_mp4(frame_dir: str, cache_dir: str, sample_id: str, fps: int = 3) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{sample_id}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        imgs.extend(glob.glob(os.path.join(frame_dir, ext)))
    imgs = sorted(imgs)

    if not imgs:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for img in imgs:
            f.write(f"file {shell_quote_path(os.path.abspath(img))}\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-r",
            str(fps),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def prepare_media_path(item: Dict[str, Any], args: argparse.Namespace) -> str:
    media_type = item.get("media_type", "video")
    media_path = item["media_path"]

    if media_type == "video":
        return media_path

    if media_type == "frames":
        # MVBench episodic_reasoning 是图片帧目录，直接返回目录，不再 ffmpeg 转 mp4
        return media_path

    raise ValueError(f"unknown media_type={media_type}")

def sample_frame_dir_images(frame_dir: str, num_frames: int, fps: float = 3.0) -> tuple:
    """
    Sample fixed num_frames from an image-frame directory.
    Return:
      frames_np: [T, H, W, 3], RGB uint8
      metadata: dict for vLLM Qwen3-VL VideoMetadata
    """
    import cv2
    import numpy as np

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        imgs.extend(glob.glob(os.path.join(frame_dir, ext)))
    imgs = sorted(imgs)

    if not imgs:
        raise RuntimeError(f"No image frames found in frame dir: {frame_dir}")

    total = len(imgs)
    indices = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int).tolist()

    frames = []
    for idx in indices:
        img_path = imgs[idx]
        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            if frames:
                frames.append(frames[-1])
                continue
            raise RuntimeError(f"Failed to read image frame: {img_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    if len(frames) < num_frames:
        last = frames[-1]
        last_idx = indices[-1] if indices else total - 1
        while len(frames) < num_frames:
            frames.append(last)
            indices.append(last_idx)

    frames_np = np.stack(frames, axis=0).astype("uint8")

    fps = float(fps) if fps and fps > 0 else 3.0
    duration = float(total / fps)

    metadata = {
        "fps": fps,
        "frames_indices": indices,
        "total_num_frames": total,
        "duration": duration,
    }

    return frames_np, metadata
def sample_video_frames_cv2(video_path: str, num_frames: int) -> tuple:
    """
    Return:
      frames: np.ndarray, shape [T, H, W, 3], RGB uint8
      metadata: dict for vLLM Qwen3-VL VideoMetadata
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total <= 0:
        # fallback: sequentially read all frames
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        cap.release()

        if not all_frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        total = len(all_frames)
        fps = fps if fps > 0 else 1.0
        indices = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int).tolist()
        frames = [all_frames[i] for i in indices]
    else:
        fps = fps if fps > 0 else 1.0
        indices = np.linspace(0, total - 1, min(num_frames, total)).round().astype(int).tolist()

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                # fallback: use last good frame if possible
                if frames:
                    frames.append(frames[-1])
                    continue
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

    # 如果视频帧数少于 num_frames，补最后一帧，保持固定 32 帧
    if len(frames) < num_frames:
        last = frames[-1]
        while len(frames) < num_frames:
            frames.append(last)
        if indices:
            last_idx = indices[-1]
        else:
            last_idx = total - 1
        while len(indices) < num_frames:
            indices.append(last_idx)

    frames_np = np.stack(frames, axis=0).astype("uint8")

    duration = float(total / fps) if fps > 0 else float(total)
    metadata = {
        "fps": fps,
        "frames_indices": indices,
        "total_num_frames": total,
        "duration": duration,
    }

    return frames_np, metadata


def build_video_mm_data(video_path: str, args: argparse.Namespace):
    if os.path.isdir(video_path):
        frames, metadata = sample_frame_dir_images(
            video_path,
            args.num_frames,
            fps=args.frame_fps,
        )
    else:
        frames, metadata = sample_video_frames_cv2(video_path, args.num_frames)
    return (frames, metadata)

def build_vllm_input(
    item: Dict[str, Any],
    media_path: str,
    processor: AutoProcessor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    vLLM offline video input:
    - 不传裸字符串路径，否则 parser 会把 path 当 iterable，报 got: '/'
    - 不走 qwen_vl_utils，避免 video_metadata=None
    - 自己用 cv2 抽 32 帧，并传 (frames_np, metadata)
    """
    prompt_text = build_prompt(item["question"], item["candidates"])

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": media_path,
                    "num_frames": args.num_frames,
                    "fps": args.video_fps,
                    "max_pixels": args.max_pixels,
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    video_data = build_video_mm_data(media_path, args)

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "video": video_data,
        },
        "mm_processor_kwargs": {
            "num_frames": args.num_frames,
            "fps": args.video_fps,
            "max_pixels": args.max_pixels,
        },
    }


def summarize(results: List[Dict[str, Any]], format_weight: float = 0.3) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    total_format = sum(r["format_score"] for r in results)
    total_acc = sum(r["accuracy_score"] for r in results)
    total_overall = sum(r["overall_score"] for r in results)

    by_task = defaultdict(
        lambda: {
            "count": 0,
            "format_sum": 0.0,
            "accuracy_sum": 0.0,
            "overall_sum": 0.0,
            "bad_format": 0,
            "empty_response": 0,
        }
    )

    for r in results:
        t = r["task"]
        by_task[t]["count"] += 1
        by_task[t]["format_sum"] += r["format_score"]
        by_task[t]["accuracy_sum"] += r["accuracy_score"]
        by_task[t]["overall_sum"] += r["overall_score"]
        by_task[t]["bad_format"] += int(r["format_score"] == 0.0)
        by_task[t]["empty_response"] += int(not str(r.get("response", "")).strip())

    per_task = {}
    for t, s in sorted(by_task.items()):
        c = s["count"]
        per_task[t] = {
            "count": c,
            "format": s["format_sum"] / c,
            "accuracy": s["accuracy_sum"] / c,
            "overall": s["overall_sum"] / c,
            "bad_format": s["bad_format"],
            "empty_response": s["empty_response"],
        }

    return {
        "count": n,
        "format_weight": format_weight,
        "format": total_format / n,
        "accuracy": total_acc / n,
        "overall": total_overall / n,
        "bad_format": sum(int(r["format_score"] == 0.0) for r in results),
        "empty_response": sum(int(not str(r.get("response", "")).strip()) for r in results),
        "per_task": per_task,
    }


def load_done_ids(output_jsonl: str) -> set:
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    x = json.loads(line)
                    done.add(x["sample_id"])
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Merged HF model path")
    parser.add_argument(
        "--manifest",
        default="/data/xhou/datasets/MVBench/processed/mvbench_3800_manifest.jsonl",
    )
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--summary-output", default=None, help="Output summary json path")

    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)

    # 默认用确定性解码；如果你想和 LongVideoBench 的 t=0.6/p=0.9 对齐，可以命令行覆盖
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--format-weight", type=float, default=0.3)

    # 固定 32 帧
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--frame-fps", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=524288)
    parser.add_argument(
        "--frame-video-cache",
        default="/data/xhou/datasets/MVBench/processed/frame_videos_cache",
    )

    # 兼容开关：有些 vLLM 版本 Qwen3-VL V1 视频 metadata 有坑，可以命令行前 export VLLM_USE_V1=0
    parser.add_argument("--disable-custom-all-reduce", action="store_true")

    args = parser.parse_args()

    if args.summary_output is None:
        args.summary_output = args.output.replace(".jsonl", "_summary.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = read_jsonl(args.manifest)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    done_ids = load_done_ids(args.output) if args.resume else set()
    if done_ids:
        print(f"[resume] loaded done ids: {len(done_ids)}")

    pending = [x for x in data if x["sample_id"] not in done_ids]

    print(f"total manifest: {len(data)}")
    print(f"pending: {len(pending)}")
    print(f"output: {args.output}")
    print(f"summary: {args.summary_output}")
    print(f"num_frames: {args.num_frames}")
    print(f"video_fps: {args.video_fps}")
    print(f"max_pixels: {args.max_pixels}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"video": 1, "image": 0},
    )

    if args.disable_custom_all_reduce:
        llm_kwargs["disable_custom_all_reduce"] = True

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for start in tqdm(range(0, len(pending), args.batch_size), desc="Evaluating MVBench"):
        batch_items = pending[start : start + args.batch_size]

        requests = []
        prepared_items = []

        for item in batch_items:
            try:
                media_path = prepare_media_path(item, args)
                request = build_vllm_input(item, media_path, processor, args)
                requests.append(request)
                prepared_items.append((item, media_path))
            except Exception as e:
                gold = get_gold_letter(item)
                record = {
                    "sample_id": item.get("sample_id"),
                    "task": item.get("task"),
                    "question": item.get("question"),
                    "candidates": item.get("candidates"),
                    "answer": item.get("answer"),
                    "ground_truth": gold,
                    "media_path": item.get("media_path"),
                    "media_type": item.get("media_type"),
                    "prepared_media_path": None,
                    "response": "",
                    "pred": None,
                    "format_score": 0.0,
                    "accuracy_score": 0.0,
                    "overall_score": 0.0,
                    "error": repr(e),
                }
                append_jsonl(args.output, record)

        if not requests:
            continue

        try:
            outputs = llm.generate(requests, sampling_params=sampling_params)
        except Exception:
            # 这里不吞异常，方便你看到 vLLM 原始报错
            raise

        for (item, media_path), out in zip(prepared_items, outputs):
            response = out.outputs[0].text if out.outputs else ""
            gold = get_gold_letter(item)
            fmt = format_score(response)
            acc = accuracy_score(response, gold)
            overall = (1 - args.format_weight) * acc + args.format_weight * fmt
            pred = extract_pred_letter(response)

            record = {
                "sample_id": item["sample_id"],
                "task": item["task"],
                "question": item["question"],
                "candidates": item["candidates"],
                "answer": item["answer"],
                "ground_truth": gold,
                "media_path": item["media_path"],
                "media_type": item["media_type"],
                "prepared_media_path": media_path,
                "response": response,
                "pred": pred,
                "format_score": fmt,
                "accuracy_score": acc,
                "overall_score": overall,
                "error": None,
            }
            append_jsonl(args.output, record)

    final_results = read_jsonl(args.output)
    summary = summarize(final_results, format_weight=args.format_weight)
    write_json(args.summary_output, summary)

    print("=" * 80)
    print("MVBench evaluation finished")
    print("count:", summary.get("count"))
    print("format:", summary.get("format"))
    print("accuracy:", summary.get("accuracy"))
    print("overall:", summary.get("overall"))
    print("bad_format:", summary.get("bad_format"))
    print("empty_response:", summary.get("empty_response"))
    print("summary saved to:", args.summary_output)
    print("=" * 80)

    print("\nPer-task accuracy:")
    for task, s in summary.get("per_task", {}).items():
        print(
            f"{task:30s} "
            f"count={s['count']:4d} "
            f"acc={s['accuracy']:.4f} "
            f"fmt={s['format']:.4f} "
            f"bad_format={s['bad_format']:4d}"
        )


if __name__ == "__main__":
    main()
