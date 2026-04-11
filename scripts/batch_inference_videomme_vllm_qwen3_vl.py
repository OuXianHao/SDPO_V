import argparse
import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from qwen_vl_utils import process_vision_info


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Respond concisely. Your thinking should be brief and focused — identify the core logic, skip trivial steps, and avoid verbose or redundant thinking. Keep your thinking within 500 words.
Place the reasoning before the final answer, enclosed in <thinking> and </thinking> tags.

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Inside the <answer> tag, output only one uppercase letter corresponding to the correct option, for example A, B, C, D, E, or F.
Do not output any text outside the <thinking> and <answer> tags.
Do not output any text after </answer>.

User: {content}
Assistant:"""

_VALID_OPTION_SET = {"A", "B", "C", "D", "E", "F"}


# ---------- IO / Args ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference on VideoMME template with Qwen3-VL-Instruct + vLLM."
    )

    # Single-model mode
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to one HF/vLLM-loadable model or checkpoint directory.",
    )

    # Multi-checkpoint mode
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=None,
        help="Root directory containing checkpoint subdirs like global_step_50, global_step_100, ...",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default=r"global_step_(\d+)",
        help="Regex pattern for checkpoint subdir names. Must contain one capture group for step.",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Optional cap on number of discovered checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_stride",
        type=int,
        default=1,
        help="Keep every N-th checkpoint after sorting by step. Useful when there are too many checkpoints.",
    )
    parser.add_argument(
        "--reverse_checkpoints",
        action="store_true",
        help="Evaluate checkpoints from largest step to smallest.",
    )

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)

    # Single-model output
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output json path for single-model mode. If omitted, auto-generated under --output_dir.",
    )

    # Multi-model output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store per-checkpoint jsons and a summary.json. Recommended for multi-checkpoint mode.",
    )

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of video cases per batch (each case usually has 3 questions).",
    )
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output json if it already exists.",
    )
    parser.add_argument(
        "--allow_non_900_cases",
        action="store_true",
        help="Skip the strict check that input_json must contain exactly 900 video cases.",
    )
    parser.add_argument(
        "--overwrite_summary",
        action="store_true",
        help="Overwrite summary.json instead of appending/updating existing entries.",
    )
    parser.add_argument(
        "--format_weight",
        type=float,
        default=0.1,
        help="Weight used for overall score: (1-format_weight)*accuracy + format_weight*format.",
    )
    return parser.parse_args()


# ---------- Utility ----------


def ensure_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def ensure_dir(path: str, desc: str) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{desc} not found: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_videomme_list(data: Any) -> bool:
    return isinstance(data, list)


def normalize_basename(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def checkpoint_tag(path: str) -> str:
    name = normalize_basename(path)
    m = re.search(r"global_step_(\d+)", name)
    if m:
        return f"step_{m.group(1)}"
    return name


def discover_checkpoints(
    checkpoint_root: str,
    pattern: str,
    stride: int = 1,
    reverse: bool = False,
    max_checkpoints: Optional[int] = None,
) -> List[str]:
    ensure_dir(checkpoint_root, "Checkpoint root")
    prog = re.compile(pattern)

    items: List[Tuple[int, str]] = []
    for name in os.listdir(checkpoint_root):
        full = os.path.join(checkpoint_root, name)
        if not os.path.isdir(full):
            continue
        match = prog.fullmatch(name)
        if match:
            try:
                step = int(match.group(1))
            except Exception as exc:
                raise ValueError(
                    f"checkpoint_pattern must contain an integer capture group, got name={name}, pattern={pattern}"
                ) from exc
            items.append((step, full))

    if not items:
        raise ValueError(
            f"No checkpoint directories matched under {checkpoint_root} with pattern: {pattern}"
        )

    items.sort(key=lambda x: x[0], reverse=reverse)

    if stride > 1:
        items = items[::stride]
    if max_checkpoints is not None:
        items = items[:max_checkpoints]

    return [p for _, p in items]


def resolve_model_paths(args: argparse.Namespace) -> List[str]:
    if args.model_path and args.checkpoint_root:
        raise ValueError("Please provide either --model_path or --checkpoint_root, not both.")
    if not args.model_path and not args.checkpoint_root:
        raise ValueError("You must provide one of --model_path or --checkpoint_root.")

    if args.model_path:
        return [args.model_path]

    return discover_checkpoints(
        checkpoint_root=args.checkpoint_root,
        pattern=args.checkpoint_pattern,
        stride=max(1, args.checkpoint_stride),
        reverse=args.reverse_checkpoints,
        max_checkpoints=args.max_checkpoints,
    )


def resolve_output_json(args: argparse.Namespace, model_path: str) -> str:
    if args.output_json and args.model_path and not args.checkpoint_root:
        return args.output_json

    output_dir = args.output_dir
    if output_dir is None:
        if args.output_json is not None:
            return args.output_json
        output_dir = os.path.join(os.getcwd(), "videomme_eval_outputs")

    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"videomme_{checkpoint_tag(model_path)}.json")


def resolve_summary_json(args: argparse.Namespace) -> Optional[str]:
    if args.output_dir is None:
        if args.output_json and args.model_path and not args.checkpoint_root:
            return os.path.join(os.path.dirname(args.output_json) or ".", "summary.json")
        return None
    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.join(args.output_dir, "summary.json")


# ---------- Prompt / Answer ----------


def build_user_content(question_text: str, options: List[str]) -> str:
    options_text = "\n".join(options)
    return f"Question: {question_text.strip()}\n\nOptions:\n{options_text}"


def format_question_prompt(question: str, options: List[str]) -> str:
    if not options:
        raise ValueError("Question options are empty.")
    user_content = build_user_content(question, options)
    return PROMPT_TEMPLATE.format(content=user_content.strip())


# ---------- R1V-style robust scoring ----------


def _normalize_response(response: str) -> str:
    if response is None:
        return ""
    response = str(response).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s*<\s*", "<", response)
    response = re.sub(r"\s*>\s*", ">", response)
    response = re.sub(r"\s*/\s*", "/", response)
    return response.strip()


def _extract_all_answer_contents(response: str) -> List[str]:
    if not response:
        return []
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m and m.strip()]


def _extract_answer_content(response: str) -> Optional[str]:
    answers = _extract_all_answer_contents(response)
    if not answers:
        return None
    return answers[-1]


def _extract_option_letter_strict(text: str) -> Optional[str]:
    """Strictly validate that *text* is exactly one option letter A-F."""
    if not text:
        return None
    ch = str(text).strip().upper()
    if ch in _VALID_OPTION_SET:
        return ch
    return None


def _extract_gold_option_letter(text: str) -> Optional[str]:
    """Parse an option letter from a gold / ground-truth string."""
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


def format_reward_eval(response: str) -> float:
    response = _normalize_response(response)

    open_count = len(re.findall(r"<answer>", response, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</answer>", response, flags=re.IGNORECASE))

    if open_count != 1 or close_count != 1:
        return 0.0

    match = re.search(r"^(.*)<answer>(.*?)</answer>\s*$", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return 0.0

    answer_content = match.group(2).strip()
    if not answer_content:
        return 0.0

    # Answer content must be exactly one valid option letter A-F
    if _extract_option_letter_strict(answer_content) is None:
        return 0.0

    return 1.0


def accuracy_reward_eval(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)
    gt_raw = _normalize_response(ground_truth)

    if format_reward_eval(response) == 0.0:
        return 0.0

    # Parse prediction (strict: must be exactly one letter A-F)
    given_answer = _extract_answer_content(response)
    pred = _extract_option_letter_strict(given_answer) if given_answer is not None else None

    # Parse ground truth (slightly more lenient for dataset formats)
    gt = _extract_gold_option_letter(gt_raw)

    if gt not in _VALID_OPTION_SET:
        return 0.0

    return 1.0 if pred == gt else 0.0


def compute_score_eval(response: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
    response = _normalize_response(response)
    format_score = format_reward_eval(response)
    accuracy_score = accuracy_reward_eval(response, ground_truth)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


def extract_answer_letter(text: str) -> str:
    final_answer_content = _extract_answer_content(text)
    pred = _extract_option_letter_strict(final_answer_content) if final_answer_content is not None else None
    return pred or ""


# ---------- VideoMME data ----------


def prepare_case_video_cache(
    video_path: str, num_frames: int, processor: AutoProcessor
) -> Tuple[List[Any], Dict]:
    probe_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": num_frames},
                {"type": "text", "text": " "},
            ],
        }
    ]
    _, video_inputs, video_kwargs = process_vision_info(
        probe_messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if not video_inputs:
        raise RuntimeError(f"Failed to process video inputs for: {video_path}")
    return video_inputs, video_kwargs


def build_vllm_input_with_cached_video(
    prompt: str,
    video_path: str,
    cached_video_inputs: List[Any],
    cached_video_kwargs: Dict,
    num_frames: int,
    processor: AutoProcessor,
) -> Dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": num_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": text,
        "multi_modal_data": {"video": list(cached_video_inputs)},
        "mm_processor_kwargs": dict(cached_video_kwargs or {}),
    }


def build_todo_items(
    dataset: List[Dict],
    video_dir: str,
) -> List[Tuple[int, str, List[Tuple[int, str]]]]:
    todo = []
    for case_idx, case in enumerate(dataset):
        video_name = case.get("video_name")
        if not video_name:
            raise ValueError(f"Missing video_name at case index {case_idx}")

        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        questions = case.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"Invalid questions at case index {case_idx}")
        if len(questions) != 3:
            raise ValueError(
                f"Expected 3 questions per case, got {len(questions)} at case index {case_idx}"
            )

        todo_questions = []
        for q_idx, q in enumerate(questions):
            existing_resp = str(q.get("response", "")).strip()
            if existing_resp:
                continue

            question = str(q.get("question", "")).strip()
            options = q.get("options", [])
            if not question:
                raise ValueError(f"Empty question text at case {case_idx}, q {q_idx}")
            if not isinstance(options, list):
                raise ValueError(f"Invalid options at case {case_idx}, q {q_idx}")

            prompt = format_question_prompt(question, options)
            todo_questions.append((q_idx, prompt))

        if todo_questions:
            todo.append((case_idx, video_path, todo_questions))
    return todo


# ---------- Metrics ----------


def get_gold_answer(question_item: Dict[str, Any]) -> str:
    candidate_keys = [
        "answer",
        "gt_answer",
        "correct_answer",
        "gold_answer",
        "label",
        "response_gt",
    ]
    for key in candidate_keys:
        if key in question_item:
            value = str(question_item[key]).strip()
            if value:
                parsed = _extract_gold_option_letter(value)
                if parsed:
                    return parsed
    return ""


def compute_accuracy(dataset: List[Dict[str, Any]], format_weight: float = 0.1) -> Dict[str, Any]:
    total = 0
    correct = 0
    missing_gold = 0
    missing_pred = 0
    bad_format = 0
    format_pass = 0
    overall_sum = 0.0
    format_sum = 0.0
    accuracy_sum = 0.0

    for case in dataset:
        for q in case.get("questions", []):
            gold = get_gold_answer(q)
            raw_pred = str(q.get("response_raw", "")).strip()

            if not gold:
                missing_gold += 1
                continue

            total += 1

            if not raw_pred:
                missing_pred += 1
                continue

            scores = compute_score_eval(raw_pred, gold, format_weight=format_weight)
            overall_sum += scores["overall"]
            format_sum += scores["format"]
            accuracy_sum += scores["accuracy"]

            if scores["format"] == 1.0:
                format_pass += 1
            else:
                bad_format += 1

            if scores["accuracy"] == 1.0:
                correct += 1

    accuracy = (correct / total) if total > 0 else 0.0
    format_rate = (format_pass / total) if total > 0 else 0.0
    overall_mean = (overall_sum / total) if total > 0 else 0.0
    format_mean = (format_sum / total) if total > 0 else 0.0
    accuracy_mean = (accuracy_sum / total) if total > 0 else 0.0

    return {
        "total_scored_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "overall_mean": overall_mean,
        "format_mean": format_mean,
        "accuracy_mean": accuracy_mean,
        "missing_gold": missing_gold,
        "missing_pred": missing_pred,
        "bad_format": bad_format,
    }


# ---------- Core evaluation ----------


def validate_model_dir(model_path: str) -> None:
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Model/checkpoint path not found: {model_path}")

    expected_any = [
        "config.json",
        "adapter_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "model-00001-of-00004.safetensors",
    ]
    if not any(os.path.exists(os.path.join(model_path, name)) for name in expected_any):
        raise ValueError(
            f"{model_path} does not look like a loadable HF/vLLM model directory. "
            "If this is an FSDP/LoRA training checkpoint, you may need to export or merge it first."
        )


def run_one_model(
    model_path: str,
    args: argparse.Namespace,
    base_data: List[Dict[str, Any]],
    output_json: str,
) -> Dict[str, Any]:
    validate_model_dir(model_path)

    if args.resume and os.path.exists(output_json):
        data = load_json(output_json)
        if not isinstance(data, list) or len(data) != len(base_data):
            raise ValueError(
                f"Resume output size/type mismatch for {output_json}: got type={type(data)}, len={len(data) if isinstance(data, list) else 'N/A'}, "
                f"expected list len={len(base_data)}"
            )
        print(f"[Resume] Loaded existing output: {output_json}")
    else:
        data = copy.deepcopy(base_data)

    todo = build_todo_items(data, args.video_dir)
    total_cases = len(data)
    done_cases = total_cases - len(todo)
    total_questions = sum(len(item.get("questions", [])) for item in data)
    todo_questions = sum(len(qs) for _, _, qs in todo)
    done_questions = total_questions - todo_questions

    print(
        f"\n[{checkpoint_tag(model_path)}] Total cases: {total_cases}, done cases: {done_cases}, "
        f"todo cases: {len(todo)}; total questions: {total_questions}, done questions: {done_questions}, "
        f"todo questions: {todo_questions}"
    )

    if not todo:
        save_json(data, output_json)
        metrics = compute_accuracy(data, format_weight=args.format_weight)
        print(f"[{checkpoint_tag(model_path)}] Nothing to run. Output kept at: {output_json}")
        print(f"[{checkpoint_tag(model_path)}] Metrics: {metrics}")
        return {"data": data, "metrics": metrics}

    print(f"[{checkpoint_tag(model_path)}] Loading processor/model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=True,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    pbar = tqdm(
        range(0, len(todo), args.batch_size),
        desc=f"{checkpoint_tag(model_path)} inference",
    )

    for batch_i, start in enumerate(pbar, start=1):
        batch = todo[start : start + args.batch_size]
        vllm_inputs: List[Dict[str, Any]] = []
        index_pairs: List[Tuple[int, int]] = []

        for case_idx, video_path, case_questions in batch:
            cached_video_inputs, cached_video_kwargs = prepare_case_video_cache(
                video_path=video_path, num_frames=args.num_frames, processor=processor
            )
            for q_idx, prompt in case_questions:
                vllm_inputs.append(
                    build_vllm_input_with_cached_video(
                        prompt=prompt,
                        video_path=video_path,
                        cached_video_inputs=cached_video_inputs,
                        cached_video_kwargs=cached_video_kwargs,
                        num_frames=args.num_frames,
                        processor=processor,
                    )
                )
                index_pairs.append((case_idx, q_idx))

        if not vllm_inputs:
            continue

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        if len(outputs) != len(index_pairs):
            raise RuntimeError(
                f"Output length mismatch with inputs: {len(outputs)} vs {len(index_pairs)}"
            )

        for output, (case_idx, q_idx) in zip(outputs, index_pairs):
            text = output.outputs[0].text if output.outputs else ""
            gold = str(data[case_idx]["questions"][q_idx].get("answer", ""))
            scores = compute_score_eval(text, gold, format_weight=args.format_weight)
            answer = extract_answer_letter(text)

            data[case_idx]["questions"][q_idx]["response"] = answer or ""
            data[case_idx]["questions"][q_idx]["response_raw"] = text
            data[case_idx]["questions"][q_idx]["format_score"] = scores["format"]
            data[case_idx]["questions"][q_idx]["accuracy_score"] = scores["accuracy"]
            data[case_idx]["questions"][q_idx]["overall_score"] = scores["overall"]

        if batch_i % args.save_every == 0:
            save_json(data, output_json)
            pbar.set_postfix_str(f"saved@batch{batch_i}")

    save_json(data, output_json)
    metrics = compute_accuracy(data, format_weight=args.format_weight)
    print(f"[{checkpoint_tag(model_path)}] Done. Saved to: {output_json}")
    print(f"[{checkpoint_tag(model_path)}] Metrics: {metrics}")

    del llm
    del processor

    return {"data": data, "metrics": metrics}


# ---------- Summary ----------


def load_existing_summary(summary_json: str) -> List[Dict[str, Any]]:
    if not summary_json or not os.path.exists(summary_json):
        return []
    data = load_json(summary_json)
    if not isinstance(data, list):
        return []
    return data


def upsert_summary_row(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = row.get("model_path")
    updated = False
    new_rows = []
    for old in rows:
        if old.get("model_path") == key:
            new_rows.append(row)
            updated = True
        else:
            new_rows.append(old)
    if not updated:
        new_rows.append(row)
    return new_rows


# ---------- Main ----------


def main() -> None:
    args = parse_args()

    ensure_file(args.input_json, "Input JSON")
    ensure_dir(args.video_dir, "Video directory")

    raw_input = load_json(args.input_json)
    if not is_videomme_list(raw_input):
        raise ValueError(f"Expected list JSON, got: {type(raw_input)}")
    base_data: List[Dict[str, Any]] = raw_input

    if not args.allow_non_900_cases and len(base_data) != 900:
        raise ValueError(
            f"Expected 900 cases, got {len(base_data)} from {args.input_json}. "
            "If this is intentional, add --allow_non_900_cases."
        )

    model_paths = resolve_model_paths(args)
    print("Models/checkpoints to evaluate:")
    for i, path in enumerate(model_paths, start=1):
        print(f"  [{i}] {path}")

    summary_json = resolve_summary_json(args)
    if summary_json is not None:
        if args.overwrite_summary:
            summary_rows: List[Dict[str, Any]] = []
        else:
            summary_rows = load_existing_summary(summary_json)
    else:
        summary_rows = []

    for model_path in model_paths:
        output_json = resolve_output_json(args, model_path)
        result = run_one_model(
            model_path=model_path,
            args=args,
            base_data=base_data,
            output_json=output_json,
        )

        summary_row = {
            "model_path": model_path,
            "checkpoint_tag": checkpoint_tag(model_path),
            "output_json": output_json,
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "format_weight": args.format_weight,
            **result["metrics"],
        }
        summary_rows = upsert_summary_row(summary_rows, summary_row)

        if summary_json is not None:
            save_json(summary_rows, summary_json)
            print(f"Updated summary: {summary_json}")

    if summary_rows:
        print("\n===== Final Summary =====")
        sorted_rows = sorted(summary_rows, key=lambda x: x.get("accuracy", 0.0), reverse=True)
        for row in sorted_rows:
            print(
                f"{row['checkpoint_tag']}: accuracy={row.get('accuracy', 0.0):.4f}, "
                f"format_rate={row.get('format_rate', 0.0):.4f}, "
                f"overall_mean={row.get('overall_mean', 0.0):.4f} "
                f"({row.get('correct', 0)}/{row.get('total_scored_questions', 0)}) -> {row.get('output_json', '')}"
            )


if __name__ == "__main__":
    main()
