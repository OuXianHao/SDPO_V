import argparse
import copy
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# -----------------------------
# Core prompt templates
# -----------------------------

BASE_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Keep the reasoning clear and concise, and place it before the final answer.

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Do not output any text after </answer>.

User: {content}
Assistant:"""

GUIDELINE_SUMMARY_PROMPT = """You are given multiple candidate reasoning traces for the same video multiple-choice question. Some are correct and some are incorrect.

Your task is to analyze these rollouts and produce a feedback guideline that can improve later reasoning on this sample.

Instructions:
- Identify the key reasoning patterns used by correct rollouts.
- Identify the typical mistakes made by incorrect rollouts.
- Focus on how to inspect the video evidence better.
- Pay special attention to temporal sequence, transitions, object interactions, state changes, repeated actions, and mismatches between the video and the options.
- Do NOT say which option is correct.
- Avoid vague advice such as "reason carefully" or "pay attention to the video".
- Make each guidance point concrete and operational.
- The final "Actionable guidance" section should be the most important part, because it will be reused as feedback for later reasoning.

Output format:

<guideline_analysis>
Strengths of correct reasoning:
- ...
- ...

Common mistakes to avoid:
- ...
- ...
</guideline_analysis>

<guideline_feedback>
- ...
- ...
- ...
- ...
</guideline_feedback>

Question:
{question}

Options:
{options}

Candidate rollouts:

{rollout_blocks}

Now write the guideline.
"""

CORRECT_ROLLOUT_FEEDBACK_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Keep the reasoning clear and concise, and place it before the final answer.

Below is one rollout that is known to be a correct solution for this same sample. Use it as reference feedback to guide your reasoning, but still ground your answer in the video.

<correct_rollout_feedback>
{correct_rollout}
</correct_rollout_feedback>

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Do not output any text after </answer>.

User: {content}
Assistant:"""

GUIDELINE_FEEDBACK_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before giving the final answer, the Assistant should reason about how the answer is obtained from the video.
The reasoning should be grounded in the visual evidence, especially temporal order, repeated actions, object motion, state changes, and outcome differences.
Keep the reasoning clear and concise, and place it before the final answer.

Below is feedback distilled from multiple candidate rollouts for this same sample. Use it to guide your reasoning, but still ground your answer in the video.

<feedback_guideline>
{guideline_feedback}
</feedback_guideline>

The final answer must be enclosed in exactly one pair of <answer> and </answer> tags.
Do not output any text after </answer>.

User: {content}
Assistant:"""

_VALID_OPTION_SET = {"A", "B", "C", "D", "E", "F"}


# -----------------------------
# Args / IO
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-branch vLLM experiment for Video-R1 flat MCQ JSON."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=20)

    parser.add_argument("--max_samples", type=int, default=200, help="How many samples to run.")
    parser.add_argument("--sample_mode", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--export_guideline_cases", type=int, default=10)

    parser.add_argument("--rollout_count", type=int, default=8)
    parser.add_argument("--mixed_pick_count", type=int, default=5)
    parser.add_argument("--rollout_max_tokens", type=int, default=256)
    parser.add_argument("--summary_max_tokens", type=int, default=256)
    parser.add_argument("--answer_max_tokens", type=int, default=256)

    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--summary_temperature", type=float, default=0.0)
    parser.add_argument("--summary_top_p", type=float, default=1.0)
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument("--answer_top_p", type=float, default=1.0)

    parser.add_argument("--format_weight", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


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


def validate_model_dir(model_path: str) -> None:
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Model path not found: {model_path}")

    expected_any = [
        "config.json",
        "adapter_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "model-00001-of-00004.safetensors",
    ]
    if not any(os.path.exists(os.path.join(model_path, name)) for name in expected_any):
        raise ValueError(f"{model_path} does not look like a loadable HF/vLLM model directory.")


# -----------------------------
# Prompt helpers
# -----------------------------

def build_user_content(question_text: str, options: List[str]) -> str:
    options_text = "\n".join(options)
    return f"Question: {question_text.strip()}\n\nOptions:\n{options_text}"


def format_base_question_prompt(question: str, options: List[str]) -> str:
    user_content = build_user_content(question, options)
    return BASE_PROMPT_TEMPLATE.format(content=user_content.strip())


def format_correct_rollout_feedback_prompt(question: str, options: List[str], correct_rollout: str) -> str:
    user_content = build_user_content(question, options)
    return CORRECT_ROLLOUT_FEEDBACK_TEMPLATE.format(
        correct_rollout=correct_rollout.strip(),
        content=user_content.strip(),
    )


def format_guideline_feedback_prompt(question: str, options: List[str], guideline_feedback: str) -> str:
    user_content = build_user_content(question, options)
    return GUIDELINE_FEEDBACK_TEMPLATE.format(
        guideline_feedback=guideline_feedback.strip(),
        content=user_content.strip(),
    )


def format_rollout_block(idx: int, correctness_label: str, response: str) -> str:
    return (
        f"[Rollout {idx}]\n"
        f"Correctness: {correctness_label}\n"
        f"Response:\n"
        f"{response.strip()}"
    )


def format_guideline_summary_prompt(
    question: str,
    options: List[str],
    selected_rollouts: List[Dict[str, Any]],
) -> str:
    option_text = "\n".join(options)
    rollout_blocks = []
    for i, item in enumerate(selected_rollouts, start=1):
        correctness_label = "correct" if item["accuracy_score"] >= 1.0 else "incorrect"
        rollout_blocks.append(format_rollout_block(i, correctness_label, item["text"]))
    return GUIDELINE_SUMMARY_PROMPT.format(
        question=question.strip(),
        options=option_text,
        rollout_blocks="\n\n".join(rollout_blocks),
    )


# -----------------------------
# Scoring / parsing
# -----------------------------

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


def _extract_option_letter_robust(text: str) -> Optional[str]:
    if not text:
        return None

    text = str(text).strip()
    patterns = [
        r"^\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"^\s*(?:option|choice)\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"^\s*(?:the answer is|answer is|answer)\s*[:\-]?\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"\b([A-Z])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            ch = match.group(1).upper()
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

    return 1.0


def accuracy_reward_eval(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)
    gt_raw = _normalize_response(ground_truth)

    if format_reward_eval(response) == 0.0:
        return 0.0

    given_answer = _extract_answer_content(response)
    pred = _extract_option_letter_robust(given_answer) if given_answer is not None else None

    gt_answer = _extract_answer_content(gt_raw)
    if gt_answer is not None:
        gt = _extract_option_letter_robust(gt_answer)
    else:
        gt = _extract_option_letter_robust(gt_raw)

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
    pred = _extract_option_letter_robust(final_answer_content) if final_answer_content is not None else None
    return pred or ""


def extract_xml_block(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def get_gold_answer(item: Dict[str, Any]) -> str:
    candidate_keys = ["solution", "answer", "gt_answer", "correct_answer", "gold_answer", "label"]
    for key in candidate_keys:
        if key in item:
            value = str(item[key]).strip()
            if not value:
                continue
            answer_block = _extract_answer_content(value)
            if answer_block is not None:
                pred = _extract_option_letter_robust(answer_block)
                if pred:
                    return pred
            pred = _extract_option_letter_robust(value)
            if pred:
                return pred
    return ""


# -----------------------------
# Video helpers
# -----------------------------

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
) -> Dict[str, Any]:
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


# -----------------------------
# Data prep
# -----------------------------

def validate_record(item: Dict[str, Any], idx: int) -> None:
    if str(item.get("data_type", "")).lower() != "video":
        raise ValueError(f"Record {idx} is not video data.")
    if "problem" not in item or not str(item["problem"]).strip():
        raise ValueError(f"Record {idx} missing problem.")
    if "options" not in item or not isinstance(item["options"], list) or len(item["options"]) == 0:
        raise ValueError(f"Record {idx} missing options.")
    if "path" not in item or not str(item["path"]).strip():
        raise ValueError(f"Record {idx} missing path.")
    if not os.path.isfile(item["path"]):
        raise FileNotFoundError(f"Record {idx} video path not found: {item['path']}")
    gold = get_gold_answer(item)
    if gold not in _VALID_OPTION_SET:
        raise ValueError(f"Record {idx} invalid gold answer: {item.get('solution', '')}")


def select_subset(dataset: List[Dict[str, Any]], max_samples: int, sample_mode: str, seed: int) -> List[Dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(dataset):
        return copy.deepcopy(dataset)
    if sample_mode == "first":
        return copy.deepcopy(dataset[:max_samples])
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    picked = sorted(indices[:max_samples])
    return copy.deepcopy([dataset[i] for i in picked])


def init_output_data(base_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data = copy.deepcopy(base_data)
    for item in data:
        item.setdefault("exp_rollouts", [])
        item.setdefault("exp_rollout_stats", {})
        item.setdefault("exp_correct_rollout_baseline", {})
        item.setdefault("exp_guideline_method", {})
    return data


def build_records(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for idx, item in enumerate(dataset):
        records.append({
            "idx": idx,
            "problem_id": item.get("problem_id"),
            "question": str(item["problem"]).strip(),
            "options": item["options"],
            "gold": get_gold_answer(item),
            "video_path": item["path"],
        })
    return records


# -----------------------------
# Generation stages
# -----------------------------

def generate_rollouts_for_record(
    llm: LLM,
    processor: AutoProcessor,
    record: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    prompt = format_base_question_prompt(record["question"], record["options"])
    cached_video_inputs, cached_video_kwargs = prepare_case_video_cache(
        record["video_path"], args.num_frames, processor
    )
    vllm_input = build_vllm_input_with_cached_video(
        prompt=prompt,
        video_path=record["video_path"],
        cached_video_inputs=cached_video_inputs,
        cached_video_kwargs=cached_video_kwargs,
        num_frames=args.num_frames,
        processor=processor,
    )

    sp = SamplingParams(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        max_tokens=args.rollout_max_tokens,
        n=args.rollout_count,
    )
    outputs = llm.generate([vllm_input], sampling_params=sp)
    result = []
    request_output = outputs[0]
    for i, out in enumerate(request_output.outputs):
        text = out.text if out else ""
        scores = compute_score_eval(text, record["gold"], format_weight=args.format_weight)
        result.append({
            "rollout_id": i,
            "text": text,
            "answer": extract_answer_letter(text),
            "format_score": scores["format"],
            "accuracy_score": scores["accuracy"],
            "overall_score": scores["overall"],
        })
    return result


def pick_mixed_rollouts(rollouts: List[Dict[str, Any]], pick_count: int) -> List[Dict[str, Any]]:
    correct = [r for r in rollouts if r["accuracy_score"] >= 1.0]
    incorrect = [r for r in rollouts if r["accuracy_score"] < 1.0]
    if not correct or not incorrect:
        return []

    correct = sorted(correct, key=lambda x: (-x["overall_score"], x["rollout_id"]))
    incorrect = sorted(incorrect, key=lambda x: (x["overall_score"], x["rollout_id"]))

    selected = [correct[0], incorrect[0]]
    used_ids = {selected[0]["rollout_id"], selected[1]["rollout_id"]}

    remaining = [r for r in rollouts if r["rollout_id"] not in used_ids]
    remaining = sorted(remaining, key=lambda x: (-abs(x["overall_score"] - 0.5), x["rollout_id"]))

    for r in remaining:
        if len(selected) >= pick_count:
            break
        selected.append(r)
    return selected[:pick_count]


def select_one_correct_rollout(rollouts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    correct = [r for r in rollouts if r["accuracy_score"] >= 1.0]
    if not correct:
        return None
    correct = sorted(correct, key=lambda x: (-x["overall_score"], x["rollout_id"]))
    return correct[0]


def summarize_guideline_for_record(
    llm: LLM,
    processor: AutoProcessor,
    record: Dict[str, Any],
    selected_rollouts: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, str]:
    prompt = format_guideline_summary_prompt(
        question=record["question"],
        options=record["options"],
        selected_rollouts=selected_rollouts,
    )
    cached_video_inputs, cached_video_kwargs = prepare_case_video_cache(
        record["video_path"], args.num_frames, processor
    )
    vllm_input = build_vllm_input_with_cached_video(
        prompt=prompt,
        video_path=record["video_path"],
        cached_video_inputs=cached_video_inputs,
        cached_video_kwargs=cached_video_kwargs,
        num_frames=args.num_frames,
        processor=processor,
    )
    sp = SamplingParams(
        temperature=args.summary_temperature,
        top_p=args.summary_top_p,
        max_tokens=args.summary_max_tokens,
    )
    outputs = llm.generate([vllm_input], sampling_params=sp)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return {
        "summary_raw": text,
        "guideline_analysis": extract_xml_block(text, "guideline_analysis"),
        "guideline_feedback": extract_xml_block(text, "guideline_feedback"),
    }


def answer_with_prompt(
    llm: LLM,
    processor: AutoProcessor,
    record: Dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cached_video_inputs, cached_video_kwargs = prepare_case_video_cache(
        record["video_path"], args.num_frames, processor
    )
    vllm_input = build_vllm_input_with_cached_video(
        prompt=prompt,
        video_path=record["video_path"],
        cached_video_inputs=cached_video_inputs,
        cached_video_kwargs=cached_video_kwargs,
        num_frames=args.num_frames,
        processor=processor,
    )
    sp = SamplingParams(
        temperature=args.answer_temperature,
        top_p=args.answer_top_p,
        max_tokens=args.answer_max_tokens,
    )
    outputs = llm.generate([vllm_input], sampling_params=sp)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    scores = compute_score_eval(text, record["gold"], format_weight=args.format_weight)
    return {
        "response_raw": text,
        "response": extract_answer_letter(text),
        "format_score": scores["format"],
        "accuracy_score": scores["accuracy"],
        "overall_score": scores["overall"],
    }


# -----------------------------
# Metrics / export
# -----------------------------

def compute_experiment_metrics(dataset: List[Dict[str, Any]], exp_key: str) -> Dict[str, Any]:
    total = 0
    correct = 0
    format_pass = 0
    overall_sum = 0.0
    accuracy_sum = 0.0
    format_sum = 0.0
    missing = 0

    for item in dataset:
        exp = item.get(exp_key, {})
        raw_pred = str(exp.get("response_raw", "")).strip()
        if not raw_pred:
            missing += 1
            continue
        total += 1
        format_sum += float(exp.get("format_score", 0.0))
        accuracy_sum += float(exp.get("accuracy_score", 0.0))
        overall_sum += float(exp.get("overall_score", 0.0))
        if float(exp.get("format_score", 0.0)) >= 1.0:
            format_pass += 1
        if float(exp.get("accuracy_score", 0.0)) >= 1.0:
            correct += 1

    return {
        "total_scored_questions": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "format_rate": (format_pass / total) if total > 0 else 0.0,
        "overall_mean": (overall_sum / total) if total > 0 else 0.0,
        "format_mean": (format_sum / total) if total > 0 else 0.0,
        "accuracy_mean": (accuracy_sum / total) if total > 0 else 0.0,
        "missing_predictions": missing,
    }


def export_guideline_cases(dataset: List[Dict[str, Any]], path: str, limit: int) -> None:
    cases = []
    for item in dataset:
        g = item.get("exp_guideline_method", {})
        if g.get("status") == "ok":
            cases.append({
                "problem_id": item.get("problem_id"),
                "data_source": item.get("data_source"),
                "path": item.get("path"),
                "problem": item.get("problem"),
                "options": item.get("options"),
                "solution": item.get("solution"),
                "guideline_feedback": g.get("guideline_feedback", ""),
                "guideline_analysis": g.get("guideline_analysis", ""),
                "guideline_response": g.get("response", ""),
                "guideline_accuracy_score": g.get("accuracy_score", 0.0),
                "baseline_response": item.get("exp_correct_rollout_baseline", {}).get("response", ""),
                "baseline_accuracy_score": item.get("exp_correct_rollout_baseline", {}).get("accuracy_score", 0.0),
            })
        if len(cases) >= limit:
            break
    save_json(cases, path)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ensure_file(args.input_json, "Input JSON")
    validate_model_dir(args.model_path)

    base_data = load_json(args.input_json)
    if not isinstance(base_data, list):
        raise ValueError(f"Expected list JSON, got: {type(base_data)}")

    selected_data = select_subset(base_data, args.max_samples, args.sample_mode, args.seed)
    for i, item in enumerate(selected_data):
        validate_record(item, i)

    if args.resume and os.path.exists(args.output_json):
        data = load_json(args.output_json)
        if not isinstance(data, list):
            raise ValueError("Resume output is not a list.")
        print(f"[Resume] Loaded existing output: {args.output_json}")
    else:
        data = init_output_data(selected_data)

    records = build_records(data)
    print(f"Total selected questions: {len(records)}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=True,
        seed=args.seed,
    )

    pbar = tqdm(records, desc="Video-R1 two-branch experiment")
    for step, record in enumerate(pbar, start=1):
        obj = data[record["idx"]]

        if args.resume:
            base_done = bool(obj.get("exp_correct_rollout_baseline", {}).get("response_raw"))
            guide_done = bool(obj.get("exp_guideline_method", {}).get("response_raw"))
            if base_done and guide_done and obj.get("exp_rollouts"):
                continue

        rollouts = generate_rollouts_for_record(llm, processor, record, args)
        obj["exp_rollouts"] = rollouts
        obj["exp_rollout_stats"] = {
            "rollout_count": len(rollouts),
            "num_correct": sum(1 for r in rollouts if r["accuracy_score"] >= 1.0),
            "num_incorrect": sum(1 for r in rollouts if r["accuracy_score"] < 1.0),
        }

        correct_rollout = select_one_correct_rollout(rollouts)
        baseline_result = {
            "status": "ok" if correct_rollout is not None else "skipped_no_correct_rollout",
            "selected_correct_rollout": correct_rollout["text"] if correct_rollout is not None else "",
            "selected_correct_rollout_answer": correct_rollout["answer"] if correct_rollout is not None else "",
        }
        if correct_rollout is not None:
            baseline_prompt = format_correct_rollout_feedback_prompt(
                question=record["question"],
                options=record["options"],
                correct_rollout=correct_rollout["text"],
            )
            baseline_answer = answer_with_prompt(llm, processor, record, baseline_prompt, args)
            baseline_result["prompt"] = baseline_prompt
            baseline_result.update(baseline_answer)
        obj["exp_correct_rollout_baseline"] = baseline_result

        selected_rollouts = pick_mixed_rollouts(rollouts, pick_count=args.mixed_pick_count)
        guide_result = {
            "status": "ok" if selected_rollouts else "skipped_no_mixed_rollouts",
            "selected_rollouts": selected_rollouts,
        }
        if selected_rollouts:
            summary = summarize_guideline_for_record(llm, processor, record, selected_rollouts, args)
            guideline_feedback = summary["guideline_feedback"] or summary["summary_raw"]
            guide_prompt = format_guideline_feedback_prompt(
                question=record["question"],
                options=record["options"],
                guideline_feedback=guideline_feedback,
            )
            guide_answer = answer_with_prompt(llm, processor, record, guide_prompt, args)
            guide_result.update(summary)
            guide_result["prompt"] = guide_prompt
            guide_result.update(guide_answer)
        obj["exp_guideline_method"] = guide_result

        if step % args.save_every == 0:
            save_json(data, args.output_json)
            pbar.set_postfix_str(f"saved@{step}")

    save_json(data, args.output_json)

    baseline_metrics = compute_experiment_metrics(data, "exp_correct_rollout_baseline")
    guideline_metrics = compute_experiment_metrics(data, "exp_guideline_method")
    summary = {
        "model_path": args.model_path,
        "output_json": args.output_json,
        "input_json": args.input_json,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "num_frames": args.num_frames,
        "rollout_count": args.rollout_count,
        "mixed_pick_count": args.mixed_pick_count,
        "rollout_temperature": args.rollout_temperature,
        "summary_temperature": args.summary_temperature,
        "answer_temperature": args.answer_temperature,
        "correct_rollout_baseline": baseline_metrics,
        "guideline_method": guideline_metrics,
    }

    summary_path = os.path.splitext(args.output_json)[0] + ".summary.json"
    save_json(summary, summary_path)

    cases_path = os.path.splitext(args.output_json)[0] + f".guideline_cases_top{args.export_guideline_cases}.json"
    export_guideline_cases(data, cases_path, args.export_guideline_cases)

    print("\n===== Experiment Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved guideline cases to: {cases_path}")

    del llm
    del processor


if __name__ == "__main__":
    main()