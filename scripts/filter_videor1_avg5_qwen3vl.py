import os
import re
import json
import argparse
import multiprocessing as mp
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "请先安装 qwen-vl-utils：\n"
        "pip install qwen-vl-utils[decord]\n"
        "或者：pip install qwen-vl-utils"
    )


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def append_jsonl(item: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_sample_key(item: Dict[str, Any]) -> str:
    if "problem_id" in item:
        return str(item["problem_id"])
    return str(item.get("path", "")) + "||" + str(item.get("problem", ""))


def load_scored_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    scored = {}

    if not os.path.exists(path):
        return scored

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                key = get_sample_key(item)
                scored[key] = item
            except Exception:
                continue

    return scored


def extract_answer_letter(text: str) -> Optional[str]:
    if text is None:
        return None

    text = text.strip()

    patterns = [
        r"<answer>\s*([A-F])\s*</answer>",
        r"<answer>\s*([A-F])",
        r"answer\s*[:：]\s*([A-F])",
        r"答案\s*[:：]?\s*([A-F])",
        r"\b([A-F])\b",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


def extract_gt_letter(solution: str) -> Optional[str]:
    return extract_answer_letter(solution)


def build_prompt(sample: Dict[str, Any]) -> str:
    problem = sample["problem"]
    options = sample.get("options", [])
    option_text = "\n".join(options)

    prompt = f"""You are given a video and a multiple-choice question.

Question:
{problem}

Options:
{option_text}

Please reason based on the video and choose the correct option.

You must output in the following format:
<thinking>Your reasoning here.</thinking>
<answer>A</answer>

The answer must be one single letter from A to F.
"""
    return prompt


def resolve_video_path(raw_path: str, video_root: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path

    raw_path = raw_path.lstrip("./")
    return os.path.join(video_root, raw_path)


@torch.inference_mode()
def infer_once(
    model,
    processor,
    sample: Dict[str, Any],
    video_root: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    num_frames: int = 16,
) -> str:
    video_path = resolve_video_path(sample["path"], video_root)
    prompt = build_prompt(sample)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": num_frames,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def evaluate_sample(
    model,
    processor,
    sample: Dict[str, Any],
    video_root: str,
    device: torch.device,
    n_rollouts: int = 5,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    num_frames: int = 16,
) -> Dict[str, Any]:
    gt = extract_gt_letter(sample.get("solution", ""))

    outputs = []
    pred_letters = []
    correct_list = []

    for _ in range(n_rollouts):
        try:
            output = infer_once(
                model=model,
                processor=processor,
                sample=sample,
                video_root=video_root,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_frames=num_frames,
            )
            pred = extract_answer_letter(output)
            correct = int(pred == gt)

        except Exception as e:
            output = f"[ERROR] {repr(e)}"
            pred = None
            correct = 0

        outputs.append(output)
        pred_letters.append(pred)
        correct_list.append(correct)

    pass_rate = sum(correct_list) / n_rollouts

    scored_item = dict(sample)
    scored_item["avg5_pass_rate"] = pass_rate
    scored_item["avg5_correct_list"] = correct_list
    scored_item["avg5_pred_letters"] = pred_letters
    scored_item["avg5_outputs"] = outputs
    scored_item["gt_letter"] = gt

    return scored_item


def rebuild_filtered_from_scored(
    scored_items: Dict[str, Dict[str, Any]],
    min_pass_rate: float,
    max_pass_rate: float,
) -> List[Dict[str, Any]]:
    filtered = []

    for _, item in scored_items.items():
        pass_rate = item.get("avg5_pass_rate", None)
        if pass_rate is None:
            continue

        if min_pass_rate <= pass_rate <= max_pass_rate:
            filtered.append(item)

    return filtered


def worker_loop(
    worker_id: int,
    gpu_id: int,
    samples: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    result_queue,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[Worker {worker_id}] loading model on cuda:{gpu_id}, samples={len(samples)}")

    processor = AutoProcessor.from_pretrained(
        args_dict["model_path"],
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args_dict["model_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args_dict["attn_implementation"],
    )

    model.to(device)
    model.eval()

    print(f"[Worker {worker_id}] model loaded on cuda:{gpu_id}")

    for sample in samples:
        key = get_sample_key(sample)

        scored_item = evaluate_sample(
            model=model,
            processor=processor,
            sample=sample,
            video_root=args_dict["video_root"],
            device=device,
            n_rollouts=args_dict["n_rollouts"],
            temperature=args_dict["temperature"],
            top_p=args_dict["top_p"],
            max_new_tokens=args_dict["max_new_tokens"],
            num_frames=args_dict["num_frames"],
        )

        result_queue.put(
            {
                "type": "result",
                "key": key,
                "item": scored_item,
                "worker_id": worker_id,
                "gpu_id": gpu_id,
            }
        )

    result_queue.put(
        {
            "type": "done",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
        }
    )


def split_samples(samples: List[Dict[str, Any]], num_workers: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(num_workers)]

    for idx, sample in enumerate(samples):
        shards[idx % num_workers].append(sample)

    return shards


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/xhou/datasets/Video-R1-data/Video-R1-260k_video_mc.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/xhou/datasets/Video-R1-data/Video-R1-260k_video_mc_avg5_filtered.json",
    )
    parser.add_argument(
        "--scored_jsonl_path",
        type=str,
        default="/data/xhou/datasets/Video-R1-data/Video-R1-260k_video_mc_avg5_scored.jsonl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data0/xhou/models/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="/data/xhou/datasets/Video-R1-data",
    )

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")

    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16)

    parser.add_argument("--min_pass_rate", type=float, default=0.2)
    parser.add_argument("--max_pass_rate", type=float, default=0.8)
    parser.add_argument("--save_every", type=int, default=20)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    assert args.num_workers <= len(gpu_ids), "num_workers 不能大于 gpu_ids 数量"

    print("=" * 80)
    print("Input:", args.input_path)
    print("Output:", args.output_path)
    print("Scored jsonl:", args.scored_jsonl_path)
    print("Model:", args.model_path)
    print("Video root:", args.video_root)
    print("num_workers:", args.num_workers)
    print("gpu_ids:", gpu_ids)
    print("n_rollouts:", args.n_rollouts)
    print("temperature:", args.temperature)
    print("pass_rate range:", args.min_pass_rate, args.max_pass_rate)
    print("=" * 80)

    data = load_json(args.input_path)
    print(f"Total input samples: {len(data)}")

    data = [
        item for item in data
        if item.get("data_type") == "video"
        and item.get("problem_type") == "multiple choice"
    ]
    print(f"Valid video multiple-choice samples: {len(data)}")

    scored_items = load_scored_jsonl(args.scored_jsonl_path)
    processed_keys = set(scored_items.keys())
    print(f"Already processed samples: {len(processed_keys)}")

    remaining_samples = [
        item for item in data
        if get_sample_key(item) not in processed_keys
    ]
    print(f"Remaining samples: {len(remaining_samples)}")

    filtered_data = rebuild_filtered_from_scored(
        scored_items=scored_items,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )
    save_json(filtered_data, args.output_path)
    print(f"Current filtered samples from previous scored file: {len(filtered_data)}")

    if len(remaining_samples) == 0:
        print("No remaining samples. Done.")
        return

    shards = split_samples(remaining_samples, args.num_workers)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=args.num_workers * 4)

    args_dict = vars(args)

    processes = []
    for worker_id in range(args.num_workers):
        gpu_id = gpu_ids[worker_id]
        p = ctx.Process(
            target=worker_loop,
            args=(
                worker_id,
                gpu_id,
                shards[worker_id],
                args_dict,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    done_workers = 0
    newly_processed = 0

    pbar = tqdm(total=len(remaining_samples), desc="8GPU filtering")

    while done_workers < args.num_workers:
        msg = result_queue.get()

        if msg["type"] == "done":
            done_workers += 1
            print(
                f"\nWorker {msg['worker_id']} on cuda:{msg['gpu_id']} done. "
                f"done_workers={done_workers}/{args.num_workers}"
            )
            continue

        if msg["type"] == "result":
            key = msg["key"]
            scored_item = msg["item"]

            append_jsonl(scored_item, args.scored_jsonl_path)

            scored_items[key] = scored_item
            processed_keys.add(key)
            newly_processed += 1

            pass_rate = scored_item["avg5_pass_rate"]

            if args.min_pass_rate <= pass_rate <= args.max_pass_rate:
                filtered_data.append(scored_item)

            if newly_processed % args.save_every == 0:
                filtered_data = rebuild_filtered_from_scored(
                    scored_items=scored_items,
                    min_pass_rate=args.min_pass_rate,
                    max_pass_rate=args.max_pass_rate,
                )
                save_json(filtered_data, args.output_path)

                print(
                    f"\nSaved checkpoint. "
                    f"newly_processed={newly_processed}, "
                    f"total_processed={len(processed_keys)}, "
                    f"filtered={len(filtered_data)}"
                )

            pbar.update(1)

    pbar.close()

    for p in processes:
        p.join()

    filtered_data = rebuild_filtered_from_scored(
        scored_items=scored_items,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
    )
    save_json(filtered_data, args.output_path)

    print("=" * 80)
    print("Done.")
    print(f"Total processed: {len(processed_keys)}")
    print(f"Final filtered samples: {len(filtered_data)}")
    print(f"Saved filtered json to: {args.output_path}")
    print(f"Saved scored jsonl to: {args.scored_jsonl_path}")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()