import os
import re
from typing import Any, Optional


# Metadata
REWARD_NAME = "r1v"
REWARD_TYPE = "sequential"


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
    """
    取最后一个 <answer>...</answer> 的内容。
    这样即使模型偶发输出多个 answer block，accuracy 也还能尽量稳定。
    """
    answers = _extract_all_answer_contents(response)
    if not answers:
        return None
    return answers[-1]


def _extract_option_letter(text: str) -> Optional[str]:
    """
    从文本中鲁棒提取选项字母。
    支持：
      B
      b
      B. eggs
      (B)
      Option B
      Choice B
      The answer is B
      answer: B
    """
    if not text:
        return None

    text = str(text).strip()

    patterns = [
        r"^\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"^\s*(?:option|choice)\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"^\s*(?:the answer is|answer is|answer)\s*[:\-]?\s*\(?([A-Z])\)?(?:[\s\.\:\-\)]|$)",
        r"\b([A-Z])\b",
    ]

    for p in patterns:
        match = re.search(p, text, flags=re.IGNORECASE)
        if match:
            ch = match.group(1).upper()
            if ch in _VALID_OPTION_SET:
                return ch

    return None


def format_reward(response: str) -> float:
    """Validate format: exactly one <thinking>...</thinking> followed by exactly one <answer>...</answer>.

    Requirements:
    - exactly one <thinking>...</thinking> block
    - exactly one <answer>...</answer> block
    - <thinking> must appear before <answer>
    - no text outside those tags (except optional whitespace)
    - no text after </answer>
    - <answer> content must be non-empty
    """
    response = _normalize_response(response)

    thinking_open = len(re.findall(r"<thinking>", response, flags=re.IGNORECASE))
    thinking_close = len(re.findall(r"</thinking>", response, flags=re.IGNORECASE))
    answer_open = len(re.findall(r"<answer>", response, flags=re.IGNORECASE))
    answer_close = len(re.findall(r"</answer>", response, flags=re.IGNORECASE))

    # Must have exactly one pair of each tag
    if thinking_open != 1 or thinking_close != 1:
        return 0.0
    if answer_open != 1 or answer_close != 1:
        return 0.0

    # Full structure: optional whitespace, <thinking>...</thinking>, optional whitespace,
    # <answer>...</answer>, optional trailing whitespace. Nothing else.
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

    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)
    gt_raw = _normalize_response(ground_truth)

    # 格式不合法，accuracy 直接记 0
    if format_reward(response) == 0.0:
        return 0.0

    # 解析 prediction
    given_answer = _extract_answer_content(response)
    pred = _extract_option_letter(given_answer) if given_answer is not None else None

    # 解析 ground truth
    gt_answer = _extract_answer_content(gt_raw)
    if gt_answer is not None:
        gt = _extract_option_letter(gt_answer)
    else:
        gt = _extract_option_letter(gt_raw)

    if gt not in _VALID_OPTION_SET:
        return 0.0

    return 1.0 if pred == gt else 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.3) -> dict[str, float]:
    response = _normalize_response(reward_input["response"])

    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

    if str(os.environ.get("R1V_DEBUG", "0")) == "1":
        all_answers = _extract_all_answer_contents(response)
        final_answer = _extract_answer_content(response)
        pred_letter = _extract_option_letter(final_answer) if final_answer is not None else None
        if pred_letter is None:
            pred_letter = _extract_option_letter(response)

        thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL | re.IGNORECASE)
        thinking_content = thinking_match.group(1).strip() if thinking_match else None

        print("=== R1V DEBUG START ===")
        print("raw_response:", repr(reward_input["response"]))
        print("normalized_response:", repr(response))
        print("ground_truth:", repr(reward_input["ground_truth"]))
        print("thinking_content:", repr(thinking_content[:200] if thinking_content else None))
        print("answer_open_count:", len(re.findall(r"<answer>", response, flags=re.IGNORECASE)))
        print("answer_close_count:", len(re.findall(r"</answer>", response, flags=re.IGNORECASE)))
        print("thinking_open_count:", len(re.findall(r"<thinking>", response, flags=re.IGNORECASE)))
        print("thinking_close_count:", len(re.findall(r"</thinking>", response, flags=re.IGNORECASE)))
        print("all_answer_contents:", repr(all_answers))
        print("final_answer_content:", repr(final_answer))
        print("pred_letter:", repr(pred_letter))
        print("format_score:", format_score)
        print("accuracy_score:", accuracy_score)
        print("=== R1V DEBUG END ===")

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
