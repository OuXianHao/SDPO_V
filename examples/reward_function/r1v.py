import re
from typing import Any

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "r1v"
REWARD_TYPE = "sequential"


def _normalize_response(response: str) -> str:
    if response is None:
        return ""
    response = str(response).strip()
    response = re.sub(r"\s+", " ", response)
    response = re.sub(r"\s*<\s*", "<", response)
    response = re.sub(r"\s*>\s*", ">", response)
    response = re.sub(r"\s*/\s*", "/", response)
    return response.strip()


def _extract_answer_content(response: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>\s*$", response, re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    return content if content else None


def format_reward(response: str) -> float:
    response = _normalize_response(response)

    # 必须恰好有一对 answer 标签
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return 0.0

    # 必须以 <answer>...</answer> 结尾，后面不能再有别的文本
    match = re.search(r"^(.*)<answer>(.*?)</answer>\s*$", response, re.DOTALL)
    if not match:
        return 0.0

    answer_content = match.group(2).strip()
    if not answer_content:
        return 0.0

    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)

    try:
        given_answer = _extract_answer_content(response)
        if given_answer is None:
            return 0.0
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0
    except Exception:
        pass

    return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    response = _normalize_response(reward_input["response"])

    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

    if str(__import__("os").environ.get("R1V_DEBUG", "0")) == "1":
        answer_match = re.search(r"^(.*)<answer>(.*?)</answer>\s*$", response, re.DOTALL)
        print("=== R1V DEBUG START ===")
        print("raw_response:", repr(reward_input["response"]))
        print("normalized_response:", repr(response))
        print("answer_open_count:", response.count("<answer>"))
        print("answer_close_count:", response.count("</answer>"))
        print("ends_with_answer_block:", answer_match is not None)
        print("answer_content:", repr(answer_match.group(2).strip()) if answer_match else None)
        print("format_score:", format_score)
        print("accuracy_score:", accuracy_score)
        print("=== R1V DEBUG END ===")

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }