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


def format_reward(response: str) -> float:
    response = _normalize_response(response)

    has_think = re.search(r"<think>.*?</think>", response, re.DOTALL) is not None
    has_answer = re.search(r"<answer>.*?</answer>", response, re.DOTALL) is not None

    return 1.0 if (has_think and has_answer) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    response = _normalize_response(response)

    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        given_answer = content_match.group(1).strip() if content_match else response.strip()
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
        print("=== R1V DEBUG START ===")
        print("raw_response:", repr(reward_input["response"]))
        print("normalized_response:", repr(response))
        print("has_think:", re.search(r"<think>.*?</think>", response, re.DOTALL) is not None)
        print("has_answer:", re.search(r"<answer>.*?</answer>", response, re.DOTALL) is not None)
        print("format_score:", format_score)
        print("accuracy_score:", accuracy_score)
        print("=== R1V DEBUG END ===")

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }