# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from typing import Any

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "r1v"
REWARD_TYPE = "sequential"


def _normalize_response(response: str) -> str:
    # Handle Qwen-VL style tag spacing variants such as "< think >...< /think >".
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    # Avoid false negative fullmatch when model adds harmless leading/trailing newlines.
    return response.strip()


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", response)
        given_answer = content_match.group(1).strip() if content_match else response.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0

    except Exception:
        pass

    return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    raw_response = reward_input["response"]
    response = _normalize_response(raw_response)
    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

    if os.getenv("R1V_DEBUG", "0") == "1":
        has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
        print(
            "[R1V_DEBUG] "
            f"raw={raw_response!r} "
            f"normalized={response!r} "
            f"has_think={has_think} "
            f"has_answer={has_answer}"
        )

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
