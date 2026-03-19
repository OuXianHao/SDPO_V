import os
from typing import Any, Optional

import numpy as np
import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in _TRUE_VALUES


def is_debug_shapes_enabled() -> bool:
    return _env_enabled("EASYR1_DEBUG_SHAPES")


def is_debug_mm_enabled() -> bool:
    return _env_enabled("EASYR1_DEBUG_MM")


def detect_rank(default: str = "unknown") -> str:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def _safe_shape_dtype(value: Any) -> Optional[tuple[tuple[int, ...], str]]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape), str(value.dtype)
    if isinstance(value, np.ndarray):
        return tuple(value.shape), str(value.dtype)
    if hasattr(value, "shape"):
        shape = value.shape
        try:
            shape = tuple(shape)
        except Exception:
            shape = (str(shape),)
        dtype = getattr(value, "dtype", type(value).__name__)
        return shape, str(dtype)
    return None


def summarize_tensor_dict_for_debug(data: Any, max_items: int = 8) -> str:
    if data is None:
        return "none"
    pieces = []
    for idx, key in enumerate(sorted(data.keys())):
        if idx >= max_items:
            pieces.append("...")
            break
        info = _safe_shape_dtype(data[key])
        if info is None:
            pieces.append(f"{key}:type={type(data[key]).__name__}")
        else:
            shape, dtype = info
            pieces.append(f"{key}:shape={shape},dtype={dtype}")
    return "; ".join(pieces) if pieces else "empty"


def summarize_batch_for_debug(batch: Any, rank: Optional[Any] = None, max_problem_ids: int = 4) -> str:
    rank = detect_rank() if rank is None else rank
    batch_size = len(batch) if batch is not None else 0
    tensor_summary = summarize_tensor_dict_for_debug(getattr(batch, "batch", None))
    non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    problem_ids = non_tensor.get("problem_id")
    uid = non_tensor.get("uid")
    if problem_ids is None:
        problem_ids = uid
    preview = []
    if problem_ids is not None:
        for x in list(problem_ids)[:max_problem_ids]:
            preview.append(str(x))
    return (
        f"rank={rank} batch_size={batch_size} "
        f"tensor=({tensor_summary}) "
        f"problem_ids_preview={preview}"
    )


def summarize_multimodal_input_for_debug(
    *,
    rank: Any,
    problem_id: Any,
    video_paths: Any,
    fixed_frames: Optional[int],
    metadata: Optional[dict[str, Any]],
    mm_input: Any,
    input_kind: str,
) -> str:
    info = _safe_shape_dtype(mm_input)
    if info is None:
        shape_dtype = f"shape=None,dtype={type(mm_input).__name__ if mm_input is not None else 'None'}"
    else:
        shape, dtype = info
        shape_dtype = f"shape={shape},dtype={dtype}"
    metadata_exists = metadata is not None
    return (
        f"rank={rank} problem_id={problem_id} videos={video_paths} "
        f"fixed_frames={fixed_frames} metadata_exists={metadata_exists} "
        f"input_kind={input_kind} {shape_dtype}"
    )
