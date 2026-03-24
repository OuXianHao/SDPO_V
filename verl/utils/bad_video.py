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
"""
Bad-video construction for SDPO-V.

Degrades pixel_values_videos tensor to create the bad-video branch
while preserving exact tensor shapes expected by the model.

Supported modes:
  - "blur": Gaussian-blur a fraction of temporal frames.
  - "drop": Zero-out a fraction of temporal frames.
  - "blur_and_drop": Apply both blur and drop (on disjoint frame subsets).

All operations are applied in-place on a **clone** of the original tensor
so the good-video branch is never modified.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: float, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_frames(
    frames: torch.Tensor,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Apply Gaussian blur to frames tensor.

    Args:
        frames: (N, C, H, W) or (N, C*H*W) — if flattened, we cannot blur
                spatially, so we blur along the feature dimension as a proxy.
                For pixel_values_videos from Qwen2-VL, the shape is typically
                (total_patches, patch_dim) where each row is a flattened patch.
        sigma: Gaussian blur sigma.

    Returns:
        Blurred frames with same shape.
    """
    if frames.dim() == 2:
        # pixel_values_videos is (total_visual_tokens, hidden_dim)
        # Apply 1D blur along the hidden_dim as spatial proxy
        kernel_size = max(int(math.ceil(sigma * 3)) | 1, 3)  # ensure odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = _gaussian_kernel_1d(sigma, kernel_size, frames.device, frames.dtype)
        # Treat as (N, 1, D) for conv1d
        x = frames.unsqueeze(1)  # (N, 1, D)
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")
        blurred = F.conv1d(x_padded, kernel.view(1, 1, -1))
        return blurred.squeeze(1)
    elif frames.dim() == 4:
        # (N, C, H, W) — proper 2D spatial blur
        N, C, H, W = frames.shape
        kernel_size = max(int(math.ceil(sigma * 3)) | 1, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_1d = _gaussian_kernel_1d(sigma, kernel_size, frames.device, frames.dtype)
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (K, K)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
        pad = kernel_size // 2
        x_padded = F.pad(frames, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(x_padded, kernel_2d, groups=C)
        return blurred
    else:
        # Fallback: add Gaussian noise as degradation
        noise = torch.randn_like(frames) * sigma * 0.01
        return frames + noise


def construct_bad_video_inputs(
    multi_modal_inputs: dict[str, torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    mode: str = "blur",
    blur_sigma: float = 5.0,
    blur_fraction: float = 0.5,
    drop_fraction: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Construct bad-video multimodal inputs by degrading pixel_values_videos.

    This function clones the multimodal inputs and applies degradation to the
    video pixel values. The output has exactly the same tensor shapes as the
    input, so it can be passed directly to the model.

    The degradation is applied at the temporal-frame level:
    - video_grid_thw tells us (T, H_patches, W_patches) for each video
    - We select a fraction of temporal frames and degrade those frames'
      visual tokens.

    Args:
        multi_modal_inputs: Dict of multimodal tensors (pixel_values_videos,
            video_grid_thw, etc). Will NOT be modified.
        video_grid_thw: (num_videos, 3) tensor of (T, H, W) per video.
        mode: Degradation mode — "blur", "drop", or "blur_and_drop".
        blur_sigma: Gaussian blur sigma.
        blur_fraction: Fraction of frames to blur.
        drop_fraction: Fraction of frames to drop (zero out).

    Returns:
        New dict with degraded pixel_values_videos (and all other keys
        unchanged/cloned).
    """
    bad_inputs = {}
    for key, value in multi_modal_inputs.items():
        if isinstance(value, torch.Tensor):
            bad_inputs[key] = value.clone()
        else:
            bad_inputs[key] = value

    if "pixel_values_videos" not in bad_inputs:
        # No video data — return as-is (e.g. image-only samples)
        return bad_inputs

    pixel_values = bad_inputs["pixel_values_videos"]  # (total_tokens, dim)
    grid_thw = video_grid_thw  # (num_videos, 3)

    if grid_thw is None or grid_thw.numel() == 0:
        return bad_inputs

    # pixel_values_videos is a flat tensor of all visual tokens across all
    # videos in the batch. We need to identify which tokens belong to which
    # temporal frames using video_grid_thw.
    #
    # For each video: T temporal frames, each frame has H*W spatial patches.
    # Total tokens for video i = T_i * H_i * W_i
    offset = 0
    for vid_idx in range(grid_thw.size(0)):
        t_frames = int(grid_thw[vid_idx, 0].item())
        h_patches = int(grid_thw[vid_idx, 1].item())
        w_patches = int(grid_thw[vid_idx, 2].item())
        tokens_per_frame = h_patches * w_patches
        total_tokens = t_frames * tokens_per_frame

        if total_tokens == 0 or t_frames == 0:
            continue

        # Select which frames to degrade
        if mode == "blur":
            num_blur = max(1, int(t_frames * blur_fraction))
            blur_indices = torch.randperm(t_frames, device=pixel_values.device)[:num_blur]
            for fi in blur_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = _gaussian_blur_frames(
                    pixel_values[start:end].unsqueeze(0), sigma=blur_sigma
                ).squeeze(0)

        elif mode == "drop":
            num_drop = max(1, int(t_frames * drop_fraction))
            drop_indices = torch.randperm(t_frames, device=pixel_values.device)[:num_drop]
            for fi in drop_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = 0.0

        elif mode == "blur_and_drop":
            # Split frames into blur set and drop set (disjoint)
            perm = torch.randperm(t_frames, device=pixel_values.device)
            num_blur = max(1, int(t_frames * blur_fraction * 0.5))
            num_drop = max(1, int(t_frames * drop_fraction * 0.5))
            # Ensure we don't exceed available frames
            num_blur = min(num_blur, t_frames)
            num_drop = min(num_drop, max(0, t_frames - num_blur))
            blur_indices = perm[:num_blur]
            drop_indices = perm[num_blur:num_blur + num_drop]

            for fi in blur_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = _gaussian_blur_frames(
                    pixel_values[start:end].unsqueeze(0), sigma=blur_sigma
                ).squeeze(0)
            for fi in drop_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = 0.0
        else:
            raise ValueError(f"Unknown bad-video mode: {mode}")

        offset += total_tokens

    bad_inputs["pixel_values_videos"] = pixel_values
    return bad_inputs
