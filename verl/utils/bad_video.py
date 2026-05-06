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
  - "shuffle": Randomly select a fraction of temporal frames and shuffle
    their temporal positions, leaving the rest unchanged.

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


def _gaussian_blur_2d(
    image: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply 2D Gaussian blur to a (C, H, W) tensor."""
    C, H, W = image.shape
    kernel_size = max(int(math.ceil(sigma * 3)) | 1, 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_1d = _gaussian_kernel_1d(sigma, kernel_size, image.device, image.dtype)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (K, K)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, -1, -1)
    pad = kernel_size // 2
    # (C, H, W) -> (1, C, H, W) for grouped conv2d
    x = image.unsqueeze(0)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(x_padded, kernel_2d, groups=C)
    return blurred.squeeze(0)  # (C, H, W)


def _blur_patchified_frame(
    frame_patches: torch.Tensor,
    h_patches: int,
    w_patches: int,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Apply true 2D spatial Gaussian blur to one temporal frame's patchified pixels.

    The Qwen processor patchifies video frames into a flat tensor where each row
    is a flattened spatiotemporal patch: (temporal_patch_size, in_channels, patch_size, patch_size).
    Rows are laid out in (H_patches, W_patches) order.

    This function reconstructs the spatial image, applies 2D blur, then
    re-patchifies to the original shape.

    Args:
        frame_patches: (H_patches * W_patches, patch_dim) patchified pixels for
            one temporal frame.
        h_patches: Number of spatial patches in height.
        w_patches: Number of spatial patches in width.
        patch_size: Spatial patch size (pixels per patch side).
        temporal_patch_size: Number of raw frames grouped per temporal patch.
        in_channels: Number of color channels (typically 3).
        sigma: Gaussian blur sigma.

    Returns:
        Blurred patches with the same shape as input.
    """
    num_patches, patch_dim = frame_patches.shape
    assert num_patches == h_patches * w_patches

    # Reconstruct spatial form.
    # Qwen processor layout per patch row: (temporal_patch_size, in_channels, patch_size, patch_size)
    # Patches ordered as (H_patches, W_patches) in row-major.
    frame = frame_patches.view(
        h_patches, w_patches, temporal_patch_size, in_channels, patch_size, patch_size
    )
    # -> (temporal_patch_size * in_channels, H_patches * patch_size, W_patches * patch_size)
    frame = frame.permute(2, 3, 0, 4, 1, 5).reshape(
        temporal_patch_size * in_channels,
        h_patches * patch_size,
        w_patches * patch_size,
    )

    # Apply real 2D spatial Gaussian blur.
    frame = _gaussian_blur_2d(frame, sigma)

    # Re-patchify: reverse the reshape.
    frame = frame.view(
        temporal_patch_size, in_channels, h_patches, patch_size, w_patches, patch_size
    )
    frame = frame.permute(2, 4, 0, 1, 3, 5).reshape(num_patches, patch_dim)
    return frame


def construct_bad_video_inputs(
    multi_modal_inputs: dict[str, torch.Tensor],
    video_grid_thw: Optional[torch.Tensor],
    mode: str = "blur",
    blur_sigma: float = 5.0,
    blur_fraction: float = 0.5,
    drop_fraction: float = 0.5,
    shuffle_fraction: float = 0.2,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    in_channels: int = 3,
    keyframe_indices: Optional[list[list[int]]] = None,
    use_keyframe_mask: bool = False,
    debug: bool = False,
    sample_ids: Optional[list[str]] = None,
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
        mode: Degradation mode — "blur", "drop", "blur_and_drop", or "shuffle".
        blur_sigma: Gaussian blur sigma.
        blur_fraction: Fraction of frames to blur.
        drop_fraction: Fraction of frames to drop (zero out).
        shuffle_fraction: Fraction of frames to temporally shuffle (for "shuffle" mode).
        patch_size: Spatial patch size from the vision config (default 14).
        temporal_patch_size: Temporal patch size from the vision config (default 2).
        in_channels: Number of input channels from the vision config (default 3).

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

        # Deterministic keyframe masking path (drop/blackout selected frames only).
        if use_keyframe_mask:
            raw_indices = keyframe_indices[vid_idx] if (keyframe_indices is not None and vid_idx < len(keyframe_indices)) else []
            valid_indices = sorted({int(fi) for fi in raw_indices if 0 <= int(fi) < t_frames})
            if debug:
                sid = sample_ids[vid_idx] if sample_ids is not None and vid_idx < len(sample_ids) else str(vid_idx)
                print(
                    f"[sdpo_v_keyframe_mask] sample={sid} mode={mode} "
                    f"ground_frame_indices={list(raw_indices)} valid_corrupted_frame_indices={valid_indices} "
                    f"t_frames={t_frames} tokens_per_frame={tokens_per_frame}"
                )
            for fi in valid_indices:
                start = offset + fi * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = 0.0
        elif mode == "blur":
            num_blur = max(1, int(t_frames * blur_fraction))
            blur_indices = torch.randperm(t_frames, device=pixel_values.device)[:num_blur]
            for fi in blur_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = _blur_patchified_frame(
                    pixel_values[start:end],
                    h_patches=h_patches,
                    w_patches=w_patches,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    in_channels=in_channels,
                    sigma=blur_sigma,
                )

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
                pixel_values[start:end] = _blur_patchified_frame(
                    pixel_values[start:end],
                    h_patches=h_patches,
                    w_patches=w_patches,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    in_channels=in_channels,
                    sigma=blur_sigma,
                )
            for fi in drop_indices:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = 0.0

        elif mode == "shuffle":
            # Randomly select a fraction of temporal frames and shuffle their
            # temporal positions among themselves.  Unselected frames stay put.
            # Need at least 2 frames to perform a meaningful shuffle.
            if t_frames < 2:
                offset += total_tokens
                continue

            num_shuffle = max(2, int(t_frames * shuffle_fraction))
            num_shuffle = min(num_shuffle, t_frames)

            # Pick which frame indices participate in the shuffle
            selected = torch.randperm(t_frames, device=pixel_values.device)[:num_shuffle]
            selected_sorted = selected.sort().values  # original positions in order

            # Extract pixel data for the selected frames
            frame_data = []
            for fi in selected_sorted:
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                frame_data.append(pixel_values[start:end].clone())

            # Generate a non-identity permutation (retry if identity)
            perm = torch.randperm(num_shuffle, device=pixel_values.device)
            if num_shuffle > 1:
                identity = torch.arange(num_shuffle, device=pixel_values.device)
                max_retries = 10
                for _ in range(max_retries):
                    if not torch.equal(perm, identity):
                        break
                    perm = torch.randperm(num_shuffle, device=pixel_values.device)

            # Write shuffled frame data back into the original positions
            for i, fi in enumerate(selected_sorted):
                start = offset + int(fi.item()) * tokens_per_frame
                end = start + tokens_per_frame
                pixel_values[start:end] = frame_data[perm[i]]

        else:
            raise ValueError(f"Unknown bad-video mode: {mode}")

        offset += total_tokens

    bad_inputs["pixel_values_videos"] = pixel_values
    return bad_inputs
