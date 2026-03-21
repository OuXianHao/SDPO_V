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
Implement Actor
"""

import os
import time
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import compute_grpo_loss, compute_sdpo_logit_loss
from ...utils.dataset import process_image, process_video
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        processor: Optional[Any] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_fps: float = 2.0,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.is_trainable_actor = self.actor_optimizer is not None
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_fps = video_fps
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        if self.is_trainable_actor and self.rank == 0:
            print(f"[actor] selected loss_mode={self.config.loss_mode}")
            if self.config.loss_mode == "sdpo_logit":
                print(
                    f"[actor] sdpo settings: topk={self.config.sdpo_topk}, "
                    f"divergence={self.config.sdpo_divergence}, use_tail={self.config.sdpo_use_tail}, "
                    f"approx_mode={self.config.sdpo_approx_mode}"
                )

        self._sdpo_backward_shape_debug = os.getenv("SDPO_BACKWARD_SHAPE_DEBUG", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._sdpo_update_debug = os.getenv("EASYR1_DEBUG_SDPO_UPDATE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _render_teacher_prompt_text(self, content_text: str) -> str:
        format_prompt = self.config.teacher_format_prompt
        if format_prompt is None or format_prompt == "":
            return content_text

        from jinja2 import Template
        template = Template(format_prompt.strip())
        return template.render(content=content_text)

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _forward_response_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute response-position logits.

        In SDPO top-k mode (when NOT using padding_free), this uses the
        ``response_only_logits`` parameter supported by the Qwen2-VL / Qwen3-VL
        monkey-patched forward.  That flag tells the model to apply ``lm_head``
        only on the response-span hidden states **inside** the FSDP-managed
        forward context, avoiding both:
          * a full ``(batch, full_seq, vocab_size)`` logits tensor, and
          * the FSDP-unsafe ``self.actor_module.model(...)`` bypass that the
            old compact path used.

        Returns:
            logits: ``(batch, response_length, vocab_size)``
        """
        if self._sdpo_update_debug:
            print(
                "[RCA_FORWARD_RESP_LOGITS_ENTRY] "
                f"rank={self.rank}/{self.world_size} "
                f"padding_free={self.config.padding_free} ulysses_size={self.config.ulysses_size}"
            )
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if input_ids is None:
            input_ids = micro_batch["input_ids"]
        if attention_mask is None:
            attention_mask = micro_batch["attention_mask"]
        if position_ids is None:
            position_ids = micro_batch["position_ids"]
        batch_size, seqlen = input_ids.shape
        if position_ids.dim() == 3:  # qwen2vl/qwen3vl mrope
            position_ids = position_ids.transpose(0, 1)

        if multi_modal_inputs is None:
            mm_inputs = defaultdict(list)
            if "multi_modal_inputs" in micro_batch:
                mm_inputs = batch_collate(micro_batch["multi_modal_inputs"])
                mm_inputs = {key: torch.cat(value, dim=0) for key, value in mm_inputs.items()}
            else:
                mm_inputs = {}
        else:
            mm_inputs = multi_modal_inputs

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            pad_size = 0
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **mm_inputs,
                use_cache=False,
            )
            logits_rmpad: torch.Tensor = output.logits.squeeze(0) / temperature
            if self.config.ulysses_size > 1:
                logits_rmpad = gather_outputs_and_unpad(
                    logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False
                )
            full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            # Decide whether to use the FSDP-safe response-only lm_head path.
            sdpo_topk_mode = (
                getattr(self.config, "loss_mode", None) == "sdpo_logit"
                and getattr(self.config, "sdpo_approx_mode", "topk") in ("topk", "student_topk_tail")
            )
            force_full_logits = os.getenv("SDPO_FORCE_FULL_LOGITS", "0").strip().lower() in {"1", "true", "yes", "on"}
            use_response_only = sdpo_topk_mode and (not force_full_logits)

            if use_response_only:
                # Pass response_only_logits=response_length through the top-level
                # FSDP-wrapped module.  The monkey-patched Qwen forward will:
                #   1. Run the full backbone (embeddings + transformer blocks).
                #   2. Slice hidden_states to the response span.
                #   3. Apply lm_head only on that span.
                # This keeps everything inside the FSDP forward context
                # (parameters properly gathered/sharded) and avoids the massive
                # (batch, full_seq, vocab_size) logits tensor.

                # --- Live-path diagnostic: verify patched forward is active ---
                if self.rank == 0:
                    # Unwrap FSDP to get the actual model class
                    unwrapped = self.actor_module
                    while hasattr(unwrapped, "module"):
                        unwrapped = unwrapped.module
                    model_cls = type(unwrapped)
                    fwd_fn = getattr(model_cls, "forward", None)
                    fwd_qualname = getattr(fwd_fn, "__qualname__", "UNKNOWN") if fwd_fn else "NONE"
                    fwd_module = getattr(fwd_fn, "__module__", "UNKNOWN") if fwd_fn else "NONE"
                    print(
                        f"[sdpo-patch-check] model_class={model_cls.__name__} "
                        f"forward.__qualname__={fwd_qualname} "
                        f"forward.__module__={fwd_module} "
                        f"response_only_logits={response_length}"
                    )

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **mm_inputs,
                    use_cache=False,
                    response_only_logits=response_length,
                )
                logits = output.logits

                # Verify the model actually honored response_only_logits.
                # If it didn't (patch not applied), the logits will be
                # (batch, full_seq_len, vocab) instead of (batch, response_length, vocab).
                # Do NOT silently fall back — that re-enters the dangerous
                # full-sequence backward path that causes SplitWithSizesBackward0.
                if logits.size(1) != response_length:
                    raise RuntimeError(
                        f"[SDPO FATAL] response_only_logits={response_length} was NOT "
                        f"honored by the live model forward. Returned logits shape "
                        f"{tuple(logits.shape)} (expected seq_dim={response_length}). "
                        f"The VL model forward patch is NOT active. "
                        f"Ensure apply_vl_forward_patch() is called in "
                        f"_build_model_optimizer() BEFORE model construction. "
                        f"Do NOT fall back to full-sequence logits — that path "
                        f"causes SplitWithSizesBackward0 gradient failures."
                    )
                logits = logits / temperature
                if self.rank == 0:
                    print(
                        f"[sdpo-path-ok] response_only_logits HONORED. "
                        f"logits_shape={tuple(logits.shape)} "
                        f"(expected response_length={response_length})"
                    )
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **mm_inputs,
                    use_cache=False,
                )
                full_logits = output.logits
                logits = full_logits[:, -response_length - 1 : -1, :] / temperature
                if self._sdpo_backward_shape_debug and self.rank == 0:
                    print(
                        "[sdpo-shape-debug] response_only_logits=False "
                        f"full_logits_shape={tuple(full_logits.shape)} "
                        f"response_logits_shape={tuple(logits.shape)}"
                    )
        return logits

    def _build_teacher_message_content(self, prompt_text: str, multi_modal_data: Optional[dict[str, Any]]) -> Any:
        if multi_modal_data is None:
            return prompt_text

        if "videos" in multi_modal_data:
            marker = "<video>"
            media_type = "video"
        elif "images" in multi_modal_data:
            marker = "<image>"
            media_type = "image"
        else:
            return prompt_text

        content_list = []
        for idx, content in enumerate(prompt_text.split(marker)):
            if idx != 0:
                content_list.append({"type": media_type})
            if content:
                content_list.append({"type": "text", "text": content})
        return content_list

    def _build_teacher_inputs(
        self, model_inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.processor is None:
            raise RuntimeError("SDPO-T multimodal teacher reconstruction requires a processor on actor worker.")

        device = model_inputs["input_ids"].device
        responses = model_inputs["responses"]
        response_mask = model_inputs["response_mask"]
        raw_prompt_texts = model_inputs["raw_prompt_text"]
        feedback_texts = model_inputs.get("feedback_text", None)
        teacher_prompt_texts = model_inputs.get("teacher_prompt_text", None)
        sdpo_feedback_mode = getattr(self.config, "sdpo_feedback_mode", "scalar_text")
        batch_multi_modal_data = model_inputs.get("multi_modal_data", None)
        pad_token_ids = model_inputs.get("pad_token_id", None)

        teacher_sequences: list[torch.Tensor] = []
        teacher_attention_masks: list[torch.Tensor] = []
        teacher_position_ids: list[torch.Tensor] = []
        teacher_multi_modal_inputs: list[dict[str, torch.Tensor]] = []

        for i in range(responses.size(0)):
            multi_modal_data = None if batch_multi_modal_data is None else batch_multi_modal_data[i]
            if teacher_prompt_texts is not None:
                teacher_prompt_text = str(teacher_prompt_texts[i])
            else:
                if sdpo_feedback_mode == "successful_rollout":
                    raise KeyError(
                        "Missing `teacher_prompt_text` for sdpo_feedback_mode='successful_rollout'. "
                        "This mode requires explicit successful-rollout teacher prompts and does not allow scalar fallback."
                    )
                raw_prompt_text = str(raw_prompt_texts[i])
                feedback_text = "" if feedback_texts is None else str(feedback_texts[i])
                teacher_content_text = f"{raw_prompt_text}\n\n[Feedback]: {feedback_text}"
                teacher_prompt_text = self._render_teacher_prompt_text(teacher_content_text)

            teacher_messages = [
                {
                    "role": "user",
                    "content": self._build_teacher_message_content(teacher_prompt_text, multi_modal_data),
                }
            ]
            teacher_prompt = self.processor.apply_chat_template(
                teacher_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            processor_inputs: dict[str, Any]
            if multi_modal_data is not None and "videos" in multi_modal_data:
                processed_videos = []
                video_fps_list = []
                for video in multi_modal_data["videos"]:
                    processed_video, video_sample_fps = process_video(
                        video,
                        self.min_pixels,
                        self.max_pixels,
                        self.video_fps,
                        return_fps=True,
                    )
                    processed_videos.append(processed_video)
                    video_fps_list.append(video_sample_fps)
                processor_inputs = dict(
                    self.processor(videos=processed_videos, text=[teacher_prompt], add_special_tokens=False, return_tensors="pt")
                )
                if "second_per_grid_ts" in self.processor.model_input_names and len(video_fps_list) > 0:
                    processor_inputs["second_per_grid_ts"] = torch.tensor(
                        [2.0 / max(float(video_sample_fps), 1e-6) for video_sample_fps in video_fps_list],
                        dtype=torch.float32,
                    )
            elif multi_modal_data is not None and "images" in multi_modal_data:
                processed_images = [process_image(image, self.min_pixels, self.max_pixels) for image in multi_modal_data["images"]]
                processor_inputs = dict(
                    self.processor(processed_images, [teacher_prompt], add_special_tokens=False, return_tensors="pt")
                )
            else:
                processor_inputs = dict(self.processor(text=[teacher_prompt], add_special_tokens=False, return_tensors="pt"))

            prompt_ids = processor_inputs.pop("input_ids")[0]
            prompt_attention = processor_inputs.pop("attention_mask")[0]

            if (
                hasattr(self.processor, "image_processor")
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ...models.transformers.qwen3_vl import get_rope_index
                else:
                    from ...models.transformers.qwen2_vl import get_rope_index

                prompt_position = get_rope_index(
                    self.processor,
                    input_ids=prompt_ids,
                    image_grid_thw=processor_inputs.get("image_grid_thw", None),
                    video_grid_thw=processor_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=processor_inputs.get("second_per_grid_ts", None),
                    attention_mask=prompt_attention,
                )
                text_position_ids = torch.arange(len(prompt_ids)).unsqueeze(0)
                prompt_position = torch.cat((text_position_ids, prompt_position), dim=0)
            else:
                prompt_position = torch.clamp(prompt_attention.cumsum(dim=0) - 1, min=0)

            sequence_ids = torch.cat([prompt_ids.to(dtype=responses.dtype), responses[i].cpu()], dim=0)
            sequence_attention = torch.cat([prompt_attention.to(dtype=response_mask.dtype), response_mask[i].cpu()], dim=0)

            response_delta = torch.arange(1, responses.size(-1) + 1)
            if prompt_position.dim() == 2:
                response_delta = response_delta.view(1, -1).expand(prompt_position.size(0), -1)
                sequence_position = torch.cat([prompt_position, prompt_position[:, -1:] + response_delta], dim=-1)
            else:
                sequence_position = torch.cat([prompt_position, prompt_position[-1:] + response_delta], dim=-1)

            for key, value in processor_inputs.items():
                processor_inputs[key] = value.cpu()

            teacher_sequences.append(sequence_ids)
            teacher_attention_masks.append(sequence_attention)
            teacher_position_ids.append(sequence_position)
            teacher_multi_modal_inputs.append(processor_inputs)

        max_length = max(sequence_ids.size(0) for sequence_ids in teacher_sequences)
        padded_input_ids = []
        padded_attention_masks = []
        padded_position_ids = []

        for i, (sequence_ids, sequence_attention, sequence_position) in enumerate(
            zip(teacher_sequences, teacher_attention_masks, teacher_position_ids)
        ):
            pad_length = max_length - sequence_ids.size(0)
            if pad_length > 0:
                if pad_token_ids is None:
                    pad_token_id = 0
                else:
                    pad_token_id = int(pad_token_ids[i])
                left_pad_ids = torch.full(
                    (pad_length,),
                    fill_value=pad_token_id,
                    dtype=responses.dtype,
                    device=sequence_ids.device,
                )
                left_pad_attention = torch.zeros((pad_length,), dtype=response_mask.dtype, device=sequence_ids.device)
                sequence_ids = torch.cat([left_pad_ids, sequence_ids], dim=0)
                sequence_attention = torch.cat([left_pad_attention, sequence_attention], dim=0)
                if sequence_position.dim() == 2:
                    left_pad_position = torch.zeros((sequence_position.size(0), pad_length), dtype=sequence_position.dtype)
                else:
                    left_pad_position = torch.zeros((pad_length,), dtype=sequence_position.dtype)
                sequence_position = torch.cat([left_pad_position, sequence_position], dim=-1)

            padded_input_ids.append(sequence_ids)
            padded_attention_masks.append(sequence_attention)
            padded_position_ids.append(sequence_position)

        teacher_input_ids = torch.stack(padded_input_ids, dim=0).to(device)
        teacher_attention_mask = torch.stack(padded_attention_masks, dim=0).to(device)
        teacher_position_ids = torch.stack(padded_position_ids, dim=0).to(device)
        teacher_multi_modal_inputs_batch = batch_collate(teacher_multi_modal_inputs)
        teacher_multi_modal_inputs_batch = {
            key: torch.cat(value, dim=0).to(device) for key, value in teacher_multi_modal_inputs_batch.items()
        }
        return teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_multi_modal_inputs_batch

    def _extract_topk_logps(
        self,
        logits: torch.Tensor,
        k: int,
        topk_indices: Optional[torch.Tensor] = None,
        detach_lse: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract top-k log-probabilities from response logits.

        Uses ``log_softmax`` + ``topk`` instead of separate ``topk`` +
        ``logsumexp`` to avoid creating two backward references to the full
        ``(B, N, V)`` logits tensor.  The dual-reference pattern
        (``topk(logits)`` + ``logsumexp(logits)``) causes
        ``SplitWithSizesBackward0`` errors under FSDP because both ops
        produce independent backward paths through the same flat-parameter
        split.  ``log_softmax`` has a fused backward kernel that creates
        only a single reference — matching the pattern used by the standard
        PPO path (``log_softmax`` + ``gather``) which is proven to work.

        Since ``log_softmax`` is monotonic, ``topk(log_softmax(x))`` returns
        the same indices as ``topk(x)``.

        Args:
            logits: ``(batch, response_len, vocab_size)`` — response logits
                (already temperature-scaled).
            k: number of top tokens.
            topk_indices: optional pre-computed indices from another
                distribution (student's indices reused for teacher).
            detach_lse: if True, detach the log_softmax result before topk
                so gradients only flow sparsely through gathered entries.
                (Kept for API compatibility; less important now that the
                dual-reference issue is resolved.)

        Returns:
            dict with ``topk_logps``, ``topk_indices``, ``logsumexp``.
        """
        _debug = os.getenv("SDPO_T_DEBUG_SHAPES", "0").strip().lower() in {"1", "true", "yes", "on"}
        _caller = "teacher" if topk_indices is not None else "student"

        vocab_size = logits.size(-1)
        k = min(max(k, 1), vocab_size)

        if _debug and self.rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=E01 _extract_topk_logps ENTRY caller={_caller} "
                f"logits shape={tuple(logits.shape)} dtype={logits.dtype} "
                f"device={logits.device} requires_grad={logits.requires_grad} "
                f"is_contiguous={logits.is_contiguous()} stride={logits.stride()} "
                f"data_ptr={logits.data_ptr()} storage_offset={logits.storage_offset()} "
                f"numel={logits.numel()} k={k}"
            )
            # Walk grad_fn chain for the input logits
            gf = logits.grad_fn
            chain = []
            for _i in range(15):
                if gf is None:
                    break
                chain.append(type(gf).__name__)
                nexts = gf.next_functions
                gf = nexts[0][0] if nexts else None
            print(f"[SDPO_T_SHAPE] step=E02 logits grad_fn_chain caller={_caller} chain={chain}")

        # Single fused backward through log_softmax — avoids the dual
        # topk + logsumexp backward references that trigger FSDP errors.
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, resp_len, vocab)

        if _debug and self.rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=E03 log_probs caller={_caller} "
                f"shape={tuple(log_probs.shape)} dtype={log_probs.dtype} "
                f"requires_grad={log_probs.requires_grad} "
                f"is_contiguous={log_probs.is_contiguous()} stride={log_probs.stride()} "
                f"grad_fn={log_probs.grad_fn}"
            )
            # Register backward hook on log_probs (full-vocab intermediate) —
            # this is the tensor between topk and FSDP.
            if log_probs.requires_grad and _caller == "student":
                def _log_probs_bwd_hook(grad):
                    print(
                        f"[SDPO_T_BACKWARD] step=B03 log_probs grad "
                        f"shape={tuple(grad.shape)} dtype={grad.dtype} "
                        f"numel={grad.numel()} "
                        f"is_contiguous={grad.is_contiguous()} stride={grad.stride()} "
                        f"has_nan={bool(grad.isnan().any())} "
                        f"has_inf={bool(grad.isinf().any())}"
                    )
                    return grad
                log_probs.register_hook(_log_probs_bwd_hook)

        if detach_lse:
            # Detach to make gradient sparse (only through gathered entries).
            log_probs_for_topk = log_probs.detach()
        else:
            log_probs_for_topk = log_probs

        if _debug and self.rank == 0 and _caller == "student":
            print(
                f"[SDPO_T_SHAPE] step=E04 log_probs_for_topk caller={_caller} "
                f"requires_grad={log_probs_for_topk.requires_grad} "
                f"is_same_as_log_probs={log_probs_for_topk.data_ptr() == log_probs.data_ptr()} "
                f"detach_lse={detach_lse}"
            )

        if topk_indices is None:
            topk_logps, topk_indices = torch.topk(log_probs_for_topk, k, dim=-1)
            if _debug and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] step=E05 topk (fresh) caller={_caller} "
                    f"topk_logps shape={tuple(topk_logps.shape)} stride={topk_logps.stride()} "
                    f"is_contiguous={topk_logps.is_contiguous()} "
                    f"requires_grad={topk_logps.requires_grad} grad_fn={topk_logps.grad_fn} "
                    f"topk_indices shape={tuple(topk_indices.shape)} dtype={topk_indices.dtype}"
                )
        else:
            topk_logps = torch.gather(log_probs_for_topk, dim=-1, index=topk_indices)
            if _debug and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] step=E05 gather caller={_caller} "
                    f"topk_logps shape={tuple(topk_logps.shape)} stride={topk_logps.stride()} "
                    f"is_contiguous={topk_logps.is_contiguous()} "
                    f"requires_grad={topk_logps.requires_grad} grad_fn={topk_logps.grad_fn} "
                    f"topk_indices shape={tuple(topk_indices.shape)}"
                )

        # Compute logsumexp for metrics compatibility (detached — no grad needed).
        with torch.no_grad():
            lse = torch.logsumexp(logits, dim=-1)  # (batch, resp_len)

        if _debug and self.rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=E06 _extract_topk_logps EXIT caller={_caller} "
                f"topk_logps shape={tuple(topk_logps.shape)} numel={topk_logps.numel()} "
                f"topk_indices shape={tuple(topk_indices.shape)} "
                f"lse shape={tuple(lse.shape)}"
            )

        return {
            "topk_logps": topk_logps,       # (batch, resp_len, k)
            "topk_indices": topk_indices,    # (batch, resp_len, k)
            "logsumexp": lse,               # (batch, resp_len)
        }

    def _compute_sdpo_logit_loss(self, model_inputs: dict[str, Any], temperature: float) -> tuple[torch.Tensor, dict[str, float]]:
        if self._sdpo_update_debug:
            print(
                "[RCA_SDPO_LOSS_ENTRY] "
                f"rank={self.rank}/{self.world_size} "
                f"responses_shape={tuple(model_inputs['responses'].shape)} "
                f"input_ids_shape={tuple(model_inputs['input_ids'].shape)} "
                f"attention_mask_shape={tuple(model_inputs['attention_mask'].shape)} "
                f"position_ids_shape={tuple(model_inputs['position_ids'].shape)}"
            )
            print(
                "[RCA_SDPO_LOSS_CONFIG_PATH] "
                f"rank={self.rank}/{self.world_size} "
                f"loss_mode={self.config.loss_mode} approx_mode={self.config.sdpo_approx_mode} "
                f"divergence={self.config.sdpo_divergence} use_tail={self.config.sdpo_use_tail} "
                f"feedback_mode={getattr(self.config, 'sdpo_feedback_mode', 'N/A')}"
            )
        response_mask = (
            model_inputs["response_mask"].bool()
            & model_inputs["response_token_mask"].bool()
            & model_inputs["sdpo_valid_mask"].bool()
        )
        valid_sample_mask = response_mask.any(dim=-1)
        num_valid_samples = int(valid_sample_mask.sum().item())
        num_total_samples = int(response_mask.shape[0])
        if response_mask.shape != model_inputs["responses"].shape:
            raise ValueError("response_token_mask must align with sampled responses shape.")

        # Robustness: if every token is masked out (e.g. all groups skipped), return a safe zero loss.
        if not torch.any(response_mask):
            trainable_param = next((p for p in self.actor_module.parameters() if p.requires_grad), None)
            if trainable_param is None:
                raise RuntimeError("No trainable actor parameters found for zero sdpo loss fallback.")
            zero_loss = trainable_param.sum() * 0.0
            return zero_loss, {"sdpo_all_masked_batch": 1.0, "sdpo_valid_token_count": 0.0}

        build_start = time.perf_counter()
        teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_multi_modal_inputs = self._build_teacher_inputs(
            model_inputs
        )
        build_elapsed_ms = (time.perf_counter() - build_start) * 1000.0

        if self._sdpo_backward_shape_debug and self.rank == 0:
            print(
                "[sdpo-shape-debug] teacher_input_ids/attention/position/responses shapes: "
                f"{tuple(teacher_input_ids.shape)} / {tuple(teacher_attention_mask.shape)} / "
                f"{tuple(teacher_position_ids.shape)} / {tuple(model_inputs['responses'].shape)}"
            )
            print(
                "[sdpo-shape-debug] student_input_ids/attention/position shapes: "
                f"{tuple(model_inputs['input_ids'].shape)} / {tuple(model_inputs['attention_mask'].shape)} / "
                f"{tuple(model_inputs['position_ids'].shape)}"
            )
            if "multi_modal_inputs" in model_inputs:
                student_mm_inputs = batch_collate(model_inputs["multi_modal_inputs"])
                student_mm_shapes = {k: [tuple(t.shape) for t in v] for k, v in student_mm_inputs.items()}
                print(f"[sdpo-shape-debug] student multimodal tensor shapes: {student_mm_shapes}")
            teacher_mm_shapes = {k: tuple(v.shape) for k, v in teacher_multi_modal_inputs.items()}
            print(f"[sdpo-shape-debug] teacher multimodal tensor shapes: {teacher_mm_shapes}")

        approx_mode = self.config.sdpo_approx_mode
        topk_mode = approx_mode in ("topk", "student_topk_tail")
        topk_k = self.config.sdpo_topk
        detach_lse = os.getenv("SDPO_DETACH_LSE", "0").strip().lower() in {"1", "true", "yes", "on"}
        detect_anomaly = os.getenv("SDPO_DETECT_ANOMALY", "0").strip().lower() in {"1", "true", "yes", "on"}
        _debug_shapes = os.getenv("SDPO_T_DEBUG_SHAPES", "0").strip().lower() in {"1", "true", "yes", "on"}

        if _debug_shapes and self.rank == 0:
            print(
                f"[SDPO_T_BRANCH] _compute_sdpo_logit_loss ENTRY "
                f"approx_mode={approx_mode} topk_mode={topk_mode} "
                f"use_tail={self.config.sdpo_use_tail} divergence={self.config.sdpo_divergence} "
                f"topk_k={topk_k} detach_lse={detach_lse} "
                f"response_mask shape={tuple(response_mask.shape)} "
                f"valid_tokens={int(response_mask.sum().item())} "
                f"num_valid_samples={num_valid_samples}/{num_total_samples}"
            )
            # FSDP flat parameter diagnostics
            if isinstance(self.actor_module, FSDP):
                for i, (name, param) in enumerate(self.actor_module.named_parameters()):
                    if i < 3 or "lm_head" in name or "embed" in name:
                        print(
                            f"[SDPO_T_SHAPE] FSDP_param name={name} "
                            f"shape={tuple(param.shape)} numel={param.numel()} "
                            f"dtype={param.dtype} requires_grad={param.requires_grad}"
                        )
                    if i == 3:
                        print(f"[SDPO_T_SHAPE] FSDP_param ... (skipping middle params)")
            else:
                print(f"[SDPO_T_SHAPE] actor_module is NOT FSDP-wrapped: {type(self.actor_module).__name__}")

        # ---- Teacher forward FIRST (no grad) ----
        # Run teacher before student so that FSDP's internal execution-order
        # tracking state reflects the student forward (which is the one that
        # backward will follow).  Running student last ensures FSDP's
        # pre/post-forward bookkeeping is consistent with the backward pass.
        teacher_forward_start = time.perf_counter()
        with torch.no_grad():
            teacher_logits = self._forward_response_logits(
                model_inputs,
                temperature=temperature,
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                position_ids=teacher_position_ids,
                multi_modal_inputs=teacher_multi_modal_inputs,
            )
        teacher_forward_elapsed_ms = (time.perf_counter() - teacher_forward_start) * 1000.0

        # ---- Student forward LAST (with grad) ----
        student_forward_start = time.perf_counter()
        student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
        student_forward_elapsed_ms = (time.perf_counter() - student_forward_start) * 1000.0

        # --- Debug: teacher/student response span alignment ---
        responses = model_inputs["responses"]
        expected_resp_len = responses.size(-1)
        if self.rank == 0:
            student_seq = model_inputs["input_ids"].size(1)
            teacher_seq = teacher_input_ids.size(1)
            print(
                f"[sdpo-align-debug] student_seq_len={student_seq} teacher_seq_len={teacher_seq} "
                f"response_length={expected_resp_len} "
                f"student_logits_shape={tuple(student_logits.shape)} "
                f"teacher_logits_shape={tuple(teacher_logits.shape)} "
                f"responses_shape={tuple(responses.shape)}"
            )

        if teacher_logits.shape[:2] != student_logits.shape[:2]:
            raise ValueError(
                f"Teacher and student response spans must align for sdpo_logit. "
                f"teacher_logits={tuple(teacher_logits.shape)}, "
                f"student_logits={tuple(student_logits.shape)}, "
                f"expected response_length={expected_resp_len}. "
                f"This usually means the model did not honor response_only_logits. "
                f"Check that apply_vl_forward_patch() was called."
            )

        if _debug_shapes and self.rank == 0:
            print(
                f"[SDPO_T_SHAPE] student_logits shape={tuple(student_logits.shape)} "
                f"dtype={student_logits.dtype} requires_grad={student_logits.requires_grad} "
                f"is_contiguous={student_logits.is_contiguous()} stride={student_logits.stride()} "
                f"numel={student_logits.numel()} grad_fn={student_logits.grad_fn}"
            )
            print(
                f"[SDPO_T_SHAPE] teacher_logits shape={tuple(teacher_logits.shape)} "
                f"dtype={teacher_logits.dtype} requires_grad={teacher_logits.requires_grad} "
                f"is_contiguous={teacher_logits.is_contiguous()} stride={teacher_logits.stride()} "
                f"numel={teacher_logits.numel()}"
            )

        if topk_mode:
            if _debug_shapes and self.rank == 0:
                print(
                    f"[SDPO_T_BRANCH] entering topk_mode path "
                    f"approx_mode={approx_mode} use_tail={self.config.sdpo_use_tail}"
                )

            # Pre-compute top-k in the actor, aligned with original SDPO design.
            # Student selects top-k indices; teacher is gathered at the SAME indices.
            student_topk = self._extract_topk_logps(student_logits, k=topk_k, detach_lse=detach_lse)

            # Register backward hooks for diagnostics: log gradient shapes
            # at multiple points in the backward chain.
            if self.rank == 0:
                def _grad_hook(grad, name="student_topk_logps"):
                    print(
                        f"[SDPO_T_HOOK] {name} grad shape={tuple(grad.shape)} "
                        f"dtype={grad.dtype} numel={grad.numel()} "
                        f"has_nan={bool(grad.isnan().any())} "
                        f"has_inf={bool(grad.isinf().any())} "
                        f"detach_lse={detach_lse}"
                    )
                    return grad
                student_topk["topk_logps"].register_hook(_grad_hook)

            # Also hook student_logits to catch the gradient just before it
            # flows into the model (where SplitWithSizesBackward0 lives).
            if _debug_shapes and self.rank == 0 and student_logits.requires_grad:
                def _student_logits_grad_hook(grad):
                    print(
                        f"[SDPO_T_BACKWARD] student_logits grad shape={tuple(grad.shape)} "
                        f"dtype={grad.dtype} numel={grad.numel()} "
                        f"is_contiguous={grad.is_contiguous()} stride={grad.stride()} "
                        f"has_nan={bool(grad.isnan().any())} "
                        f"has_inf={bool(grad.isinf().any())}"
                    )
                    return grad
                student_logits.register_hook(_student_logits_grad_hook)

            with torch.no_grad():
                teacher_topk = self._extract_topk_logps(
                    teacher_logits, k=topk_k, topk_indices=student_topk["topk_indices"]
                )

            if _debug_shapes and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] PRE_LOSS student_topk_logps shape={tuple(student_topk['topk_logps'].shape)} "
                    f"teacher_topk_logps shape={tuple(teacher_topk['topk_logps'].shape)} "
                    f"student requires_grad={student_topk['topk_logps'].requires_grad} "
                    f"teacher requires_grad={teacher_topk['topk_logps'].requires_grad}"
                )

            # Enable anomaly detection if requested — this prints the forward
            # op that created the failing backward node.
            with torch.autograd.set_detect_anomaly(detect_anomaly):
                sdpo_loss, sdpo_metrics = compute_sdpo_logit_loss(
                    student_topk_logps=student_topk["topk_logps"],
                    teacher_topk_logps=teacher_topk["topk_logps"],
                    response_mask=response_mask,
                    topk=topk_k,
                    divergence=self.config.sdpo_divergence,
                    use_tail=self.config.sdpo_use_tail,
                    approx_mode=approx_mode,
                    # Pass full logits detached for metrics only (no grad).
                    student_logits_for_metrics=student_logits.detach(),
                    teacher_logits_for_metrics=teacher_logits.detach(),
                )

            if _debug_shapes and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] POST_LOSS sdpo_loss shape={tuple(sdpo_loss.shape)} "
                    f"dtype={sdpo_loss.dtype} requires_grad={sdpo_loss.requires_grad} "
                    f"grad_fn={sdpo_loss.grad_fn} value={sdpo_loss.item():.6f}"
                )
        else:
            # full_vocab mode — pass raw logits to the loss (gradient flows
            # through the full vocab dimension; use only for debugging).
            with torch.autograd.set_detect_anomaly(detect_anomaly):
                sdpo_loss, sdpo_metrics = compute_sdpo_logit_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    response_mask=response_mask,
                    topk=topk_k,
                    divergence=self.config.sdpo_divergence,
                    use_tail=self.config.sdpo_use_tail,
                    approx_mode=approx_mode,
                )

        metrics = {f"sdpo/{k}": v for k, v in sdpo_metrics.items()}
        if self._sdpo_update_debug:
            print(
                "[sdpo-update-debug] "
                f"rank={self.rank}/{self.world_size} total_samples={num_total_samples} valid_samples={num_valid_samples} "
                f"response_mask_shape={tuple(response_mask.shape)} response_mask_numel={response_mask.numel()} "
                f"sdpo_valid_mask_shape={tuple(model_inputs['sdpo_valid_mask'].shape)} "
                f"response_token_mask_shape={tuple(model_inputs['response_token_mask'].shape)} "
                f"teacher_input_shape={tuple(teacher_input_ids.shape)} teacher_input_numel={teacher_input_ids.numel()} "
                f"teacher_logits_shape={tuple(teacher_logits.shape)} teacher_logits_numel={teacher_logits.numel()} "
                f"student_logits_shape={tuple(student_logits.shape)} student_logits_numel={student_logits.numel()} "
                f"build_teacher_ms={build_elapsed_ms:.2f} teacher_forward_ms={teacher_forward_elapsed_ms:.2f} "
                f"student_forward_ms={student_forward_elapsed_ms:.2f}"
            )
        return sdpo_loss, metrics

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        if not self.is_trainable_actor:
            raise RuntimeError("update_policy is only valid for trainable actor instances, not reference policy.")

        self.actor_module.train()

        if "temperature" not in data.meta_info:
            raise KeyError(
                "Missing meta_info[\"temperature\"] in update_policy; "
                "ray_trainer must set it from config.worker.rollout.temperature before update_actor."
            )
        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        if self.config.loss_mode == "grpo_on_policy":
            select_keys.extend(["old_log_probs", "advantages"])
        elif self.config.loss_mode == "sdpo_logit":
            select_keys.extend(["sdpo_valid_mask", "response_token_mask"])
            non_tensor_select_keys.extend(
                ["raw_prompt_text", "prompt_text", "feedback_text", "teacher_prompt_text", "multi_modal_data", "pad_token_id"]
            )
        else:
            raise ValueError(f"Unknown actor.loss_mode: {self.config.loss_mode}")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    if self.config.loss_mode == "grpo_on_policy":
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        loss, grpo_metrics = compute_grpo_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                        loss.backward()
                        append_to_dict(metrics, {f"grpo/{k}": v for k, v in grpo_metrics.items()})
                    elif self.config.loss_mode == "sdpo_logit":
                        if self._sdpo_update_debug:
                            valid_samples = int(model_inputs["sdpo_valid_mask"].bool().any(dim=-1).sum().item())
                            total_samples = int(model_inputs["sdpo_valid_mask"].shape[0])
                            print(
                                "[sdpo-update-debug] "
                                f"rank={self.rank}/{self.world_size} micro_batch_samples={total_samples} "
                                f"valid_samples={valid_samples} invalid_samples={total_samples - valid_samples} "
                                f"input_ids_shape={tuple(model_inputs['input_ids'].shape)} "
                                f"attention_mask_shape={tuple(model_inputs['attention_mask'].shape)} "
                                f"position_ids_shape={tuple(model_inputs['position_ids'].shape)}"
                            )
                        loss, sdpo_metrics = self._compute_sdpo_logit_loss(model_inputs, temperature=temperature)
                        _debug_shapes_bwd = os.getenv("SDPO_T_DEBUG_SHAPES", "0").strip().lower() in {"1", "true", "yes", "on"}
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                        if _debug_shapes_bwd and self.rank == 0:
                            print(
                                f"[SDPO_T_SHAPE] step=S01 SCALED_LOSS "
                                f"shape={tuple(loss.shape)} dtype={loss.dtype} "
                                f"requires_grad={loss.requires_grad} "
                                f"grad_fn={loss.grad_fn} value={loss.item():.6f}"
                            )
                            # Walk the full grad_fn chain from the scaled loss
                            gf = loss.grad_fn
                            for _step_i in range(30):
                                if gf is None:
                                    break
                                nf = gf.next_functions
                                nf_info = []
                                for f, idx in (nf if nf else []):
                                    nf_info.append(f"{type(f).__name__}[{idx}]" if f else f"None[{idx}]")
                                print(
                                    f"[SDPO_T_SHAPE] step=S01_chain[{_step_i}] "
                                    f"node={type(gf).__name__} next={nf_info}"
                                )
                                # Follow first branch
                                gf = nf[0][0] if nf else None
                            # Hook on scaled loss
                            if loss.requires_grad:
                                def _scaled_loss_hook(grad):
                                    print(
                                        f"[SDPO_T_BACKWARD] step=B00 scaled_loss grad "
                                        f"shape={tuple(grad.shape)} dtype={grad.dtype} "
                                        f"numel={grad.numel()}"
                                    )
                                    return grad
                                loss.register_hook(_scaled_loss_hook)
                            print(f"[SDPO_T_BACKWARD] step=B_START calling loss.backward() NOW")
                        if self._sdpo_backward_shape_debug:
                            with torch.autograd.detect_anomaly(check_nan=True):
                                loss.backward()
                        else:
                            loss.backward()
                        if _debug_shapes_bwd and self.rank == 0:
                            print(f"[SDPO_T_BACKWARD] step=B_END loss.backward() completed successfully")
                        append_to_dict(metrics, sdpo_metrics)
                    else:
                        raise ValueError(f"Unknown actor.loss_mode: {self.config.loss_mode}")

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
