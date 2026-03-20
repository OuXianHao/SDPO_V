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
from ...trainer.core_algos import compute_grpo_loss, compute_sdpo_logit_loss, compute_sdpo_topk_loss_preextracted
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
            # --- diagnostic: model class and forward implementation --------
            print(f"[actor-diag] actor_module class: {type(actor_module).__name__}")
            _inner = getattr(actor_module, "module", actor_module)  # unwrap FSDP if needed
            print(f"[actor-diag] inner model class: {type(_inner).__name__}")
            if hasattr(_inner, "model"):
                print(f"[actor-diag] base model class: {type(_inner.model).__name__}")
            if hasattr(_inner, "lm_head"):
                print(f"[actor-diag] lm_head class: {type(_inner.lm_head).__name__}")
            print(
                f"[actor-diag] padding_free={self.config.padding_free} "
                f"ulysses_size={self.config.ulysses_size}"
            )
            if self.config.loss_mode == "sdpo_logit":
                _will_use_local_topk = (
                    self.config.padding_free
                    and self.config.ulysses_size > 1
                    and self.config.sdpo_approx_mode != "full_vocab"
                )
                print(
                    f"[actor] sdpo settings: topk={self.config.sdpo_topk}, "
                    f"divergence={self.config.sdpo_divergence}, use_tail={self.config.sdpo_use_tail}, "
                    f"approx_mode={self.config.sdpo_approx_mode}"
                )
                print(
                    f"[actor-diag] SDPO student forward path: "
                    f"{'ULYSSES_LOCAL_TOPK (top-k before gather)' if _will_use_local_topk else 'FULL_VOCAB (standard logits gather)'}"
                )
                if _will_use_local_topk:
                    print(
                        f"[actor-diag] Ulysses gather size: (total_nnz/sp, {self.config.sdpo_topk}) "
                        f"instead of (total_nnz/sp, vocab_size)"
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
                # Keep gradient local scale to avoid expanding grads by sp_world_size for full-vocab logits.
                logits_rmpad = gather_outputs_and_unpad(
                    logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False
                )
            full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **mm_inputs,
                use_cache=False,
            )
            logits = output.logits[:, -response_length - 1 : -1, :] / temperature
        return logits

    def _forward_student_topk_ulysses(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Student forward for SDPO in padding-free + Ulysses mode.

        Computes log_softmax and top-k **locally** on each SP rank BEFORE the
        Ulysses all-gather, then gathers only the compact top-k results
        (size k, e.g. 100) instead of full-vocab logits (size V ≈ 151936).
        This avoids the backward crash caused by gathering full-vocab tensors.

        Returns:
            student_topk_log_probs: (batch, response_len, k) — with gradient
            student_topk_indices:   (batch, response_len, k) — int64, no grad
            student_tail_log_prob:  (batch, response_len)    — with gradient
        """
        assert self.config.padding_free, "This method requires padding_free=True"
        assert self.config.ulysses_size > 1, "This method requires ulysses_size > 1"

        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        batch_size, seqlen = input_ids.shape

        if position_ids.dim() == 3:  # qwen2vl/qwen3vl mrope
            position_ids = position_ids.transpose(0, 1)

        mm_inputs: dict[str, torch.Tensor] = {}
        if "multi_modal_inputs" in micro_batch:
            mm_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            mm_inputs = {key: torch.cat(value, dim=0) for key, value in mm_inputs.items()}

        # --- unpad --------------------------------------------------------
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
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

        # --- ulysses pad + slice ------------------------------------------
        pad_size = 0
        if self.config.ulysses_size > 1:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
            )

        # --- model forward (local shard) ---------------------------------
        output = self.actor_module(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            **mm_inputs,
            use_cache=False,
        )
        logits_local = output.logits.squeeze(0)  # (L, V)  — L = total_nnz/sp + pad
        logits_local = logits_local / temperature

        # --- local log_softmax + top-k (BEFORE gather) --------------------
        log_probs_local = torch.nn.functional.log_softmax(logits_local, dim=-1)  # (L, V)
        topk_log_probs_local, topk_indices_local = torch.topk(
            log_probs_local, k=min(topk, log_probs_local.size(-1)), dim=-1
        )  # both (L, k)

        # tail = log(1 - sum(exp(topk_log_probs)))
        topk_sum = topk_log_probs_local.exp().sum(dim=-1)  # (L,)
        tail_log_prob_local = torch.log((1.0 - topk_sum).clamp_min(1e-12))  # (L,)

        if self._sdpo_update_debug and self.rank == 0:
            print(
                f"[sdpo-ulysses-local-topk] rank={self.rank}/{self.world_size} "
                f"logits_local={tuple(logits_local.shape)} topk_local={tuple(topk_log_probs_local.shape)} "
                f"tail_local={tuple(tail_log_prob_local.shape)} pad_size={pad_size}"
            )

        # --- gather only top-k across SP ranks ---------------------------
        # topk_log_probs: (L, k) → (total_nnz, k)  — WITH gradient
        topk_log_probs_gathered = gather_outputs_and_unpad(
            topk_log_probs_local, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False,
        )
        # topk_indices: (L, k) → (total_nnz, k)  — NO gradient needed
        with torch.no_grad():
            topk_indices_gathered = gather_outputs_and_unpad(
                topk_indices_local, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False,
            )
        # tail: (L,) → (total_nnz,)  — WITH gradient
        tail_log_prob_gathered = gather_outputs_and_unpad(
            tail_log_prob_local, gather_dim=0, unpad_dim=0, padding_size=pad_size, grad_scaler=False,
        )

        # --- pad back to (batch, seqlen, ...) and slice to response ------
        topk_log_probs_full = pad_input(topk_log_probs_gathered, indices, batch_size, seqlen)
        topk_log_probs_response = topk_log_probs_full[:, -response_length - 1 : -1, :]  # (B, R, k)

        topk_indices_full = pad_input(topk_indices_gathered, indices, batch_size, seqlen)
        topk_indices_response = topk_indices_full[:, -response_length - 1 : -1, :].long()  # (B, R, k)

        tail_log_prob_full = pad_input(tail_log_prob_gathered.unsqueeze(-1), indices, batch_size, seqlen)
        tail_log_prob_response = tail_log_prob_full.squeeze(-1)[:, -response_length - 1 : -1]  # (B, R)

        if self._sdpo_update_debug and self.rank == 0:
            print(
                f"[sdpo-ulysses-local-topk] gathered shapes: "
                f"topk_log_probs={tuple(topk_log_probs_response.shape)} "
                f"topk_indices={tuple(topk_indices_response.shape)} "
                f"tail={tuple(tail_log_prob_response.shape)}"
            )

        return topk_log_probs_response, topk_indices_response, tail_log_prob_response

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

    def _compute_sdpo_logit_loss(self, model_inputs: dict[str, Any], temperature: float) -> tuple[torch.Tensor, dict[str, float]]:
        _use_ulysses_local_topk = (
            self.config.padding_free
            and self.config.ulysses_size > 1
            and self.config.sdpo_approx_mode != "full_vocab"
        )

        if self._sdpo_update_debug:
            print(
                "[RCA_SDPO_LOSS_ENTRY] "
                f"rank={self.rank}/{self.world_size} "
                f"responses_shape={tuple(model_inputs['responses'].shape)} "
                f"input_ids_shape={tuple(model_inputs['input_ids'].shape)} "
                f"attention_mask_shape={tuple(model_inputs['attention_mask'].shape)} "
                f"position_ids_shape={tuple(model_inputs['position_ids'].shape)} "
                f"ulysses_local_topk={_use_ulysses_local_topk}"
            )
            print(
                "[RCA_SDPO_LOSS_CONFIG_PATH] "
                f"rank={self.rank}/{self.world_size} "
                f"loss_mode={self.config.loss_mode} approx_mode={self.config.sdpo_approx_mode} "
                f"divergence={self.config.sdpo_divergence} use_tail={self.config.sdpo_use_tail} "
                f"padding_free={self.config.padding_free} ulysses_size={self.config.ulysses_size} "
                f"feedback_mode={getattr(self.config, 'sdpo_feedback_mode', 'N/A')}"
            )

        # --- fail loudly if full_vocab + Ulysses --------------------------
        if (
            self.config.padding_free
            and self.config.ulysses_size > 1
            and self.config.sdpo_approx_mode == "full_vocab"
        ):
            raise ValueError(
                "sdpo_approx_mode='full_vocab' is incompatible with padding_free + ulysses_size > 1. "
                "Gathering full-vocab logits (V≈151936) across Ulysses SP ranks causes backward "
                "crashes (SplitWithSizesBackward0). Use approx_mode='topk' instead."
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

        # ------------------------------------------------------------------
        # Build teacher inputs (same for both paths)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Teacher forward — always full-vocab gather (no_grad ⇒ safe)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Student forward — path depends on Ulysses
        # ------------------------------------------------------------------
        student_forward_start = time.perf_counter()

        if _use_ulysses_local_topk:
            # ============================================================
            # NEW PATH: top-k computed locally BEFORE Ulysses gather
            # Gathers only (total_nnz, k) instead of (total_nnz, vocab_size)
            # ============================================================
            if self.rank == 0:
                print(
                    f"[sdpo-path] ULYSSES_LOCAL_TOPK active — "
                    f"gathering top-{self.config.sdpo_topk} per rank instead of full vocab "
                    f"(padding_free={self.config.padding_free}, ulysses_size={self.config.ulysses_size})"
                )
            student_topk_log_probs, student_topk_indices, student_tail_log_prob = (
                self._forward_student_topk_ulysses(model_inputs, temperature=temperature, topk=self.config.sdpo_topk)
            )
            student_forward_elapsed_ms = (time.perf_counter() - student_forward_start) * 1000.0

            if teacher_logits.shape[:2] != student_topk_log_probs.shape[:2]:
                raise ValueError(
                    f"Teacher and student response spans must align for sdpo_logit. "
                    f"teacher={tuple(teacher_logits.shape[:2])}, student_topk={tuple(student_topk_log_probs.shape[:2])}"
                )

            sdpo_loss, sdpo_metrics = compute_sdpo_topk_loss_preextracted(
                student_topk_log_probs=student_topk_log_probs,
                student_topk_indices=student_topk_indices,
                student_tail_log_prob=student_tail_log_prob,
                teacher_logits=teacher_logits,
                response_mask=response_mask,
                divergence=self.config.sdpo_divergence,
                use_tail=self.config.sdpo_use_tail,
            )

            if self._sdpo_backward_shape_debug and self.rank == 0:
                print(
                    "[sdpo-shape-debug] ULYSSES_LOCAL_TOPK — "
                    f"teacher_logits={tuple(teacher_logits.shape)} "
                    f"student_topk={tuple(student_topk_log_probs.shape)} "
                    f"student_tail={tuple(student_tail_log_prob.shape)} "
                    f"response_mask={tuple(response_mask.shape)}"
                )
        else:
            # ============================================================
            # EXISTING PATH: full-vocab logits (no Ulysses or ulysses_size=1)
            # ============================================================
            student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
            student_forward_elapsed_ms = (time.perf_counter() - student_forward_start) * 1000.0

            if teacher_logits.shape[:2] != student_logits.shape[:2]:
                raise ValueError("Teacher and student response spans must align for sdpo_logit.")

            if self._sdpo_backward_shape_debug and self.rank == 0:
                print(
                    "[sdpo-shape-debug] FULL_VOCAB_PATH — "
                    f"teacher_logits={tuple(teacher_logits.shape)} "
                    f"student_logits={tuple(student_logits.shape)} "
                    f"response_mask={tuple(response_mask.shape)}"
                )

            sdpo_loss, sdpo_metrics = compute_sdpo_logit_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                response_mask=response_mask,
                topk=self.config.sdpo_topk,
                divergence=self.config.sdpo_divergence,
                use_tail=self.config.sdpo_use_tail,
                approx_mode=self.config.sdpo_approx_mode,
            )

        metrics = {f"sdpo/{k}": v for k, v in sdpo_metrics.items()}
        if self._sdpo_update_debug:
            student_desc = (
                f"student_topk_shape={tuple(student_topk_log_probs.shape)}"
                if _use_ulysses_local_topk
                else f"student_logits_shape={tuple(student_logits.shape)} student_logits_numel={student_logits.numel()}"
            )
            print(
                "[sdpo-update-debug] "
                f"rank={self.rank}/{self.world_size} total_samples={num_total_samples} valid_samples={num_valid_samples} "
                f"ulysses_local_topk={_use_ulysses_local_topk} "
                f"response_mask_shape={tuple(response_mask.shape)} response_mask_numel={response_mask.numel()} "
                f"sdpo_valid_mask_shape={tuple(model_inputs['sdpo_valid_mask'].shape)} "
                f"response_token_mask_shape={tuple(model_inputs['response_token_mask'].shape)} "
                f"teacher_input_shape={tuple(teacher_input_ids.shape)} teacher_input_numel={teacher_input_ids.numel()} "
                f"teacher_logits_shape={tuple(teacher_logits.shape)} teacher_logits_numel={teacher_logits.numel()} "
                f"{student_desc} "
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
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                        if self._sdpo_backward_shape_debug:
                            with torch.autograd.detect_anomaly(check_nan=True):
                                loss.backward()
                        else:
                            loss.backward()
                        append_to_dict(metrics, sdpo_metrics)
                    else:
                        raise ValueError(f"Unknown actor.loss_mode: {self.config.loss_mode}")

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
