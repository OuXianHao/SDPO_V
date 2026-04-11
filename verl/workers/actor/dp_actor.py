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
from ...trainer.core_algos import compute_grpo_loss, compute_rlsd_token_advantages, compute_sdpo_logit_loss, compute_sdpo_v_loss, compute_sdpo_v_calibration_stats, compute_sdpo_v_softkl_loss
from ...utils.bad_video import construct_bad_video_inputs
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
        # Separate teacher model for SDPO distillation (EMA or trust-region).
        # When None (current default), the teacher forward falls back to
        # self.actor_module under torch.no_grad(), matching the current
        # same-model/different-prompt pattern.
        # To restore full original-SDPO EMA teacher semantics, set this to
        # a ref_module (e.g. ref_module_fsdp) from the FSDP worker init
        # and call _update_teacher() after each optimizer step.
        self.teacher_module: Optional[nn.Module] = None
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
            if self.config.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
                print(
                    f"[actor] sdpo settings: topk={self.config.sdpo_topk}, "
                    f"divergence={self.config.sdpo_divergence}, use_tail={self.config.sdpo_use_tail}, "
                    f"approx_mode={self.config.sdpo_approx_mode}, "
                    f"alpha={self.config.sdpo_alpha}"
                )
                if self.config.loss_mode == "dapo_with_sdpo":
                    print(
                        f"[actor] dapo_with_sdpo weights: lambda_dapo={self.config.lambda_dapo}, "
                        f"lambda_sdpo_t={self.config.lambda_sdpo_t}, "
                        f"lambda_sdpo_v={self.config.lambda_sdpo_v}"
                    )
                if self.config.sdpo_v_enabled:
                    print(
                        f"[actor] sdpo_v ENABLED: weight={self.config.sdpo_v_weight}, "
                        f"topk={self.config.sdpo_v_topk}, margin={self.config.sdpo_v_margin}, "
                        f"bad_video_mode={self.config.sdpo_v_bad_video_mode}, "
                        f"blur_sigma={self.config.sdpo_v_blur_sigma}, "
                        f"blur_fraction={self.config.sdpo_v_blur_fraction}, "
                        f"drop_fraction={self.config.sdpo_v_drop_fraction}, "
                        f"calibration={self.config.sdpo_v_calibration}"
                    )
                if self.config.sdpo_v_softkl_enabled:
                    print(
                        f"[actor] sdpo_v_softkl ENABLED: weight={self.config.sdpo_v_softkl_weight}, "
                        f"topk={self.config.sdpo_v_softkl_topk}, tau={self.config.sdpo_v_softkl_tau}, "
                        f"use_tail={self.config.sdpo_v_softkl_use_tail}, "
                        f"use_ema_bad_ref={self.config.sdpo_v_softkl_use_ema_bad_ref}"
                    )
                if self.config.teacher_reweight_enabled:
                    print(
                        f"[actor] teacher_reweight ENABLED: lambda={self.config.teacher_reweight_lambda}, "
                        f"eps_w_low={self.config.teacher_reweight_eps_w_low}, "
                        f"eps_w_high={self.config.teacher_reweight_eps_w_high}, "
                        f"delta_clamp={self.config.teacher_reweight_delta_clamp}, "
                        f"correct_hint={self.config.teacher_reweight_correct_hint}"
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

    def _update_teacher(self) -> None:
        """EMA update of teacher model weights from actor weights.

        Matches original SDPO's _update_teacher (dp_actor.py:132-151):
            teacher_param = (1 - rate) * teacher_param + rate * student_param

        Only runs when:
          - loss_mode is sdpo_logit, OR dapo_with_sdpo with teacher_reweight_enabled
          - teacher_module is a separate model (not None, not actor_module)
          - sdpo_teacher_update_rate > 0
        """
        _allow_ema = (
            self.config.loss_mode == "sdpo_logit"
            or (self.config.loss_mode == "dapo_with_sdpo" and self.config.teacher_reweight_enabled)
        )
        if not _allow_ema:
            return
        update_rate = self.config.sdpo_teacher_update_rate
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            return
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    def _render_teacher_prompt_text(self, content_text: str) -> str:
        format_prompt = self.config.teacher_format_prompt
        if format_prompt is None or format_prompt == "":
            return content_text

        from jinja2 import Template
        template = Template(format_prompt.strip())
        return template.render(content=content_text)

    def _build_answer_span_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        """Build a boolean mask marking tokens inside <answer>...</answer> spans.

        Used by teacher_reweight to exclude answer-span tokens from credit
        reweighting, preventing over-concentration on the final answer when the
        teacher guideline includes a correct-option hint.

        Scans each response row for the last <answer>...(</answer>) token
        subsequence and marks those positions True.
        """
        # Lazily encode marker token sequences
        if not hasattr(self, "_answer_open_ids"):
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            self._answer_open_ids = tokenizer.encode("<answer>", add_special_tokens=False)
            self._answer_close_ids = tokenizer.encode("</answer>", add_special_tokens=False)

        batch_size, seq_len = response_ids.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=response_ids.device)
        open_ids = self._answer_open_ids
        close_ids = self._answer_close_ids
        open_len = len(open_ids)
        close_len = len(close_ids)

        for i in range(batch_size):
            row = response_ids[i].tolist()
            # Find last occurrence of <answer> marker
            open_start = -1
            for j in range(len(row) - open_len + 1):
                if row[j : j + open_len] == open_ids:
                    open_start = j
            if open_start < 0:
                continue
            # Find first </answer> after it
            close_end = -1
            for j in range(open_start + open_len, len(row) - close_len + 1):
                if row[j : j + close_len] == close_ids:
                    close_end = j + close_len
                    break
            if close_end < 0:
                close_end = len(row)
            mask[i, open_start:close_end] = True

        return mask

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
        module: Optional[nn.Module] = None,
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
        # Resolve which model to use: teacher_module (separate FSDP ref model)
        # or actor_module (default).  This allows the same method to serve both
        # student and teacher forwards, matching original SDPO's module= pattern.
        model = module or self.actor_module

        if self._sdpo_update_debug:
            print(
                "[RCA_FORWARD_RESP_LOGITS_ENTRY] "
                f"rank={self.rank}/{self.world_size} "
                f"padding_free={self.config.padding_free} ulysses_size={self.config.ulysses_size} "
                f"using_teacher_module={module is not None}"
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

            output = model(
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
            #
            # DISABLED: The response_only_logits optimization sliced hidden_states
            # before lm_head inside the FSDP forward context.  This created a
            # SliceBackward0 node on hidden_states whose backward path confused
            # FSDP's flat-parameter management — the root FSDP unit's
            # SplitWithSizesBackward0 received a gradient of 2x the expected size.
            #
            # The dual-backward-reference issue (topk + logsumexp both
            # referencing the full logits tensor) is resolved by making logits
            # .contiguous() before returning (default path below), so both
            # backward paths accumulate into the copy first.
            #
            # To re-enable response_only_logits, set SDPO_FORCE_RESPONSE_ONLY=1.
            force_response_only = os.getenv("SDPO_FORCE_RESPONSE_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
            force_full_logits = os.getenv("SDPO_FORCE_FULL_LOGITS", "0").strip().lower() in {"1", "true", "yes", "on"}
            use_response_only = force_response_only and (not force_full_logits)

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
                    unwrapped = model
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

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **mm_inputs,
                    use_cache=False,
                    response_only_logits=response_length,
                )
                logits = output.logits

                # Verify the model actually honored response_only_logits.
                if logits.size(1) != response_length:
                    raise RuntimeError(
                        f"[SDPO] response_only_logits={response_length} was NOT "
                        f"honored by the live model forward. Returned logits shape "
                        f"{tuple(logits.shape)} (expected seq_dim={response_length}). "
                        f"Ensure apply_vl_forward_patch() is called before model construction."
                    )
                logits = logits / temperature
                if self.rank == 0:
                    print(
                        f"[sdpo-path-ok] response_only_logits HONORED. "
                        f"logits_shape={tuple(logits.shape)} "
                        f"(expected response_length={response_length})"
                    )
            else:
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **mm_inputs,
                    use_cache=False,
                )
                full_logits = output.logits

                # ---- Experiment gate: SDPO_FWD_EXP ----
                #   0 = default SDPO path (out-of-place div, 3D logits)
                #   1 = GRPO-style: in-place div_ + slice + .contiguous()
                #   2 = SDPO slice but add .contiguous() before returning
                #   3 = GRPO-style in-place div_ but NO .contiguous()
                _fwd_exp = int(os.getenv("SDPO_FWD_EXP", "0"))
                if _fwd_exp == 1:
                    # Match GRPO exactly: in-place div, then slice, then contiguous
                    full_logits.div_(temperature)
                    logits = full_logits[:, -response_length - 1 : -1, :].contiguous()
                    if self.rank == 0:
                        print(f"[SDPO_FWD_EXP1] in-place div + slice + contiguous, logits shape={tuple(logits.shape)}")
                elif _fwd_exp == 2:
                    # Out-of-place div (like default) but add .contiguous()
                    logits = (full_logits[:, -response_length - 1 : -1, :] / temperature).contiguous()
                    if self.rank == 0:
                        print(f"[SDPO_FWD_EXP2] out-of-place div + contiguous, logits shape={tuple(logits.shape)}")
                elif _fwd_exp == 3:
                    # In-place div but NO contiguous (isolates in-place effect)
                    full_logits.div_(temperature)
                    logits = full_logits[:, -response_length - 1 : -1, :]
                    if self.rank == 0:
                        print(f"[SDPO_FWD_EXP3] in-place div + slice (no contiguous), logits shape={tuple(logits.shape)}")
                else:
                    # .contiguous() creates a fresh tensor, breaking the view
                    # relationship with the model output.  This is required so
                    # that dual backward references in _extract_topk_logps
                    # (topk + logsumexp) both accumulate into this tensor first,
                    # then flow back as a single gradient through CopyBackward0
                    # — avoiding the FSDP SplitWithSizesBackward0 2x-gradient
                    # mismatch that occurs when two backward paths reach a
                    # shared view of the flat-parameter output.
                    logits = (full_logits[:, -response_length - 1 : -1, :] / temperature).contiguous()

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

        Uses ``topk(logits)`` with a live ``logsumexp(logits)`` for
        normalization — matching the original SDPO reference implementation.
        The returned ``topk_logps`` equal ``log_softmax(logits)`` at the
        top-k positions.  Gradient flows through both the gathered top-k
        values (via ``GatherBackward0``) and the logsumexp normalizer,
        giving the correct log-softmax gradient:
        ``d(topk_logps_j)/d(logit_i) = 1_{i=j} - softmax(logit_i)``.

        The logsumexp backward creates a dense ``(B, R, V)`` gradient.
        To prevent FSDP ``SplitWithSizesBackward0`` 2x-gradient mismatches,
        callers must ensure ``logits`` is a contiguous tensor (not a view
        of the model output).  See ``_forward_response_logits``.

        Args:
            logits: ``(batch, response_len, vocab_size)`` — response logits
                (already temperature-scaled).
            k: number of top tokens.
            topk_indices: optional pre-computed indices from another
                distribution (student's indices reused for teacher).
            detach_lse: ignored (kept for API compatibility).

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

        # Match the SDPO reference pattern: topk(logits) with live
        # logsumexp for normalization — gradient flows through both the
        # gathered topk values AND the logsumexp normalizer.  This gives
        # the correct log_softmax gradient:
        #   d(topk_logps_j)/d(logit_i) = 1_{i=j} - softmax(logit_i)
        # The softmax correction term (from logsumexp backward) pushes
        # down all vocab logits proportionally to their probability,
        # redistributing mass from non-top-k to top-k positions.
        #
        # Previously logsumexp was detached to avoid an FSDP
        # SplitWithSizesBackward0 2x-gradient mismatch.  That issue is
        # now resolved by making the logits .contiguous() in
        # _forward_response_logits, so the dual backward paths (topk +
        # logsumexp) accumulate into the contiguous copy first.
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # (batch, resp_len, 1)

        if topk_indices is None:
            # Select indices without gradient (removes TopkBackward0 from the
            # backward graph).  Values are gathered from live logits so the
            # gradient still flows through GatherBackward0 → logits → model.
            with torch.no_grad():
                _, topk_indices = torch.topk(logits, k, dim=-1)
            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
            if _debug and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] step=E05 gather (detached-idx) caller={_caller} "
                    f"topk_logits shape={tuple(topk_logits.shape)} stride={topk_logits.stride()} "
                    f"is_contiguous={topk_logits.is_contiguous()} "
                    f"requires_grad={topk_logits.requires_grad} grad_fn={topk_logits.grad_fn} "
                    f"topk_indices shape={tuple(topk_indices.shape)} dtype={topk_indices.dtype}"
                )
        else:
            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
            if _debug and self.rank == 0:
                print(
                    f"[SDPO_T_SHAPE] step=E05 gather caller={_caller} "
                    f"topk_logits shape={tuple(topk_logits.shape)} stride={topk_logits.stride()} "
                    f"is_contiguous={topk_logits.is_contiguous()} "
                    f"requires_grad={topk_logits.requires_grad} grad_fn={topk_logits.grad_fn} "
                    f"topk_indices shape={tuple(topk_indices.shape)}"
                )

        # Proper log-probs: topk_logits - logsumexp.  Grad through both terms.
        topk_logps = topk_logits - lse  # (batch, resp_len, k)
        lse = lse.squeeze(-1)  # (batch, resp_len) — for return value compatibility

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

        # NOTE: we intentionally do NOT short-circuit here when all tokens
        # are masked.  Under FSDP, every rank must execute the same forward/
        # backward structure (same all-gather / reduce-scatter sequence).
        # When some ranks have valid_samples=0, skipping the model forward
        # would create an asymmetric backward graph, causing
        # SplitWithSizesBackward0 gradient-size mismatches across ranks.
        #
        # Instead, the normal code path runs: the student forward executes
        # (keeping FSDP synchronised), and the all-zero response_mask
        # naturally produces zero loss with the correct full-model backward
        # graph.

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

        # ---- Experiment gate: SDPO_EXP selects which experiment to run ----
        #   0 = default (teacher-first, student-last, original behaviour)
        #   1 = SKIP teacher forward entirely; use detached student logits as
        #       "teacher".  Tests: is the double-forward the root cause?
        #   2 = Run student FIRST, then teacher.  Tests: does forward order
        #       matter for FSDP's execution-order bookkeeping?
        #   3 = Run teacher-first + student-last but call
        #       actor_module._reset_lazy_init() between them.  Tests: can
        #       resetting FSDP root state between forwards fix the issue?
        _sdpo_exp = int(os.getenv("SDPO_EXP", "0"))
        if _debug_shapes and self.rank == 0:
            print(f"[SDPO_T_BRANCH] step=EXP SDPO_EXP={_sdpo_exp}")

        if _sdpo_exp == 1:
            # ---- EXPERIMENT 1: no teacher forward at all ----
            # Run student only; teacher = detached copy of student logits.
            # This produces meaningless KL (KL(p||p)=0) but tests whether
            # backward succeeds when there is exactly ONE forward through FSDP.
            student_forward_start = time.perf_counter()
            student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
            student_forward_elapsed_ms = (time.perf_counter() - student_forward_start) * 1000.0
            teacher_logits = student_logits.detach().clone()
            teacher_forward_elapsed_ms = 0.0
            if self.rank == 0:
                print("[SDPO_T_EXP1] teacher=detached_student, single FSDP forward")

        elif _sdpo_exp == 2:
            # ---- EXPERIMENT 2: student FIRST, then teacher ----
            # Reverses the forward order.  If FSDP's execution-order tracking
            # is the culprit, this should ALSO crash — but the error might
            # differ (confirming order-sensitivity) or it might succeed
            # (if FSDP only cares about the LAST forward matching backward).
            student_forward_start = time.perf_counter()
            student_logits = self._forward_response_logits(model_inputs, temperature=temperature)
            student_forward_elapsed_ms = (time.perf_counter() - student_forward_start) * 1000.0
            teacher_forward_start = time.perf_counter()
            with torch.no_grad():
                teacher_logits = self._forward_response_logits(
                    model_inputs,
                    temperature=temperature,
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    position_ids=teacher_position_ids,
                    multi_modal_inputs=teacher_multi_modal_inputs,
                    module=self.teacher_module,
                )
            teacher_forward_elapsed_ms = (time.perf_counter() - teacher_forward_start) * 1000.0
            if self.rank == 0:
                print("[SDPO_T_EXP2] student-first, teacher-second (reversed order)")

        else:
            # ---- Default (exp 0) or exp 3: teacher FIRST, student LAST ----
            # When self.teacher_module is set (e.g. ref_fsdp_module from FSDP
            # worker), teacher forward goes through a SEPARATE FSDP model —
            # matching original SDPO's EMA teacher architecture.  This also
            # eliminates the double-forward-through-same-FSDP issue.
            # When teacher_module is None, falls back to self.actor_module
            # (same-model/different-prompt, legacy behavior).
            teacher_model = self.teacher_module  # None → falls back in _forward_response_logits
            teacher_forward_start = time.perf_counter()
            with torch.no_grad():
                if _debug_shapes and self.rank == 0:
                    print(
                        f"[SDPO_T_BRANCH] step=T00 teacher forward ENTRY "
                        f"fsdp_wrapped={isinstance(self.actor_module, FSDP)} "
                        f"separate_teacher={teacher_model is not None and teacher_model is not self.actor_module}"
                    )
                teacher_logits = self._forward_response_logits(
                    model_inputs,
                    temperature=temperature,
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    position_ids=teacher_position_ids,
                    multi_modal_inputs=teacher_multi_modal_inputs,
                    module=teacher_model,
                )
                if _debug_shapes and self.rank == 0:
                    print(
                        f"[SDPO_T_BRANCH] step=T01 teacher forward EXIT "
                        f"teacher_logits shape={tuple(teacher_logits.shape)} "
                        f"requires_grad={teacher_logits.requires_grad}"
                    )
            teacher_forward_elapsed_ms = (time.perf_counter() - teacher_forward_start) * 1000.0

            if _sdpo_exp == 3:
                # ---- EXPERIMENT 3: reset FSDP lazy init between forwards ----
                # Attempts to clear stale execution-order state left by the
                # teacher forward.  If the crash disappears, it confirms the
                # double-forward corrupts FSDP's _exec_order_data.
                if isinstance(self.actor_module, FSDP):
                    if hasattr(self.actor_module, '_reset_lazy_init'):
                        self.actor_module._reset_lazy_init()
                        if self.rank == 0:
                            print("[SDPO_T_EXP3] called _reset_lazy_init() between forwards")
                    else:
                        if self.rank == 0:
                            print("[SDPO_T_EXP3] _reset_lazy_init not available, skipping reset")

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
                    alpha=self.config.sdpo_alpha,
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
                    alpha=self.config.sdpo_alpha,
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

        # ================================================================
        # SDPO-V: visual separation losses (good-video vs bad-video)
        # Hinge line: sdpo_v_enabled. Soft-KL line: sdpo_v_softkl_enabled.
        # Both are default-off and can coexist independently.
        # ================================================================
        combined_loss = sdpo_loss
        has_v_component = False

        if self.config.sdpo_v_enabled:
            sdpo_v_loss, sdpo_v_metrics = self._compute_sdpo_v_loss(
                model_inputs=model_inputs,
                student_logits=student_logits,
                temperature=temperature,
            )
            combined_loss = combined_loss + self.config.sdpo_v_weight * sdpo_v_loss
            metrics.update({f"sdpo_v/{k}": v for k, v in sdpo_v_metrics.items()})
            has_v_component = True

        if self.config.sdpo_v_softkl_enabled:
            sdpo_v_sk_loss, sdpo_v_sk_metrics = self._compute_sdpo_v_softkl_loss(
                model_inputs=model_inputs,
                student_logits=student_logits,
                temperature=temperature,
            )
            combined_loss = combined_loss + self.config.sdpo_v_softkl_weight * sdpo_v_sk_loss
            metrics.update({f"sdpo_v_softkl/{k}": v for k, v in sdpo_v_sk_metrics.items()})
            has_v_component = True

        if has_v_component:
            metrics["sdpo/combined_loss"] = combined_loss.detach().item()
            metrics["sdpo/t_loss"] = sdpo_loss.detach().item()
            return combined_loss, metrics

        return sdpo_loss, metrics

    def _compute_sdpo_v_loss(
        self,
        model_inputs: dict[str, Any],
        student_logits: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SDPO-V visual separation loss.

        This method:
        1. Constructs bad-video multimodal inputs by degrading pixel_values_videos
        2. Runs a forward pass with bad-video inputs (same text/response tokens)
        3. Extracts good-branch and bad-branch top-k log-probs at the same
           good-selected top-k indices
        4. Computes a token-level margin hinge loss

        Only samples with overall reward == 1 participate (sdpo_v_valid_mask).
        """
        _debug = self.config.sdpo_v_debug

        # ---- SDPO-V response mask: response_mask & sdpo_v_valid_mask ----
        # sdpo_v_valid_mask selects samples where accuracy_reward == 1
        sdpo_v_mask = (
            model_inputs["response_mask"].bool()
            & model_inputs["response_token_mask"].bool()
            & model_inputs["sdpo_v_valid_mask"].bool()
        )
        num_v_valid = int(sdpo_v_mask.any(dim=-1).sum().item())

        if _debug and self.rank == 0:
            print(
                f"[sdpo_v] ENTRY num_v_valid_samples={num_v_valid}/{model_inputs['responses'].shape[0]} "
                f"bad_video_mode={self.config.sdpo_v_bad_video_mode} "
                f"margin={self.config.sdpo_v_margin} topk={self.config.sdpo_v_topk}"
            )

        # ---- Construct bad-video multimodal inputs ----
        # Collate multimodal inputs from the micro-batch
        mm_inputs = defaultdict(list)
        if "multi_modal_inputs" in model_inputs:
            mm_inputs = batch_collate(model_inputs["multi_modal_inputs"])
            mm_inputs = {key: torch.cat(value, dim=0) for key, value in mm_inputs.items()}
        else:
            mm_inputs = {}

        # Get video_grid_thw for frame-level degradation
        video_grid_thw = mm_inputs.get("video_grid_thw", None)

        # Extract vision config for proper spatial blur reconstruction.
        _vcfg = getattr(getattr(self.actor_module, "config", None), "vision_config", None)
        _patch_size = getattr(_vcfg, "patch_size", 14)
        _temporal_patch_size = getattr(_vcfg, "temporal_patch_size", 2)
        _in_channels = getattr(_vcfg, "in_channels", 3)

        bad_mm_inputs = construct_bad_video_inputs(
            multi_modal_inputs=mm_inputs,
            video_grid_thw=video_grid_thw,
            mode=self.config.sdpo_v_bad_video_mode,
            blur_sigma=self.config.sdpo_v_blur_sigma,
            blur_fraction=self.config.sdpo_v_blur_fraction,
            drop_fraction=self.config.sdpo_v_drop_fraction,
            shuffle_fraction=self.config.sdpo_v_shuffle_fraction,
            patch_size=_patch_size,
            temporal_patch_size=_temporal_patch_size,
            in_channels=_in_channels,
        )

        if _debug and self.rank == 0:
            if "pixel_values_videos" in mm_inputs:
                orig_pv = mm_inputs["pixel_values_videos"]
                bad_pv = bad_mm_inputs["pixel_values_videos"]
                diff = (orig_pv - bad_pv).abs().mean().item()
                print(
                    f"[sdpo_v] bad_video constructed: pixel_values shape={tuple(orig_pv.shape)} "
                    f"mean_abs_diff={diff:.6f}"
                )

        # ---- Bad-video EMA reference forward (no_grad) ----
        # Use the same text tokens (input_ids, attention_mask, position_ids)
        # but with degraded multimodal inputs. The bad-video reference branch
        # runs through the EMA teacher module under no_grad for stability,
        # matching the soft-KL line's reference-branch design.
        if self.teacher_module is not None and self.teacher_module is not self.actor_module:
            bad_ref_module = self.teacher_module
        else:
            bad_ref_module = self.actor_module

        with torch.no_grad():
            bad_logits = self._forward_response_logits(
                model_inputs,
                temperature=temperature,
                multi_modal_inputs=bad_mm_inputs,
                module=bad_ref_module,
            )

        _ema_active = (bad_ref_module is not self.actor_module)
        if _debug and self.rank == 0:
            print(
                f"[sdpo_v] bad_logits shape={tuple(bad_logits.shape)} "
                f"student_logits shape={tuple(student_logits.shape)} "
                f"ema_active={_ema_active} requires_grad={bad_logits.requires_grad}"
            )

        # ---- Extract top-k log-probs ----
        # Good branch: student_logits (already computed in SDPO-T path)
        # The good branch selects the top-k indices (anchor).
        v_topk = self.config.sdpo_v_topk

        # Good-branch top-k (indices selected from good/student logits)
        # We need student_logits to still have grad for the good-branch score.
        good_topk = self._extract_topk_logps(student_logits, k=v_topk)

        # Bad-branch: gather at the SAME good-selected indices
        bad_topk = self._extract_topk_logps(
            bad_logits, k=v_topk, topk_indices=good_topk["topk_indices"]
        )

        good_topk_logps = good_topk["topk_logps"]  # (batch, resp_len, k)
        bad_topk_logps = bad_topk["topk_logps"]    # (batch, resp_len, k)

        # Optionally add tail bucket for proper distribution
        if self.config.sdpo_v_use_tail:
            def _add_tail(logps: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(logps, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_logp = torch.log(-torch.expm1(log_s))
                return torch.cat([logps, tail_logp], dim=-1)
            good_topk_logps = _add_tail(good_topk_logps)
            bad_topk_logps = _add_tail(bad_topk_logps)

        # ---- Calibration mode: collect stats and return zero loss ----
        if self.config.sdpo_v_calibration:
            calib_stats = compute_sdpo_v_calibration_stats(
                good_topk_logps=good_topk_logps.detach(),
                bad_topk_logps=bad_topk_logps.detach(),
                response_mask=sdpo_v_mask,
            )
            if self.rank == 0:
                print(f"[sdpo_v_calibration] {calib_stats}")
            # Return zero loss so training is unaffected during calibration
            zero_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            return zero_loss, calib_stats

        # ---- Compute SDPO-V margin loss ----
        v_loss, v_metrics = compute_sdpo_v_loss(
            good_topk_logps=good_topk_logps,
            bad_topk_logps=bad_topk_logps,
            response_mask=sdpo_v_mask,
            margin=self.config.sdpo_v_margin,
            use_tail=self.config.sdpo_v_use_tail,
        )

        if _debug and self.rank == 0:
            print(
                f"[sdpo_v] loss={v_loss.item():.6f} "
                f"delta_mean={v_metrics.get('v_delta_mean', 0):.6f} "
                f"active_frac={v_metrics.get('v_active_frac', 0):.4f}"
            )

        return v_loss, v_metrics

    def _compute_sdpo_v_softkl_loss(
        self,
        model_inputs: dict[str, Any],
        student_logits: torch.Tensor,
        temperature: float,
        bad_mm_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute SDPO-V soft-capped forward KL loss.

        This method:
        1. Constructs bad-video multimodal inputs (or reuses pre-computed ones)
        2. Runs a forward pass with bad-video inputs through the EMA teacher
           module under no_grad (or actor_module fallback)
        3. Extracts good-branch and bad-ref-branch top-k log-probs at the same
           good-selected top-k indices
        4. Computes a token-level soft-capped forward KL loss

        Only samples with overall reward == 1 participate (sdpo_v_valid_mask).
        """
        _debug = self.config.sdpo_v_softkl_debug

        # ---- SDPO-V response mask (shared with hinge line) ----
        sdpo_v_mask = (
            model_inputs["response_mask"].bool()
            & model_inputs["response_token_mask"].bool()
            & model_inputs["sdpo_v_valid_mask"].bool()
        )
        num_v_valid = int(sdpo_v_mask.any(dim=-1).sum().item())

        if _debug and self.rank == 0:
            print(
                f"[sdpo_v_softkl] ENTRY num_v_valid_samples={num_v_valid}/{model_inputs['responses'].shape[0]} "
                f"topk={self.config.sdpo_v_softkl_topk} tau={self.config.sdpo_v_softkl_tau} "
                f"use_ema_bad_ref={self.config.sdpo_v_softkl_use_ema_bad_ref}"
            )

        # ---- Construct bad-video multimodal inputs (reuse if provided) ----
        if bad_mm_inputs is None:
            mm_inputs = defaultdict(list)
            if "multi_modal_inputs" in model_inputs:
                mm_inputs = batch_collate(model_inputs["multi_modal_inputs"])
                mm_inputs = {key: torch.cat(value, dim=0) for key, value in mm_inputs.items()}
            else:
                mm_inputs = {}

            video_grid_thw = mm_inputs.get("video_grid_thw", None)
            _vcfg = getattr(getattr(self.actor_module, "config", None), "vision_config", None)
            _patch_size = getattr(_vcfg, "patch_size", 14)
            _temporal_patch_size = getattr(_vcfg, "temporal_patch_size", 2)
            _in_channels = getattr(_vcfg, "in_channels", 3)

            bad_mm_inputs = construct_bad_video_inputs(
                multi_modal_inputs=mm_inputs,
                video_grid_thw=video_grid_thw,
                mode=self.config.sdpo_v_bad_video_mode,
                blur_sigma=self.config.sdpo_v_blur_sigma,
                blur_fraction=self.config.sdpo_v_blur_fraction,
                drop_fraction=self.config.sdpo_v_drop_fraction,
                shuffle_fraction=self.config.sdpo_v_shuffle_fraction,
                patch_size=_patch_size,
                temporal_patch_size=_temporal_patch_size,
                in_channels=_in_channels,
            )

        # ---- Bad-video EMA reference forward (no_grad) ----
        use_ema = self.config.sdpo_v_softkl_use_ema_bad_ref
        if use_ema and self.teacher_module is not None and self.teacher_module is not self.actor_module:
            bad_ref_module = self.teacher_module
        else:
            bad_ref_module = self.actor_module

        ema_active = (bad_ref_module is not self.actor_module)

        with torch.no_grad():
            bad_ref_logits = self._forward_response_logits(
                model_inputs,
                temperature=temperature,
                multi_modal_inputs=bad_mm_inputs,
                module=bad_ref_module,
            )

        if _debug and self.rank == 0:
            print(
                f"[sdpo_v_softkl] bad_ref_logits shape={tuple(bad_ref_logits.shape)} "
                f"ema_active={ema_active} requires_grad={bad_ref_logits.requires_grad}"
            )

        # ---- Extract top-k log-probs ----
        sk_topk = self.config.sdpo_v_softkl_topk

        # Good branch: student_logits (with grad), selects top-k indices
        good_topk = self._extract_topk_logps(student_logits, k=sk_topk)

        # Bad-ref branch: gather at the SAME good-selected indices (no grad)
        bad_ref_topk = self._extract_topk_logps(
            bad_ref_logits, k=sk_topk, topk_indices=good_topk["topk_indices"]
        )

        good_topk_logps = good_topk["topk_logps"]      # (batch, resp_len, k)
        bad_ref_topk_logps = bad_ref_topk["topk_logps"]  # (batch, resp_len, k)

        # Optionally add tail bucket
        if self.config.sdpo_v_softkl_use_tail:
            def _add_tail(logps: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(logps, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_logp = torch.log(-torch.expm1(log_s))
                return torch.cat([logps, tail_logp], dim=-1)
            good_topk_logps = _add_tail(good_topk_logps)
            bad_ref_topk_logps = _add_tail(bad_ref_topk_logps)

        # ---- Compute soft-capped forward KL loss ----
        sk_loss, sk_metrics = compute_sdpo_v_softkl_loss(
            good_topk_logps=good_topk_logps,
            bad_ref_topk_logps=bad_ref_topk_logps,
            response_mask=sdpo_v_mask,
            tau=self.config.sdpo_v_softkl_tau,
            use_tail=self.config.sdpo_v_softkl_use_tail,
            kl_max=self.config.sdpo_v_softkl_kl_max,
        )

        # Add EMA status to metrics
        sk_metrics["vsk_ema_active"] = 1.0 if ema_active else 0.0

        if _debug and self.rank == 0:
            print(
                f"[sdpo_v_softkl] loss={sk_loss.item():.6f} "
                f"kl_raw_mean={sk_metrics.get('vsk_kl_raw_mean', 0):.6f} "
                f"kl_capped_mean={sk_metrics.get('vsk_kl_capped_mean', 0):.6f} "
                f"ema_active={ema_active}"
            )

        return sk_loss, sk_metrics

    def _compute_dapo_with_sdpo_loss(
        self,
        model_inputs: dict[str, Any],
        temperature: float,
        total_response_tokens: torch.Tensor,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combined DAPO + SDPO-T + SDPO-V loss for dapo_with_sdpo mode.

        Uses a SINGLE student forward pass:
        1. Calls _forward_response_logits() → student_logits (batch, resp_len, vocab)
        2. Derives log_probs from student_logits for the GRPO/DAPO loss
        3. Computes SDPO-T loss from student_logits + teacher_logits
        4. Optionally computes SDPO-V loss from student_logits + bad-video logits

        Returns:
            combined_loss: scalar (before per-token scaling)
            metrics: dict of float metrics
        """
        metrics: dict[str, float] = {}
        responses = model_inputs["responses"]
        response_mask = model_inputs["response_mask"]  # used by DAPO (unmasked)

        # ================================================================
        # STEP 1: Build teacher inputs (needed for SDPO-T)
        # ================================================================
        teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_multi_modal_inputs = (
            self._build_teacher_inputs(model_inputs)
        )

        # ================================================================
        # STEP 2: Teacher forward (no_grad) — for SDPO-T
        # ================================================================
        teacher_model = self.teacher_module  # None → falls back to actor_module
        with torch.no_grad():
            teacher_logits = self._forward_response_logits(
                model_inputs,
                temperature=temperature,
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                position_ids=teacher_position_ids,
                multi_modal_inputs=teacher_multi_modal_inputs,
                module=teacher_model,
            )

        # ================================================================
        # STEP 3: Student forward (with grad) — shared for DAPO + SDPO-T + SDPO-V
        # ================================================================
        student_logits = self._forward_response_logits(model_inputs, temperature=temperature)

        # Verify alignment
        if teacher_logits.shape[:2] != student_logits.shape[:2]:
            raise ValueError(
                f"Teacher/student response span mismatch: teacher={tuple(teacher_logits.shape)}, "
                f"student={tuple(student_logits.shape)}."
            )

        # ================================================================
        # STEP 4: Derive log_probs for DAPO/GRPO loss from student_logits
        # ================================================================
        log_probs = self.log_probs_from_logits(student_logits, responses)  # (batch, resp_len)

        old_log_probs = model_inputs["old_log_probs"]
        advantages = model_inputs["advantages"]

        # ================================================================
        # STEP 4b: Teacher-guided token-level advantage reweighting
        # When enabled, replace the uniform token advantage with a
        # token-varying A_hat_t that uses teacher-student logprob
        # difference to modulate magnitude only (direction unchanged).
        # ================================================================
        if self.config.teacher_reweight_enabled:
            # Effective lambda: optionally decay linearly to 0 over training.
            # If decay_to_zero_step <= 0, lambda is constant (no decay).
            init_lambda = self.config.teacher_reweight_lambda
            decay_step = self.config.teacher_reweight_lambda_decay_to_zero_step
            if decay_step > 0:
                effective_lambda = init_lambda * max(0.0, 1.0 - global_step / decay_step)
            else:
                effective_lambda = init_lambda

            with torch.no_grad():
                teacher_log_probs = self.log_probs_from_logits(teacher_logits, responses)
            # Build answer-span mask to exclude <answer>...</answer> from reweighting
            answer_span_mask = None
            if self.processor is not None:
                answer_span_mask = self._build_answer_span_mask(responses)
            advantages, reweight_metrics = compute_rlsd_token_advantages(
                advantages=advantages,
                teacher_logprobs=teacher_log_probs.detach(),
                student_logprobs=log_probs.detach(),
                response_mask=response_mask,
                rlsd_lambda=effective_lambda,
                rlsd_eps_w_low=self.config.teacher_reweight_eps_w_low,
                rlsd_eps_w_high=self.config.teacher_reweight_eps_w_high,
                delta_clamp=self.config.teacher_reweight_delta_clamp,
                answer_span_mask=answer_span_mask,
            )
            reweight_metrics["effective_lambda"] = effective_lambda
            metrics.update({f"reweight/{k}": v for k, v in reweight_metrics.items()})

        dapo_loss, grpo_metrics = compute_grpo_loss(
            old_log_probs=old_log_probs,
            log_probs=log_probs,
            advantages=advantages,
            response_mask=response_mask,  # DAPO uses normal response_mask, NOT sdpo_valid_mask
            clip_ratio_low=self.config.clip_ratio_low,
            clip_ratio_high=self.config.clip_ratio_high,
            clip_ratio_dual=self.config.clip_ratio_dual,
            loss_avg_mode=self.config.loss_avg_mode,
        )
        metrics.update({f"grpo/{k}": v for k, v in grpo_metrics.items()})

        # ================================================================
        # STEP 5: SDPO-T loss (uses sdpo_valid_mask for mixed-group gating)
        # Skipped when teacher_reweight_enabled — the teacher signal is used
        # for token-level advantage reweighting instead of a KL objective.
        # ================================================================
        if self.config.teacher_reweight_enabled:
            sdpo_t_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            sdpo_t_response_mask = (
                model_inputs["response_mask"].bool()
                & model_inputs["response_token_mask"].bool()
                & model_inputs["sdpo_valid_mask"].bool()
            )

            approx_mode = self.config.sdpo_approx_mode
            topk_mode = approx_mode in ("topk", "student_topk_tail")
            topk_k = self.config.sdpo_topk

            if topk_mode:
                student_topk = self._extract_topk_logps(student_logits, k=topk_k)
                with torch.no_grad():
                    teacher_topk = self._extract_topk_logps(
                        teacher_logits, k=topk_k, topk_indices=student_topk["topk_indices"]
                    )
                sdpo_t_loss, sdpo_t_metrics = compute_sdpo_logit_loss(
                    student_topk_logps=student_topk["topk_logps"],
                    teacher_topk_logps=teacher_topk["topk_logps"],
                    response_mask=sdpo_t_response_mask,
                    topk=topk_k,
                    divergence=self.config.sdpo_divergence,
                    use_tail=self.config.sdpo_use_tail,
                    approx_mode=approx_mode,
                    alpha=self.config.sdpo_alpha,
                    student_logits_for_metrics=student_logits.detach(),
                    teacher_logits_for_metrics=teacher_logits.detach(),
                )
            else:
                sdpo_t_loss, sdpo_t_metrics = compute_sdpo_logit_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    response_mask=sdpo_t_response_mask,
                    topk=topk_k,
                    divergence=self.config.sdpo_divergence,
                    use_tail=self.config.sdpo_use_tail,
                    approx_mode=approx_mode,
                    alpha=self.config.sdpo_alpha,
                )
            metrics.update({f"sdpo/{k}": v for k, v in sdpo_t_metrics.items()})

        # ================================================================
        # STEP 6: SDPO-V loss (if enabled)
        # ================================================================
        sdpo_v_loss_val = torch.tensor(0.0, device=student_logits.device)
        if self.config.sdpo_v_enabled:
            sdpo_v_loss, sdpo_v_metrics = self._compute_sdpo_v_loss(
                model_inputs=model_inputs,
                student_logits=student_logits,
                temperature=temperature,
            )
            sdpo_v_loss_val = sdpo_v_loss
            metrics.update({f"sdpo_v/{k}": v for k, v in sdpo_v_metrics.items()})

        if self.config.sdpo_v_softkl_enabled:
            sdpo_v_sk_loss, sdpo_v_sk_metrics = self._compute_sdpo_v_softkl_loss(
                model_inputs=model_inputs,
                student_logits=student_logits,
                temperature=temperature,
            )
            sdpo_v_loss_val = sdpo_v_loss_val + sdpo_v_sk_loss
            metrics.update({f"sdpo_v_softkl/{k}": v for k, v in sdpo_v_sk_metrics.items()})

        # ================================================================
        # STEP 7: Aggregate combined loss
        # ================================================================
        lambda_dapo = self.config.lambda_dapo
        lambda_sdpo_t = self.config.lambda_sdpo_t
        lambda_sdpo_v = self.config.lambda_sdpo_v

        combined_loss = (
            lambda_dapo * dapo_loss
            + lambda_sdpo_t * sdpo_t_loss
            + lambda_sdpo_v * sdpo_v_loss_val
        )

        # Logging
        metrics["combined/total_loss"] = combined_loss.detach().item()
        metrics["combined/dapo_loss"] = dapo_loss.detach().item()
        metrics["combined/sdpo_t_loss"] = sdpo_t_loss.detach().item()
        metrics["combined/sdpo_v_loss"] = sdpo_v_loss_val.detach().item()
        metrics["combined/lambda_dapo"] = lambda_dapo
        metrics["combined/lambda_sdpo_t"] = lambda_sdpo_t
        metrics["combined/lambda_sdpo_v"] = lambda_sdpo_v
        metrics["combined/sdpo_t_valid_ratio"] = float(sdpo_t_response_mask.any(dim=-1).sum().item()) / max(response_mask.shape[0], 1) if not self.config.teacher_reweight_enabled else 0.0

        if self.rank == 0:
            print(
                f"[dapo_with_sdpo] L_total={combined_loss.item():.6f} "
                f"L_dapo={dapo_loss.item():.6f} L_sdpo_t={sdpo_t_loss.item():.6f} "
                f"L_sdpo_v={sdpo_v_loss_val.item():.6f} "
                f"λ_dapo={lambda_dapo} λ_sdpo_t={lambda_sdpo_t} λ_sdpo_v={lambda_sdpo_v}"
            )

        return combined_loss, metrics

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
            select_keys.extend(["sdpo_valid_mask", "response_token_mask", "sdpo_v_valid_mask"])
            non_tensor_select_keys.extend(
                ["raw_prompt_text", "prompt_text", "feedback_text", "teacher_prompt_text", "multi_modal_data", "pad_token_id"]
            )
        elif self.config.loss_mode == "dapo_with_sdpo":
            # Combined mode needs BOTH GRPO fields and SDPO fields
            select_keys.extend(["old_log_probs", "advantages"])
            select_keys.extend(["sdpo_valid_mask", "response_token_mask", "sdpo_v_valid_mask"])
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
                    elif self.config.loss_mode == "dapo_with_sdpo":
                        loss, combined_metrics = self._compute_dapo_with_sdpo_loss(
                            model_inputs, temperature=temperature, total_response_tokens=total_response_tokens,
                            global_step=data.meta_info.get("global_step", 0),
                        )
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                        loss.backward()
                        append_to_dict(metrics, combined_metrics)
                    else:
                        raise ValueError(f"Unknown actor.loss_mode: {self.config.loss_mode}")

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm):
                    self._update_teacher()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
