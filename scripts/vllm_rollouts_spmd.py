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

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import decode_video_with_cv2, process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    if processor is None:
        return None

    logit_bias = {}
    if hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias[image_token_id] = -100
    if hasattr(processor, "video_token"):
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        logit_bias[video_token_id] = -100

    return logit_bias or None


def _safe_shape(value: Any) -> Optional[tuple]:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return tuple(value.shape)
    return None


def _summarize_vllm_multimodal(mm_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if mm_data is None:
        return {"type": "text"}

    if "image" in mm_data:
        image_shapes = [_safe_shape(image) for image in mm_data["image"]]
        return {
            "type": "image",
            "num_images": len(mm_data["image"]),
            "image_shapes": image_shapes,
        }

    if "video" in mm_data:
        video_summaries = []
        for video in mm_data["video"]:
            if isinstance(video, tuple) and len(video) == 2:
                clip, metadata = video
            else:
                clip, metadata = video, None

            summary = {"shape": _safe_shape(clip)}
            if isinstance(metadata, dict):
                frames_indices = metadata.get("frames_indices")
                summary["metadata"] = {
                    "fps": metadata.get("fps"),
                    "total_num_frames": metadata.get("total_num_frames"),
                    "frames_indices_len": len(frames_indices) if frames_indices is not None else None,
                    "video_backend": metadata.get("video_backend"),
                }
            video_summaries.append(summary)

        return {
            "type": "video",
            "num_videos": len(mm_data["video"]),
            "videos": video_summaries,
        }

    return {"type": "unknown", "keys": sorted(mm_data.keys())}


def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    video_frame: Optional[int] = None,
    return_video_metadata: bool = False,
) -> Dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            if return_video_metadata:
                processed_video, processed_video_metadata = decode_video_with_cv2(
                    video,
                    video_fps,
                    video_frame,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                videos.append((processed_video, processed_video_metadata))
            else:
                videos.append(
                    process_video(
                        video,
                        min_pixels,
                        max_pixels,
                        video_fps,
                        video_frame,
                        return_metadata=return_video_metadata,
                    )
                )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.debug_vllm_inputs = os.getenv("VERL_DEBUG_VLLM_INPUTS", "0") == "1"
        self.debug_vllm_inputs_max_samples = int(os.getenv("VERL_DEBUG_VLLM_INPUTS_MAX_SAMPLES", "2"))
        self._remaining_debug_batches = int(os.getenv("VERL_DEBUG_VLLM_INPUTS_MAX_BATCHES", "2"))
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            prompts.meta_info.get("video_frame", None),
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        if self.debug_vllm_inputs and self._remaining_debug_batches > 0:
            prompt_lengths = attention_mask.sum(-1).detach().cpu().tolist()
            debug_limit = min(len(vllm_inputs), self.debug_vllm_inputs_max_samples)
            print(
                "[vllm-debug] generate batch summary: "
                f"rank={self.rank}, "
                f"batch_size={len(vllm_inputs)}, "
                f"prompt_tensor_len_min={min(prompt_lengths)}, "
                f"prompt_tensor_len_max={max(prompt_lengths)}, "
                f"prompt_tensor_len_mean={float(np.mean(prompt_lengths)):.2f}, "
                f"response_length={self.config.response_length}, "
                f"max_model_len={self.config.max_model_len or self.config.prompt_length + self.config.response_length}, "
                f"video_frame_meta={prompts.meta_info.get('video_frame', None)}, "
                f"min_pixels={prompts.meta_info.get('min_pixels', None)}, "
                f"max_pixels={prompts.meta_info.get('max_pixels', None)}",
                file=sys.stderr,
                flush=True,
            )
            for idx in range(debug_limit):
                mm_summary = _summarize_vllm_multimodal(vllm_inputs[idx].get("multi_modal_data"))
                print(
                    "[vllm-debug] sample "
                    f"{idx}: raw_prompt_ids_len={len(vllm_inputs[idx]['prompt_token_ids'])}, "
                    f"train_prompt_tensor_len={int(prompt_lengths[idx])}, "
                    f"multimodal={mm_summary}",
                    file=sys.stderr,
                    flush=True,
                )
            self._remaining_debug_batches -= 1

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
