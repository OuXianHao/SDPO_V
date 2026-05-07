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
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from .base import BaseRollout
from .config import RolloutConfig

_FRAME_PATH_VIDEO_DEBUG_COUNT = 0
_FRAME_PATH_VIDEO_DEBUG_MAX = 5


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    global _FRAME_PATH_VIDEO_DEBUG_COUNT
    # may convert image/video paths to preprocessed multimodal objects
    images, videos = [], []

    debug_frame_path_video = os.getenv("EASYR1_DEBUG_FRAME_PATH_VIDEO", "0") == "1"
    video_is_frame_paths = bool(multi_modal_data.get("video_is_frame_paths", False))

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        if video_is_frame_paths:
            frame_paths = multi_modal_data["videos"]
            processed_frames = [process_image(frame_path, min_pixels, max_pixels) for frame_path in frame_paths]
            if return_video_metadata:
                # vLLM Qwen3-VL expects each video item as a tuple:
                # (frames, metadata_dict). If metadata is missing/None, vLLM
                # accesses metadata.get(...) and crashes.
                # For frame-path datasets, frames are already sampled/extracted.
                frame_count = len(processed_frames)
                fps = float(multi_modal_data.get("video_fps", video_fps))
                video_metadata = {
                    "fps": fps,
                    "video_fps": fps,
                    "num_frames": frame_count,
                    "total_num_frames": frame_count,
                    "nframes": frame_count,
                    "do_sample_frames": False,
                }
                videos.append((processed_frames, video_metadata))
            else:
                videos.append(processed_frames)

            if debug_frame_path_video and _FRAME_PATH_VIDEO_DEBUG_COUNT < _FRAME_PATH_VIDEO_DEBUG_MAX:
                first_frame = frame_paths[0] if len(frame_paths) > 0 else None
                final_type = type(processed_frames).__name__
                final_len = len(processed_frames)
                first_frame_type = type(processed_frames[0]).__name__ if final_len > 0 else None
                metadata_obj = videos[-1][1] if return_video_metadata else None
                metadata_type = type(metadata_obj).__name__ if metadata_obj is not None else None
                metadata_keys = sorted(metadata_obj.keys()) if isinstance(metadata_obj, dict) else None
                print(
                    "[EASYR1_DEBUG_FRAME_PATH_VIDEO] "
                    f"video_is_frame_paths={video_is_frame_paths}, "
                    f"num_frame_paths={len(frame_paths)}, "
                    f"first_frame_path={first_frame}, "
                    f"final_video_container={final_type}, "
                    f"final_video_len={final_len}, "
                    f"first_processed_frame_type={first_frame_type}, "
                    f"metadata_type={metadata_type}, "
                    f"metadata_keys={metadata_keys}, "
                    f"fps={metadata_obj.get('fps') if isinstance(metadata_obj, dict) else None}, "
                    f"video_fps={metadata_obj.get('video_fps') if isinstance(metadata_obj, dict) else None}, "
                    f"num_frames={metadata_obj.get('num_frames') if isinstance(metadata_obj, dict) else None}, "
                    f"total_num_frames={metadata_obj.get('total_num_frames') if isinstance(metadata_obj, dict) else None}, "
                    f"do_sample_frames={metadata_obj.get('do_sample_frames') if isinstance(metadata_obj, dict) else None}, "
                    f"final_mm_keys={list({'video': videos}.keys())}"
                )
                _FRAME_PATH_VIDEO_DEBUG_COUNT += 1
        else:
            for video in multi_modal_data["videos"]:
                processed_video = process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
                # Keep metadata tuples for Qwen3-VL path. vLLM's Qwen3-VL video
                # processor consumes metadata (e.g. do_sample_frames / fps info)
                # from the tuple payload. Stripping to only frames makes metadata
                # become None downstream and crashes in qwen3_vl.py.
                if isinstance(processed_video, tuple) and not return_video_metadata:
                    videos.append(processed_video[0])
                else:
                    videos.append(processed_video)

            if debug_frame_path_video and _FRAME_PATH_VIDEO_DEBUG_COUNT < _FRAME_PATH_VIDEO_DEBUG_MAX:
                print(
                    "[EASYR1_DEBUG_FRAME_PATH_VIDEO] "
                    f"video_is_frame_paths={video_is_frame_paths}, "
                    f"num_frame_paths=0, "
                    f"first_frame_path=None, "
                    f"final_video_container={type(videos).__name__}, "
                    f"final_video_len={len(videos)}"
                )
                _FRAME_PATH_VIDEO_DEBUG_COUNT += 1

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
        **kwargs,
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
        # Qwen3-VL in vLLM expects per-video metadata in multimodal payload.
        # Keep metadata enabled only for Qwen3VLProcessor to avoid changing
        # behavior for other processors.
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

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
            load_format="dummy" if not self.lora_kwargs else "safetensors",
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
            **lora_kwargs,
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
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            effective_n = self.sampling_params.n
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            expected_response_num = len(vllm_inputs) * effective_n
            if len(response_ids) != expected_response_num:
                raise RuntimeError(
                    "vLLM rollout returned unexpected number of responses. "
                    f"effective_n={effective_n}, prompts={len(vllm_inputs)}, "
                    f"expected_responses={expected_response_num}, actual_responses={len(response_ids)}"
                )
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if effective_n > 1:
                batch_size = batch_size * effective_n
                input_ids = _repeat_interleave(input_ids, effective_n)
                attention_mask = _repeat_interleave(attention_mask, effective_n)
                position_ids = _repeat_interleave(position_ids, effective_n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, effective_n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
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
