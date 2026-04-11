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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from jinja2 import Template
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


GUIDELINE_SUMMARY_PROMPT = """You are given multiple candidate reasoning traces for the same video multiple-choice question. Some are correct and some are incorrect.

Your task is to analyze these rollouts based on the provided video, and produce a feedback guideline that can improve future reasoning on this sample.

Instructions:
- Identify the key reasoning patterns used by correct rollouts.
- Identify the typical mistakes made by incorrect rollouts.
- Focus on how to inspect the video evidence better.
- Pay special attention to temporal sequence, transitions, object interactions, state changes, repeated actions, and mismatches between the video and the options.
- Do NOT say which option is correct.
- Avoid vague advice such as "reason carefully" or "pay attention to the video".
- Make each guidance point concrete and operational.
- If any incorrect rollout has format issues (missing <answer> tags, extra text after </answer>, multiple answer blocks), include a guidance point about strict output format compliance.
- The final "Actionable guidance" section should be the most important part, because it will be reused as guiding feedback for later reasoning.
- Provide a concise guideline within 5 sentences that captures only the core reasoning strategy.

Output format:

<guideline_analysis>
Strengths of correct reasoning:
- ...
- ...

Common mistakes to avoid:
- ...
- ...
</guideline_analysis>

<guideline_feedback>
- ...
- ...
- ...
- ...
</guideline_feedback>

Question:
{question}

Options:
{options}

Candidate rollouts:

{rollout_blocks}

Now write the guideline.
"""


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.loss_mode = config.algorithm.loss_mode
        print(f"[trainer] selected loss_mode={self.loss_mode}")
        if self.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
            print(
                f"[trainer] sdpo settings: topk={config.algorithm.sdpo_topk}, "
                f"divergence={config.algorithm.sdpo_divergence}, use_tail={config.algorithm.sdpo_use_tail}, "
                f"approx_mode={config.algorithm.sdpo_approx_mode}, "
                f"feedback_mode={config.algorithm.sdpo_feedback_mode}"
            )
            if config.algorithm.sdpo_v_enabled:
                print(
                    f"[trainer] sdpo_v ENABLED: weight={config.algorithm.sdpo_v_weight}, "
                    f"margin={config.algorithm.sdpo_v_margin}, topk={config.algorithm.sdpo_v_topk}, "
                    f"bad_video_mode={config.algorithm.sdpo_v_bad_video_mode}, "
                    f"calibration={config.algorithm.sdpo_v_calibration}"
                )
        if self.loss_mode == "dapo_with_sdpo":
            print(
                f"[trainer] dapo_with_sdpo weights: lambda_dapo={config.algorithm.lambda_dapo}, "
                f"lambda_sdpo_t={config.algorithm.lambda_sdpo_t}, "
                f"lambda_sdpo_v={config.algorithm.lambda_sdpo_v}"
            )

        # define KL control
        # dapo_with_sdpo uses filter-disabled DAPO: no KL, no reference policy
        if config.algorithm.disable_kl or self.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            if config.algorithm.disable_kl:
                print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if self.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
            self.use_critic = False
        elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if self.loss_mode in ("grpo_on_policy", "dapo_with_sdpo") and config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.algorithm.loss_mode not in ("grpo_on_policy", "sdpo_logit", "dapo_with_sdpo"):
            raise ValueError("algorithm.loss_mode must be one of {`grpo_on_policy`, `sdpo_logit`, `dapo_with_sdpo`}.")
        print(f"[trainer] teacher feedback fields enabled={config.algorithm.loss_mode in ('sdpo_logit', 'dapo_with_sdpo')}")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            self.loss_mode in ("grpo_on_policy", "dapo_with_sdpo")
            and config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if self.loss_mode == "grpo_on_policy" and config.worker.actor.ppo_epochs != 1:
            raise ValueError("grpo_on_policy requires actor.ppo_epochs=1 for strict on-policy single update.")
        if self.loss_mode == "sdpo_logit" and config.worker.actor.ppo_epochs != 1:
            raise ValueError("sdpo_logit requires actor.ppo_epochs=1 for clean standalone online SDPO.")
        if self.loss_mode == "dapo_with_sdpo" and config.worker.actor.ppo_epochs != 1:
            raise ValueError("dapo_with_sdpo requires actor.ppo_epochs=1 for on-policy combined training.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _build_feedback_text(self, batch: DataProto) -> np.ndarray:
        response_texts = self._decode_response_texts(batch)
        feedback_texts = []
        for i, response_text in enumerate(response_texts):
            score = float(batch.batch["token_level_scores"][i].sum().item())
            feedback = f"Response quality score: {score:.4f}. Improve correctness and faithfulness.\n[Response]: {response_text}"
            feedback_texts.append(feedback)

        return np.array(feedback_texts, dtype=object)

    def _decode_response_texts(self, batch: DataProto) -> np.ndarray:
        response_ids = batch.batch["responses"]
        response_lengths = torch.sum(batch.batch["response_mask"], dim=-1)
        response_texts = []
        for i in range(len(batch)):
            cur_len = int(response_lengths[i].item())
            response_text = self.tokenizer.decode(response_ids[i][:cur_len], skip_special_tokens=True)
            response_texts.append(response_text)
        return np.array(response_texts, dtype=object)

    def _build_teacher_prompt_text_scalar(self, batch: DataProto) -> np.ndarray:
        format_prompt = self.config.worker.actor.teacher_format_prompt
        raw_prompt_texts = batch.non_tensor_batch["raw_prompt_text"]
        feedback_texts = batch.non_tensor_batch["feedback_text"]

        teacher_prompt_texts = []
        for raw_prompt_text, feedback_text in zip(raw_prompt_texts, feedback_texts):
            content_text = f"{raw_prompt_text}\n\n[Feedback]: {feedback_text}"
            if format_prompt is None or format_prompt == "":
                teacher_prompt_text = content_text
            else:
                teacher_prompt_text = Template(format_prompt.strip()).render(content=content_text)
            teacher_prompt_texts.append(teacher_prompt_text)

        return np.array(teacher_prompt_texts, dtype=object)

    @staticmethod
    def _collect_successful_rollout_indices_per_uid(
        uids: np.ndarray, accuracy_reward: np.ndarray
    ) -> dict[Any, list[int]]:
        uid2successful_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, (uid, reward) in enumerate(zip(uids, accuracy_reward)):
            if np.isclose(float(reward), 1.0):
                uid2successful_indices[uid].append(idx)
        return uid2successful_indices

    def _sample_successful_sibling_texts(
        self, uids: np.ndarray, accuracy_reward: np.ndarray, response_texts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        uid2successful_indices = self._collect_successful_rollout_indices_per_uid(uids=uids, accuracy_reward=accuracy_reward)
        sampled_successful_texts: list[Optional[str]] = []
        has_successful_sibling = np.zeros(len(uids), dtype=bool)

        for idx, uid in enumerate(uids):
            if np.isclose(float(accuracy_reward[idx]), 1.0):
                sampled_successful_texts.append(None)
                continue

            successful_candidates = [candidate_idx for candidate_idx in uid2successful_indices.get(uid, []) if candidate_idx != idx]
            if len(successful_candidates) == 0:
                sampled_successful_texts.append(None)
                continue

            sampled_idx = int(np.random.choice(successful_candidates))
            sampled_successful_texts.append(str(response_texts[sampled_idx]))
            has_successful_sibling[idx] = True

        return np.array(sampled_successful_texts, dtype=object), has_successful_sibling

    def _build_teacher_prompt_text_successful_rollout(
        self, raw_prompt_texts: np.ndarray, successful_sibling_texts: np.ndarray
    ) -> np.ndarray:
        format_prompt = self.config.worker.actor.teacher_format_prompt
        teacher_prompt_texts = []
        for raw_prompt_text, successful_sibling_text in zip(raw_prompt_texts, successful_sibling_texts):
            if successful_sibling_text is None:
                content_text = str(raw_prompt_text)
            else:
                content_text = (
                    f"{raw_prompt_text}\n\n"
                    "Correct solution:\n\n"
                    f"{successful_sibling_text}\n\n"
                    "Correctly solve the original question.\n"
                    "Respond concisely. Your thinking should be brief and focused — "
                    "identify the core logic, skip trivial steps, and avoid verbose or redundant thinking. "
                    "Place reasoning in <thinking> tags and the final answer in <answer> tags."
                )
            if format_prompt is None or format_prompt == "":
                teacher_prompt_text = content_text
            else:
                teacher_prompt_text = Template(format_prompt.strip()).render(content=content_text)
            teacher_prompt_texts.append(teacher_prompt_text)

        return np.array(teacher_prompt_texts, dtype=object)

    # ---- Guideline-based feedback helpers (SDPO-T only) ----

    def _select_mixed_rollouts_per_uid(
        self,
        uids: np.ndarray,
        accuracy_reward: np.ndarray,
        response_texts: np.ndarray,
        max_rollouts: int = 5,
    ) -> dict[str, list[tuple[str, bool]]]:
        """Select up to ``max_rollouts`` mixed (correct+incorrect) rollouts per uid for guideline generation.

        Args:
            max_rollouts: maximum number of rollouts to include per uid.
                Default 5 preserves the original behaviour.  The teacher-
                reweight path passes 3 to keep guideline prompts compact.
        """
        uid2rollouts: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            is_correct = bool(np.isclose(float(accuracy_reward[idx]), 1.0))
            uid2rollouts[uid].append((str(response_texts[idx]), is_correct))

        result: dict[str, list[tuple[str, bool]]] = {}
        for uid, rollouts in uid2rollouts.items():
            correct = [r for r in rollouts if r[1]]
            incorrect = [r for r in rollouts if not r[1]]
            if not correct or not incorrect:
                continue
            n_target = min(max_rollouts, len(correct) + len(incorrect))
            n_correct = max(1, min(len(correct), n_target // 2))
            n_incorrect = n_target - n_correct
            if n_incorrect > len(incorrect):
                n_incorrect = len(incorrect)
                n_correct = n_target - n_incorrect
            result[uid] = correct[:n_correct] + incorrect[:n_incorrect]
        return result

    @staticmethod
    def _parse_guideline_feedback(text: str) -> Optional[str]:
        """Extract <guideline_feedback>...</guideline_feedback> block from generated text."""
        match = re.search(r"<guideline_feedback>(.*?)</guideline_feedback>", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content if content else None
        return None

    def _generate_guidelines(
        self,
        batch: DataProto,
        uid_mixed_rollouts: dict[str, list[tuple[str, bool]]],
    ) -> dict[str, Optional[str]]:
        """Generate guideline feedback for each qualifying uid group using the rollout engine."""
        from tensordict import TensorDict

        uids = batch.non_tensor_batch["uid"]
        raw_prompt_texts = batch.non_tensor_batch["raw_prompt_text"]
        multi_modal_data_all = batch.non_tensor_batch.get("multi_modal_data", None)

        # Map uid -> first index in batch (all samples in a uid share the same question/video)
        uid2first_idx: dict[str, int] = {}
        for idx, uid in enumerate(uids):
            if uid not in uid2first_idx:
                uid2first_idx[uid] = int(idx)

        uid_order: list[str] = []
        raw_prompt_ids_list: list[list[int]] = []
        gen_multi_modal_data: list = []

        apply_template = (
            self.processor.apply_chat_template if self.processor is not None else self.tokenizer.apply_chat_template
        )

        for uid, rollouts in uid_mixed_rollouts.items():
            first_idx = uid2first_idx[uid]
            raw_prompt_text = str(raw_prompt_texts[first_idx])

            # Build rollout blocks (truncate each rollout to prevent
            # progressive prompt bloat as student outputs get longer/messier)
            _MAX_ROLLOUT_CHARS = 3072
            blocks = []
            for i, (text, is_correct) in enumerate(rollouts):
                label = "CORRECT" if is_correct else "INCORRECT"
                truncated = text[:_MAX_ROLLOUT_CHARS] + "..." if len(text) > _MAX_ROLLOUT_CHARS else text
                blocks.append(f"--- Rollout {i + 1} [{label}] ---\n{truncated}")
            rollout_blocks_text = "\n\n".join(blocks)

            question_text = raw_prompt_text.replace("<video>", "").strip()
            guideline_body = GUIDELINE_SUMMARY_PROMPT.format(
                question=question_text,
                options="(included in question above)",
                rollout_blocks=rollout_blocks_text,
            )

            # Prepend video marker if the original prompt uses video
            content = f"<video>\n{guideline_body}" if "<video>" in raw_prompt_text else guideline_body
            messages = [{"role": "user", "content": content}]
            prompt_str = apply_template(messages, add_generation_prompt=True, tokenize=False)
            raw_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

            uid_order.append(uid)
            raw_prompt_ids_list.append(raw_ids)
            gen_multi_modal_data.append(multi_modal_data_all[first_idx] if multi_modal_data_all is not None else None)

        if not uid_order:
            return {}

        # Build simplified input_ids / attention_mask / position_ids for DataProto
        # (vLLM generation uses raw_prompt_ids + multi_modal_data, not these tensors)
        max_len = max(len(ids) for ids in raw_prompt_ids_list)
        pad_token_id = self.tokenizer.pad_token_id or 0
        padded_ids, padded_attn, padded_pos = [], [], []
        for raw_ids in raw_prompt_ids_list:
            ids_t = torch.tensor(raw_ids, dtype=torch.long)
            pad_len = max_len - ids_t.size(0)
            if pad_len > 0:
                padded_ids.append(torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), ids_t]))
                padded_attn.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones_like(ids_t)]))
                padded_pos.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.arange(ids_t.size(0))]))
            else:
                padded_ids.append(ids_t)
                padded_attn.append(torch.ones_like(ids_t))
                padded_pos.append(torch.arange(ids_t.size(0)))

        has_mm = any(d is not None for d in gen_multi_modal_data)
        non_tensor = {"raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object)}
        if has_mm:
            non_tensor["multi_modal_data"] = np.array(gen_multi_modal_data, dtype=object)

        gen_batch = DataProto(
            batch=TensorDict(
                {"input_ids": torch.stack(padded_ids), "attention_mask": torch.stack(padded_attn), "position_ids": torch.stack(padded_pos)},
                batch_size=[len(uid_order)],
            ),
            non_tensor_batch=non_tensor,
            meta_info={
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
                "temperature": 0.0,
                "n": 1,
            },
        )

        gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)

        # Use EMA teacher / ref weights for guideline generation
        self.actor_rollout_ref_wg.prepare_teacher_rollout_engine()
        try:
            gen_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        finally:
            self.actor_rollout_ref_wg.release_teacher_rollout_engine()

        if pad_size > 0:
            gen_output = unpad_dataproto(gen_output, pad_size)

        # Decode responses and parse guideline blocks
        response_ids = gen_output.batch["responses"]
        response_mask = gen_output.batch["response_mask"]
        uid_guidelines: dict[str, Optional[str]] = {}
        num_parse_success = 0
        num_parse_fail = 0
        guideline_lengths: list[int] = []
        for i, uid in enumerate(uid_order):
            resp_len = int(response_mask[i].sum().item())
            resp_text = self.tokenizer.decode(response_ids[i][:resp_len], skip_special_tokens=True)
            parsed = self._parse_guideline_feedback(resp_text)
            uid_guidelines[uid] = parsed
            if parsed is not None:
                num_parse_success += 1
                guideline_lengths.append(len(parsed))
            else:
                num_parse_fail += 1

        # Store diagnostics for logging (attached to self for retrieval by caller)
        self._guideline_diag = {
            "sdpo/guideline_parse_success": float(num_parse_success),
            "sdpo/guideline_parse_fail": float(num_parse_fail),
            "sdpo/guideline_avg_length": float(sum(guideline_lengths) / max(len(guideline_lengths), 1)),
        }

        return uid_guidelines

    def _build_teacher_prompt_text_guideline(
        self,
        raw_prompt_texts: np.ndarray,
        uids: np.ndarray,
        accuracy_reward: np.ndarray,
        uid_guidelines: dict[str, Optional[str]],
        successful_sibling_texts: np.ndarray,
        has_successful_sibling: np.ndarray,
        ground_truths: Optional[np.ndarray] = None,
        correct_hint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Build teacher prompt text using guideline feedback, with fallback to successful rollout.

        When ``correct_hint=True`` and ``ground_truths`` is provided, a
        lightweight correct-option hint is appended to the guideline/fallback
        teacher prompts for incorrect rollouts.  This strengthens the
        privileged teacher's guidance on the correct outcome without serving
        as a direct supervision target.
        """
        import re as _re

        format_prompt = self.config.worker.actor.teacher_format_prompt
        teacher_prompt_texts = []
        sample_valid_mask = np.zeros(len(uids), dtype=bool)
        guideline_used = 0
        fallback_used = 0
        hint_applied = 0

        for idx in range(len(uids)):
            uid = uids[idx]
            raw_prompt_text = str(raw_prompt_texts[idx])
            is_correct = bool(np.isclose(float(accuracy_reward[idx]), 1.0))

            # ---- Extract correct-option text (used for hint) ----
            correct_option_text = None
            if correct_hint and ground_truths is not None and not is_correct:
                gt_raw = str(ground_truths[idx])
                m = _re.search(r"<answer>(.*?)</answer>", gt_raw, _re.DOTALL | _re.IGNORECASE)
                correct_option_text = m.group(1).strip() if m else gt_raw.strip()

            if is_correct:
                content_text = raw_prompt_text
            elif uid in uid_guidelines and uid_guidelines[uid] is not None:
                hint_line = ""
                if correct_option_text:
                    hint_line = f"The known correct final option is: {correct_option_text}\n"
                    hint_applied += 1
                content_text = (
                    f"{raw_prompt_text}\n\n"
                    "Here is some guidance that may help you solve the problem:\n\n"
                    f"{uid_guidelines[uid]}\n\n"
                    f"{hint_line}"
                    "Correctly solve the original question.\n"
                    "Respond concisely. Your thinking should be brief and focused — "
                    "identify the core logic, skip trivial steps, and avoid verbose or redundant thinking.\n"
                    "Place reasoning in <thinking> tags and the final answer in <answer> tags.\n"
                    "Do not output any text outside the <thinking> and <answer> tags. "
                    "Do not output any text after </answer>."
                )
                sample_valid_mask[idx] = True
                guideline_used += 1
            elif has_successful_sibling[idx]:
                hint_line = ""
                if correct_option_text:
                    hint_line = f"The known correct final option is: {correct_option_text}\n"
                    hint_applied += 1
                content_text = (
                    f"{raw_prompt_text}\n\n"
                    "Correct solution:\n\n"
                    f"{successful_sibling_texts[idx]}\n\n"
                    f"{hint_line}"
                    "Correctly solve the original question.\n"
                    "Respond concisely. Your thinking should be brief and focused — "
                    "identify the core logic, skip trivial steps, and avoid verbose or redundant thinking. "
                    "Place reasoning in <thinking> tags and the final answer in <answer> tags."
                )
                sample_valid_mask[idx] = True
                fallback_used += 1
            else:
                content_text = raw_prompt_text

            if format_prompt and format_prompt != "":
                teacher_prompt_text = Template(format_prompt.strip()).render(content=content_text)
            else:
                teacher_prompt_text = content_text
            teacher_prompt_texts.append(teacher_prompt_text)

        guideline_metrics = {
            "sdpo/guideline_used_samples": float(guideline_used),
            "sdpo/guideline_fallback_successful_rollout": float(fallback_used),
            "sdpo/guideline_correct_hint_applied": float(hint_applied),
        }
        return np.array(teacher_prompt_texts, dtype=object), sample_valid_mask, guideline_metrics

    def _attach_sdpo_fields(self, batch: DataProto, mode: str = "scalar_text") -> tuple[DataProto, dict[str, float]]:
        if "accuracy_reward" not in batch.non_tensor_batch:
            raise KeyError(
                "Missing `accuracy_reward` in batch.non_tensor_batch. "
                "SDPO skip-by-group requires binary accuracy reward from reward function."
            )

        uids = batch.non_tensor_batch["uid"]
        accuracy_reward = np.asarray(batch.non_tensor_batch["accuracy_reward"], dtype=np.float32)
        if len(accuracy_reward) != len(batch):
            raise ValueError("`accuracy_reward` length must match batch size.")

        response_mask = batch.batch["response_mask"].clone().bool()
        batch.batch["response_token_mask"] = response_mask

        if mode == "scalar_text":
            # token-level mask consumed by sdpo_logit loss
            sdpo_valid_mask = torch.ones_like(response_mask, dtype=torch.bool)
            uid2indices = defaultdict(list)
            for idx, uid in enumerate(uids):
                uid2indices[uid].append(idx)

            num_all_correct_skipped = 0
            num_all_wrong_skipped = 0
            num_mixed_kept = 0

            for indices in uid2indices.values():
                group_acc = accuracy_reward[indices]
                all_correct = bool(np.all(np.isclose(group_acc, 1.0)))
                all_wrong = bool(np.all(np.isclose(group_acc, 0.0)))

                if all_correct:
                    sdpo_valid_mask[indices] = False
                    num_all_correct_skipped += 1
                elif all_wrong:
                    sdpo_valid_mask[indices] = False
                    num_all_wrong_skipped += 1
                else:
                    num_mixed_kept += 1

            feedback_text = self._build_feedback_text(batch)
            batch.non_tensor_batch["feedback_text"] = feedback_text
            batch.non_tensor_batch["teacher_prompt_text"] = self._build_teacher_prompt_text_scalar(batch)
            batch.batch["sdpo_valid_mask"] = sdpo_valid_mask

            # ---- SDPO-V validity mask: samples with accuracy_reward == 1 ----
            # SDPO-V trains on correct samples only (reward == 1), which is
            # different from SDPO-T's mixed-group gating logic.
            # This mask is per-sample, broadcast to token level.
            sample_reward_is_1 = np.isclose(accuracy_reward, 1.0)  # (batch_size,)
            sdpo_v_valid_mask = (
                response_mask
                & torch.from_numpy(sample_reward_is_1).to(response_mask.device).unsqueeze(-1)
            )
            batch.batch["sdpo_v_valid_mask"] = sdpo_v_valid_mask

            sdpo_skip_metrics = {
                "sdpo/num_all_correct_skipped": float(num_all_correct_skipped),
                "sdpo/num_all_wrong_skipped": float(num_all_wrong_skipped),
                "sdpo/num_mixed_kept": float(num_mixed_kept),
                "sdpo_v/num_reward_1_samples": float(sample_reward_is_1.sum()),
            }
            return batch, sdpo_skip_metrics

        if mode == "successful_rollout":
            response_texts = self._decode_response_texts(batch)
            successful_sibling_texts, has_successful_sibling = self._sample_successful_sibling_texts(
                uids=uids,
                accuracy_reward=accuracy_reward,
                response_texts=response_texts,
            )
            failed_mask = accuracy_reward != 1.0
            sample_valid_mask = failed_mask & has_successful_sibling

            batch.non_tensor_batch["teacher_prompt_text"] = self._build_teacher_prompt_text_successful_rollout(
                raw_prompt_texts=batch.non_tensor_batch["raw_prompt_text"],
                successful_sibling_texts=successful_sibling_texts,
            )
            # Keep feedback text key for backward compatibility in downstream logging.
            batch.non_tensor_batch["feedback_text"] = np.array(["" for _ in range(len(batch))], dtype=object)
            batch.non_tensor_batch["successful_sibling_text"] = successful_sibling_texts
            batch.batch["sdpo_valid_mask"] = response_mask & torch.from_numpy(sample_valid_mask).to(response_mask.device).unsqueeze(-1)

            # ---- SDPO-V validity mask: samples with accuracy_reward == 1 ----
            sample_reward_is_1 = np.isclose(accuracy_reward, 1.0)
            sdpo_v_valid_mask = (
                response_mask
                & torch.from_numpy(sample_reward_is_1).to(response_mask.device).unsqueeze(-1)
            )
            batch.batch["sdpo_v_valid_mask"] = sdpo_v_valid_mask

            sdpo_skip_metrics = {
                "sdpo/successful_rollout_failed_samples": float(failed_mask.sum().item()),
                "sdpo/successful_rollout_kept_samples": float(sample_valid_mask.sum().item()),
                "sdpo/successful_rollout_skipped_no_success_sibling": float((failed_mask & ~has_successful_sibling).sum().item()),
                "sdpo/successful_rollout_skipped_self_success": float((~failed_mask).sum().item()),
                "sdpo_v/num_reward_1_samples": float(sample_reward_is_1.sum()),
            }
            return batch, sdpo_skip_metrics

        if mode == "guideline_mixed_rollouts":
            response_texts = self._decode_response_texts(batch)

            # Try guideline generation for uid groups with mixed correct/incorrect rollouts.
            # When teacher_reweight is active, use a smaller mixed set (3) to keep
            # guideline prompts compact — the teacher signal is used for token-level
            # credit reweighting, not for detailed distillation.
            _max_rollouts = 3 if getattr(self.config.algorithm, "teacher_reweight_enabled", False) else 5
            uid_mixed_rollouts = self._select_mixed_rollouts_per_uid(
                uids, accuracy_reward, response_texts, max_rollouts=_max_rollouts,
            )
            uid_guidelines: dict[str, Optional[str]] = {}
            if uid_mixed_rollouts:
                try:
                    uid_guidelines = self._generate_guidelines(batch, uid_mixed_rollouts)
                except Exception as e:
                    print(f"[sdpo-guideline] Generation failed ({e}), falling back to successful_rollout for all.")
                    uid_guidelines = {}

            # Compute successful sibling texts for fallback
            successful_sibling_texts, has_successful_sibling = self._sample_successful_sibling_texts(
                uids=uids, accuracy_reward=accuracy_reward, response_texts=response_texts,
            )

            _correct_hint = getattr(self.config.algorithm, "teacher_reweight_correct_hint", False)
            teacher_prompt_texts, sample_valid_mask, guideline_metrics = self._build_teacher_prompt_text_guideline(
                raw_prompt_texts=batch.non_tensor_batch["raw_prompt_text"],
                uids=uids,
                accuracy_reward=accuracy_reward,
                uid_guidelines=uid_guidelines,
                successful_sibling_texts=successful_sibling_texts,
                has_successful_sibling=has_successful_sibling,
                ground_truths=batch.non_tensor_batch.get("ground_truth"),
                correct_hint=_correct_hint,
            )

            batch.non_tensor_batch["teacher_prompt_text"] = teacher_prompt_texts
            batch.non_tensor_batch["feedback_text"] = np.array(["" for _ in range(len(batch))], dtype=object)
            batch.non_tensor_batch["successful_sibling_text"] = successful_sibling_texts
            batch.batch["sdpo_valid_mask"] = (
                response_mask & torch.from_numpy(sample_valid_mask).to(response_mask.device).unsqueeze(-1)
            )

            # SDPO-V validity mask (unchanged from other modes)
            sample_reward_is_1 = np.isclose(accuracy_reward, 1.0)
            sdpo_v_valid_mask = (
                response_mask
                & torch.from_numpy(sample_reward_is_1).to(response_mask.device).unsqueeze(-1)
            )
            batch.batch["sdpo_v_valid_mask"] = sdpo_v_valid_mask

            failed_mask = accuracy_reward != 1.0
            sdpo_skip_metrics = {
                "sdpo/guideline_failed_samples": float(failed_mask.sum().item()),
                "sdpo/guideline_kept_samples": float(sample_valid_mask.sum().item()),
                "sdpo/guideline_skipped_self_success": float((~failed_mask).sum().item()),
                "sdpo_v/num_reward_1_samples": float(sample_reward_is_1.sum()),
            }
            sdpo_skip_metrics.update(guideline_metrics)
            # Attach guideline generation diagnostics if available
            if hasattr(self, "_guideline_diag"):
                sdpo_skip_metrics.update(self._guideline_diag)
                del self._guideline_diag
            return batch, sdpo_skip_metrics

        raise ValueError(f"Unsupported sdpo feedback mode: {mode}")

    def _attach_sdpo_fields_combined(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Attach SDPO fields for the dapo_with_sdpo combined mode.

        Key differences from _attach_sdpo_fields:
        - SDPO-T valid mask: mixed-group gating only (all rollouts in mixed groups participate).
          Unlike the legacy ``successful_rollout`` mode which only keeps wrong rollouts.
        - SDPO-V valid mask: all rollouts participate (no correct-only restriction).
          Only engineering safety masks (response_mask, empty-mask protection) are applied.
        - Teacher prompt construction uses the ``scalar_text`` path (feedback text with
          accuracy info) to keep it simple and compatible with all-rollout participation.
        """
        if "accuracy_reward" not in batch.non_tensor_batch:
            raise KeyError(
                "Missing `accuracy_reward` in batch.non_tensor_batch. "
                "dapo_with_sdpo requires binary accuracy reward from reward function."
            )

        uids = batch.non_tensor_batch["uid"]
        accuracy_reward = np.asarray(batch.non_tensor_batch["accuracy_reward"], dtype=np.float32)
        if len(accuracy_reward) != len(batch):
            raise ValueError("`accuracy_reward` length must match batch size.")

        response_mask = batch.batch["response_mask"].clone().bool()
        batch.batch["response_token_mask"] = response_mask

        # ---- SDPO-T valid mask: mixed-group gating, all rollouts in mixed groups ----
        sdpo_valid_mask = torch.ones_like(response_mask, dtype=torch.bool)
        uid2indices = defaultdict(list)
        for idx, uid in enumerate(uids):
            uid2indices[uid].append(idx)

        num_all_correct_skipped = 0
        num_all_wrong_skipped = 0
        num_mixed_kept = 0

        for indices in uid2indices.values():
            group_acc = accuracy_reward[indices]
            all_correct = bool(np.all(np.isclose(group_acc, 1.0)))
            all_wrong = bool(np.all(np.isclose(group_acc, 0.0)))

            if all_correct:
                sdpo_valid_mask[indices] = False
                num_all_correct_skipped += 1
            elif all_wrong:
                sdpo_valid_mask[indices] = False
                num_all_wrong_skipped += 1
            else:
                # Mixed group: ALL rollouts participate (both correct and incorrect)
                num_mixed_kept += 1

        # Build teacher feedback / prompt text using scalar_text path
        feedback_text = self._build_feedback_text(batch)
        batch.non_tensor_batch["feedback_text"] = feedback_text
        batch.non_tensor_batch["teacher_prompt_text"] = self._build_teacher_prompt_text_scalar(batch)
        batch.batch["sdpo_valid_mask"] = sdpo_valid_mask

        # ---- SDPO-V valid mask: ALL rollouts participate (widened) ----
        # Only engineering safety masks are applied: response_mask ensures
        # we only operate on valid response tokens. No correctness gating.
        sdpo_v_valid_mask = response_mask  # all rollouts with valid response tokens
        batch.batch["sdpo_v_valid_mask"] = sdpo_v_valid_mask

        sdpo_skip_metrics = {
            "sdpo/num_all_correct_skipped": float(num_all_correct_skipped),
            "sdpo/num_all_wrong_skipped": float(num_all_wrong_skipped),
            "sdpo/num_mixed_kept": float(num_mixed_kept),
            "sdpo/combined_mode_sdpo_t_valid_samples": float(sdpo_valid_mask.any(dim=-1).sum().item()),
            "sdpo_v/num_valid_samples": float(sdpo_v_valid_mask.any(dim=-1).sum().item()),
        }
        return batch, sdpo_skip_metrics

    def _export_problem_id_stats(self, batch: DataProto, global_step: int) -> dict[str, float]:
        """Aggregate rollout results by problem_id and export stats files.

        Writes three files into ``{save_checkpoint_path}/problem_id_stats/step_{global_step}/``:

        * ``valid_problem_ids.txt``   – one problem_id per line (mixed groups only)
        * ``problem_id_stats.jsonl``  – per-problem_id JSON records
        * ``summary.json``           – high-level counts and ratios

        Returns a small metrics dict for logging.
        """
        if "problem_id" not in batch.non_tensor_batch:
            print(
                "[problem_id_stats] WARNING: 'problem_id' not found in batch.non_tensor_batch. "
                "Skipping export. Make sure your parquet/dataset contains a 'problem_id' column."
            )
            return {}

        problem_ids = batch.non_tensor_batch["problem_id"]
        accuracy_reward = np.asarray(batch.non_tensor_batch["accuracy_reward"], dtype=np.float32)

        # Aggregate by problem_id
        pid2stats: dict[str, dict] = {}
        for pid, acc in zip(problem_ids, accuracy_reward):
            pid_key = str(pid)
            if pid_key not in pid2stats:
                pid2stats[pid_key] = {"problem_id": pid, "num_rollouts": 0, "num_correct": 0, "num_wrong": 0}
            pid2stats[pid_key]["num_rollouts"] += 1
            if float(acc) >= 1.0 - 1e-6:
                pid2stats[pid_key]["num_correct"] += 1
            else:
                pid2stats[pid_key]["num_wrong"] += 1

        # Classify each problem_id
        valid_pids = []
        all_correct_pids = []
        all_wrong_pids = []
        records = []
        for pid_key in sorted(pid2stats.keys()):
            entry = pid2stats[pid_key]
            nc, nw = entry["num_correct"], entry["num_wrong"]
            if nc > 0 and nw > 0:
                status = "valid"
                valid_pids.append(entry["problem_id"])
            elif nw == 0:
                status = "all_correct"
                all_correct_pids.append(entry["problem_id"])
            else:
                status = "all_wrong"
                all_wrong_pids.append(entry["problem_id"])
            records.append({
                "problem_id": entry["problem_id"],
                "num_rollouts": entry["num_rollouts"],
                "num_correct": nc,
                "num_wrong": nw,
                "status": status,
            })

        # Write files
        out_dir = os.path.join(self.config.trainer.save_checkpoint_path, "problem_id_stats", f"step_{global_step}")
        os.makedirs(out_dir, exist_ok=True)

        valid_path = os.path.join(out_dir, "valid_problem_ids.txt")
        with open(valid_path, "w", encoding="utf-8") as f:
            for pid in valid_pids:
                f.write(f"{pid}\n")

        stats_path = os.path.join(out_dir, "problem_id_stats.jsonl")
        with open(stats_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Ensure problem_id is serializable (convert numpy types)
                serializable_rec = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in rec.items()}
                f.write(json.dumps(serializable_rec, ensure_ascii=False) + "\n")

        total = len(pid2stats)
        n_valid = len(valid_pids)
        n_all_correct = len(all_correct_pids)
        n_all_wrong = len(all_wrong_pids)
        valid_ratio = n_valid / max(total, 1)

        summary = {
            "global_step": global_step,
            "total_problem_ids": total,
            "valid_problem_ids": n_valid,
            "all_correct_problem_ids": n_all_correct,
            "all_wrong_problem_ids": n_all_wrong,
            "valid_ratio": round(valid_ratio, 4),
        }
        summary_path = os.path.join(out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            f"[problem_id_stats] step={global_step} total={total} valid={n_valid} "
            f"all_correct={n_all_correct} all_wrong={n_all_wrong} valid_ratio={valid_ratio:.4f} "
            f"dir={out_dir}"
        )

        return {
            "problem_id_stats/total": float(total),
            "problem_id_stats/valid": float(n_valid),
            "problem_id_stats/all_correct": float(n_all_correct),
            "problem_id_stats/all_wrong": float(n_all_wrong),
            "problem_id_stats/valid_ratio": valid_ratio,
        }

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            if self.config.algorithm.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
                new_batch.non_tensor_batch["pad_token_id"] = np.array(
                    [self.tokenizer.pad_token_id for _ in range(len(new_batch.batch))], dtype=object
                )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            if self.config.algorithm.loss_mode in ("sdpo_logit", "dapo_with_sdpo") and "raw_prompt_ids" in gen_batch.non_tensor_batch:
                new_batch.non_tensor_batch["raw_prompt_ids"] = np.repeat(
                    gen_batch.non_tensor_batch["raw_prompt_ids"], self.config.worker.rollout.n
                )
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        batch.meta_info["global_step"] = self.global_step
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs (needed by on-policy GRPO / DAPO objective)
                if self.loss_mode in ("grpo_on_policy", "dapo_with_sdpo"):
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        if reward_tensor.size(0) != len(batch):
                            raise ValueError("Reward tensor batch size must align with expanded rollout batch.")
                        batch.batch["token_level_scores"] = reward_tensor
                        if self.config.algorithm.loss_mode in ("sdpo_logit", "dapo_with_sdpo"):
                            if "accuracy" not in reward_metrics:
                                raise KeyError(
                                    "SDPO skip-by-group requires reward metric `accuracy`. "
                                    "Please ensure the active reward function returns it."
                                )
                            batch.non_tensor_batch["accuracy_reward"] = np.asarray(
                                reward_metrics["accuracy"], dtype=np.float32
                            )
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    if self.config.algorithm.loss_mode == "sdpo_logit":
                        # In sdpo_logit mode, reward is used only to construct teacher feedback text,
                        # not to compute PPO/GRPO advantages or the main optimization objective.
                        batch, sdpo_skip_metrics = self._attach_sdpo_fields(
                            batch, mode=self.config.algorithm.sdpo_feedback_mode
                        )
                        metrics.update(sdpo_skip_metrics)
                        pid_stats_metrics = self._export_problem_id_stats(batch, global_step=self.global_step)
                        metrics.update(pid_stats_metrics)
                    elif self.config.algorithm.loss_mode == "dapo_with_sdpo":
                        # Combined mode: prepare BOTH DAPO advantages AND SDPO fields.
                        # 1) DAPO/GRPO advantages (filter-disabled: all groups participate)
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                        metrics["grpo/reward_mean"] = batch.batch["token_level_scores"].sum(dim=-1).float().mean().item()
                        metrics["grpo/adv_mean"] = batch.batch["advantages"].float().mean().item()
                        metrics["grpo/adv_std"] = batch.batch["advantages"].float().std().item()
                        # 2) SDPO fields.
                        # When teacher_reweight_enabled, use the full _attach_sdpo_fields
                        # path (which supports guideline_mixed_rollouts feedback mode)
                        # instead of the combined scalar_text shortcut.
                        if self.config.algorithm.teacher_reweight_enabled:
                            batch, sdpo_skip_metrics = self._attach_sdpo_fields(
                                batch, mode=self.config.algorithm.sdpo_feedback_mode
                            )
                        else:
                            batch, sdpo_skip_metrics = self._attach_sdpo_fields_combined(batch)
                        metrics.update(sdpo_skip_metrics)
                    else:
                        # grpo_on_policy mode
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                        metrics["grpo/reward_mean"] = batch.batch["token_level_scores"].sum(dim=-1).float().mean().item()
                        metrics["grpo/adv_mean"] = batch.batch["advantages"].float().mean().item()
                        metrics["grpo/adv_std"] = batch.batch["advantages"].float().std().item()

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    batch.meta_info["temperature"] = self.config.worker.rollout.temperature
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, loss_mode=self.loss_mode))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
