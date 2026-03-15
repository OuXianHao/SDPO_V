# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import os
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF
from ..utils.debug_dump import DEBUG_DUMP_WRITER


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    GRPO_PASSK = "grpo_passk"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


ADV_ESTIMATOR_MAP: dict[str, Any] = {}


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


def register_adv_estimator(name: AdvantageEstimator):
    """Decorator to register a advantage estimator function with a given name."""

    def decorator(fn):
        wrapped_fn = torch.no_grad()(fn)
        ADV_ESTIMATOR_MAP[getattr(name, "value", name)] = wrapped_fn
        return wrapped_fn

    return decorator


def compute_advantage_return(name: AdvantageEstimator, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantage and return for a given advantage estimator."""
    return ADV_ESTIMATOR_MAP[getattr(name, "value", name)](**kwargs)


@register_adv_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        gaelam = delta + gamma * lam * lastgaelam

        if response_mask[:, t]:  # skip values and TD-error on observation tokens
            nextvalues = values[:, t]
            lastgaelam = gaelam

        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        group_scores = torch.stack(id2score[idx])
        id2mean[idx] = group_scores.mean()
        id2std[idx] = group_scores.std()

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.GRPO_PASSK)
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)
        eps: `(float)`
            epsilon value to avoid division by zero

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(scores)
    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        rewards = torch.tensor(id2score[idx])
        topk, topk_idx = torch.topk(rewards, k=2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0]]
        advantages[i_max] = (r_max - r_second_max) / (torch.std(torch.tensor(id2score[idx])) + eps)

    returns = advantages.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_estimator(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@register_adv_estimator(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    advantages = (token_level_rewards.sum(dim=-1) - reward_baselines) * response_mask
    returns = (token_level_rewards * response_mask).flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    tau_positive: float,
    tau_negative: float,
    loss_type: Literal["default", "gspo", "gspo_token", "cispo", "sapo"],
    loss_avg_mode: Literal["token", "seq"],
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        tau_positive: (float)
            The temperature for control the positive tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        tau_negative: (float)
            The temperature for control the negative tokens' clipping in SAPO. See https://arxiv.org/pdf/2511.20347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    if loss_type in ["gspo", "gspo_token"]:
        # compute sequence-level importance ratio
        negative_approx_kl_in_seq = VF.masked_mean(negative_approx_kl, response_mask, dim=-1)
        # combined ratio at token level
        if loss_type == "gspo_token":
            log_importance_ratio = negative_approx_kl_in_seq.detach().unsqueeze(-1) + log_probs - log_probs.detach()
        else:
            log_importance_ratio = negative_approx_kl_in_seq.unsqueeze(-1) * response_mask
    else:
        log_importance_ratio = negative_approx_kl

    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    ratio = torch.exp(torch.clamp(log_importance_ratio, -20.0, 20.0))
    clipped_ratio = torch.exp(
        torch.clamp(log_importance_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    if loss_type == "cispo":
        final_pg_loss = -advantages * log_probs * clipped_ratio.detach()
    elif loss_type == "sapo":
        positive_token_mask =  (advantages >= 0).float()
        negative_token_mask =  (advantages < 0).float()
        gate_negative = 4.0 / tau_negative * torch.sigmoid(tau_negative * (ratio - 1.0))
        gate_positive = 4.0 / tau_positive * torch.sigmoid(tau_positive * (ratio - 1.0))
        final_pg_loss = -advantages * (positive_token_mask * gate_positive + negative_token_mask * gate_negative)
    else:
        pg_loss = -advantages * ratio  # -ratio * A
        pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
        pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

        clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
        metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
        clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
        final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
        metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_grpo_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics = compute_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        clip_ratio_dual=clip_ratio_dual,
        tau_positive=1.0,
        tau_negative=1.0,
        loss_type="default",
        loss_avg_mode=loss_avg_mode,
    )
    ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -20.0, 20.0))
    metrics = {
        "loss": loss.detach().item(),
        "ratio_mean": VF.masked_mean(ratio, response_mask).detach().item(),
        "clipfrac": metrics["pg_clipfrac_higher"],
    }
    return loss, metrics


def compute_sdpo_logit_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    topk: int = 100,
    divergence: Literal["forward_kl", "reverse_kl"] = "forward_kl",
    use_tail: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Standalone SDPO logit-level loss.

    This implementation uses a student top-K + aggregated-tail approximation
    rather than exact full-vocabulary KL over all logits.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape.")
    if student_logits.shape[:2] != response_mask.shape:
        raise ValueError("response_mask shape must match first two dims of logits.")
    if divergence not in ("forward_kl", "reverse_kl"):
        raise ValueError(f"Unsupported divergence: {divergence}")

    vocab_size = student_logits.size(-1)
    k = min(max(int(topk), 1), vocab_size)
    mask_f = response_mask.float()
    valid_count = mask_f.sum().clamp_min(1.0)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    topk_values, topk_indices = torch.topk(student_log_probs, k=k, dim=-1)
    student_topk_logp = topk_values
    student_topk_prob = student_topk_logp.exp()
    teacher_topk_logp = torch.gather(teacher_log_probs, dim=-1, index=topk_indices)
    teacher_topk_prob = teacher_topk_logp.exp()

    if divergence == "forward_kl":
        token_loss = (teacher_topk_prob * (teacher_topk_logp - student_topk_logp)).sum(dim=-1)
        full_token_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    else:
        token_loss = (student_topk_prob * (student_topk_logp - teacher_topk_logp)).sum(dim=-1)
        full_token_loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    student_topk_mass = student_topk_prob.sum(dim=-1)
    teacher_topk_mass = teacher_topk_prob.sum(dim=-1)
    student_tail = (1.0 - student_topk_mass).clamp_min(1e-12)
    teacher_tail = (1.0 - teacher_topk_mass).clamp_min(1e-12)

    if use_tail:
        if divergence == "forward_kl":
            token_loss = token_loss + teacher_tail * (torch.log(teacher_tail) - torch.log(student_tail))
        else:
            token_loss = token_loss + student_tail * (torch.log(student_tail) - torch.log(teacher_tail))

    loss = (token_loss * mask_f).sum() / valid_count
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)

    mass_ks = [1, 5, 10, 20, 50, 100]
    student_sorted_probs = torch.sort(student_probs, dim=-1, descending=True).values
    student_mass_metrics: dict[str, float] = {}
    for mass_k in mass_ks:
        cur_k = min(mass_k, vocab_size)
        mass_at_k = student_sorted_probs[..., :cur_k].sum(dim=-1)
        student_mass_metrics[f"mass@{mass_k}"] = (mass_at_k * mask_f).sum().detach().item() / valid_count.detach().item()

    student_cdf = torch.cumsum(student_sorted_probs, dim=-1)
    first_95 = (student_cdf >= 0.95).float().argmax(dim=-1) + 1
    first_99 = (student_cdf >= 0.99).float().argmax(dim=-1) + 1

    if DEBUG_DUMP_WRITER.should_dump() and os.getenv("SDPO_TOPK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}:
        max_positions = int(os.getenv("SDPO_TOPK_DEBUG_POSITIONS", "3"))
        valid_positions = torch.nonzero(response_mask, as_tuple=False)
        debug_positions = valid_positions[:max_positions]
        for row_idx, col_idx in debug_positions:
            b = int(row_idx.item())
            t = int(col_idx.item())
            teacher_pos_probs = teacher_probs[b, t]
            student_pos_probs = student_probs[b, t]
            teacher_pos_log_probs = teacher_log_probs[b, t]
            student_pos_log_probs = student_log_probs[b, t]
            teacher_top10_probs, teacher_top10_ids = torch.topk(teacher_pos_probs, k=min(10, vocab_size), dim=-1)
            student_top10_probs, student_top10_ids = torch.topk(student_pos_probs, k=min(10, vocab_size), dim=-1)

            teacher_sorted = torch.sort(teacher_pos_probs, descending=True).values
            student_sorted = torch.sort(student_pos_probs, descending=True).values
            mass_curves = {}
            for mass_k in mass_ks:
                cur_k = min(mass_k, vocab_size)
                mass_curves[f"teacher_mass@{mass_k}"] = float(teacher_sorted[:cur_k].sum().item())
                mass_curves[f"student_mass@{mass_k}"] = float(student_sorted[:cur_k].sum().item())

            if divergence == "forward_kl":
                full_pos_loss = float((teacher_pos_probs * (teacher_pos_log_probs - student_pos_log_probs)).sum().item())
            else:
                full_pos_loss = float((student_pos_probs * (student_pos_log_probs - teacher_pos_log_probs)).sum().item())

            debug_record = {
                "type": "sdpo_topk_debug",
                "batch_index": b,
                "token_index": t,
                "topk": int(k),
                "selected_topk_token_ids": topk_indices[b, t].detach().cpu().tolist(),
                "teacher_top10_ids": teacher_top10_ids.detach().cpu().tolist(),
                "teacher_top10_probs": teacher_top10_probs.detach().cpu().tolist(),
                "student_top10_ids": student_top10_ids.detach().cpu().tolist(),
                "student_top10_probs": student_top10_probs.detach().cpu().tolist(),
                **mass_curves,
                "teacher_entropy": float(teacher_entropy[b, t].detach().item()),
                "student_entropy": float(student_entropy[b, t].detach().item()),
                "loss_before_topk_mask": full_pos_loss,
                "loss_after_topk_mask": float(token_loss[b, t].detach().item()),
            }
            DEBUG_DUMP_WRITER.append(debug_record)

    metrics = {
        "logit_loss": loss.detach().item(),
        "topk": float(k),
        "topk_mass_mean": (student_topk_mass * mask_f).sum().detach().item() / valid_count.detach().item(),
        "tail_mass_mean": (student_tail * mask_f).sum().detach().item() / valid_count.detach().item(),
        "student_entropy": (student_entropy * mask_f).sum().detach().item() / valid_count.detach().item(),
        "teacher_entropy": (teacher_entropy * mask_f).sum().detach().item() / valid_count.detach().item(),
        "effective_topk_for_95_mass": (first_95.float() * mask_f).sum().detach().item() / valid_count.detach().item(),
        "effective_topk_for_99_mass": (first_99.float() * mask_f).sum().detach().item() / valid_count.detach().item(),
        **student_mass_metrics,
    }
    return loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
        vpred_mean: a float
            The mean of predicted values

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    metrics = {
        "vf_clipfrac": VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item(),
        "vpred_mean": VF.masked_mean(vpreds, response_mask).detach().item(),
    }
    return vf_loss, metrics


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
