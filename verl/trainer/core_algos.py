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
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..utils import torch_functional as VF

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
        "entropy_loss": metrics["entropy_loss"],
    }
    return loss, metrics


def compute_sdpo_logit_loss(
    response_mask: torch.Tensor,
    topk: int = 100,
    divergence: Literal["forward_kl", "reverse_kl"] = "forward_kl",
    use_tail: bool = True,
    approx_mode: Literal["topk", "student_topk_tail", "full_vocab"] = "topk",
    alpha: Optional[float] = None,
    # --- top-k mode inputs (pre-computed in actor) ---
    student_topk_logps: Optional[torch.Tensor] = None,
    teacher_topk_logps: Optional[torch.Tensor] = None,
    # --- full-vocab mode inputs ---
    student_logits: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    # --- optional: detached full logits for metrics only ---
    student_logits_for_metrics: Optional[torch.Tensor] = None,
    teacher_logits_for_metrics: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standalone SDPO logit-level loss.

    Supports two computation modes:

    **Top-k mode** (``approx_mode`` in ``{"topk", "student_topk_tail"}``):
        Receives pre-computed ``student_topk_logps`` and ``teacher_topk_logps``
        — proper log-probabilities of shape ``(batch, resp_len, k)`` that were
        normalised by the full-vocabulary logsumexp *inside the actor forward*.
        The loss never sees the full ``(batch, resp_len, vocab_size)`` tensor,
        keeping the backward graph compact and avoiding the
        ``SplitWithSizesBackward0`` gradient errors that plagued the old path.

    **Full-vocab mode** (``approx_mode == "full_vocab"``):
        Receives raw ``student_logits`` / ``teacher_logits`` of shape
        ``(batch, resp_len, vocab_size)`` and computes exact KL.  Use only for
        debugging — the gradient flows through the full vocabulary dimension.
    """
    if divergence not in ("forward_kl", "reverse_kl"):
        raise ValueError(f"Unsupported divergence: {divergence}")
    if approx_mode not in ("topk", "student_topk_tail", "full_vocab"):
        raise ValueError(f"Unsupported approx_mode: {approx_mode}")

    # Resolve effective alpha: if alpha is explicitly provided, use it;
    # otherwise derive from the legacy divergence string for backward compat.
    # alpha=0.0 → forward KL, alpha=1.0 → reverse KL, 0<alpha<1 → GJS
    # (matches original SDPO: SelfDistillationConfig.alpha semantics)
    if alpha is None:
        effective_alpha = 0.0 if divergence == "forward_kl" else 1.0
    else:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        effective_alpha = alpha

    mask_f = response_mask.float()
    valid_count = mask_f.sum().clamp_min(1.0)
    topk_mode = approx_mode in ("topk", "student_topk_tail")

    debug_sdpo = os.getenv("EASYR1_DEBUG_SDPO_UPDATE", "0").strip().lower() in {"1", "true", "yes", "on"}
    _debug_shapes = os.getenv("SDPO_T_DEBUG_SHAPES", "0").strip().lower() in {"1", "true", "yes", "on"}
    _rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # --- Utility: make backward hooks with step labels ---
    def _make_bwd_hook(step_id, tensor_name):
        """Factory for backward hooks with ordered step IDs."""
        def _hook(grad):
            print(
                f"[SDPO_T_BACKWARD] step={step_id} {tensor_name} grad "
                f"shape={tuple(grad.shape)} dtype={grad.dtype} numel={grad.numel()} "
                f"is_contiguous={grad.is_contiguous()} stride={grad.stride()} "
                f"has_nan={bool(grad.isnan().any())} has_inf={bool(grad.isinf().any())}"
            )
            return grad
        return _hook

    if _debug_shapes and _rank == 0:
        print(
            f"[SDPO_T_BRANCH] step=L00 compute_sdpo_logit_loss ENTRY "
            f"approx_mode={approx_mode} topk_mode={topk_mode} "
            f"divergence={divergence} use_tail={use_tail} topk={topk} "
            f"response_mask shape={tuple(response_mask.shape)} "
            f"valid_count={valid_count.item():.0f}"
        )

    if topk_mode:
        # ---- Top-k path: receive pre-computed log-probs ----
        if student_topk_logps is None or teacher_topk_logps is None:
            raise ValueError(
                "Top-k SDPO mode requires pre-computed student_topk_logps "
                "and teacher_topk_logps from the actor forward."
            )
        if student_topk_logps.shape != teacher_topk_logps.shape:
            raise ValueError(
                f"student_topk_logps shape {student_topk_logps.shape} != "
                f"teacher_topk_logps shape {teacher_topk_logps.shape}"
            )
        if student_topk_logps.shape[:2] != response_mask.shape:
            raise ValueError("response_mask shape must match first two dims of topk_logps.")

        k = student_topk_logps.size(-1)

        if _debug_shapes and _rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=L01 topk_path inputs k={k} "
                f"student_topk_logps shape={tuple(student_topk_logps.shape)} "
                f"stride={student_topk_logps.stride()} "
                f"is_contiguous={student_topk_logps.is_contiguous()} "
                f"requires_grad={student_topk_logps.requires_grad} "
                f"grad_fn={student_topk_logps.grad_fn} "
                f"teacher_topk_logps shape={tuple(teacher_topk_logps.shape)} "
                f"requires_grad={teacher_topk_logps.requires_grad}"
            )
            # Walk grad_fn chain from student_topk_logps
            gf = student_topk_logps.grad_fn
            chain = []
            for _i in range(20):
                if gf is None:
                    break
                nf = gf.next_functions
                nf_names = [(type(f).__name__ if f else "None") for f, _ in nf] if nf else []
                chain.append(f"{type(gf).__name__}(next={nf_names})")
                gf = nf[0][0] if nf else None
            print(f"[SDPO_T_SHAPE] step=L01b student_topk_logps grad_fn_chain={chain}")

        # ---- Alpha-parameterized divergence (matches original SDPO core_algos.py:1138-1160) ----
        # Build proper distributions for divergence computation.
        # Original SDPO uses add_tail (appending tail log-prob bucket) to form
        # a proper (k+1)-category distribution, then applies the alpha formula.
        def _add_tail(logps: torch.Tensor) -> torch.Tensor:
            """Append tail log-probability: log(1 - sum(exp(logps))).

            Matches original SDPO's add_tail() using expm1 for numerical stability.
            """
            log_s = torch.logsumexp(logps, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_logp = torch.log(-torch.expm1(log_s))
            return torch.cat([logps, tail_logp], dim=-1)

        def _renorm(logps: torch.Tensor) -> torch.Tensor:
            """Renormalize top-k log-probs to proper distribution over k categories."""
            return logps - torch.logsumexp(logps, dim=-1, keepdim=True)

        if use_tail:
            student_dist_logps = _add_tail(student_topk_logps)
            teacher_dist_logps = _add_tail(teacher_topk_logps)
        else:
            student_dist_logps = _renorm(student_topk_logps)
            teacher_dist_logps = _renorm(teacher_topk_logps)

        if _debug_shapes and _rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=L02 dist_logps "
                f"student shape={tuple(student_dist_logps.shape)} "
                f"requires_grad={student_dist_logps.requires_grad} "
                f"teacher shape={tuple(teacher_dist_logps.shape)} "
                f"effective_alpha={effective_alpha}"
            )

        if effective_alpha == 0.0:
            # Forward KL: KL(teacher || student) — matches original SDPO alpha=0
            kl_loss = F.kl_div(student_dist_logps, teacher_dist_logps, reduction="none", log_target=True)
            token_loss = kl_loss.sum(dim=-1)
        elif effective_alpha == 1.0:
            # Reverse KL: KL(student || teacher) — matches original SDPO alpha=1
            kl_loss = F.kl_div(teacher_dist_logps, student_dist_logps, reduction="none", log_target=True)
            token_loss = kl_loss.sum(dim=-1)
        else:
            # Generalized Jensen-Shannon Divergence — matches original SDPO alpha in (0,1)
            alpha_t = torch.tensor(effective_alpha, dtype=student_dist_logps.dtype, device=student_dist_logps.device)
            mixture_logps = torch.logsumexp(
                torch.stack([student_dist_logps + torch.log(1 - alpha_t), teacher_dist_logps + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_logps, teacher_dist_logps, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_logps, student_dist_logps, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
            token_loss = kl_loss.sum(dim=-1)

        if _debug_shapes and _rank == 0:
            print(
                f"[SDPO_T_SHAPE] step=L04 token_loss shape={tuple(token_loss.shape)} "
                f"requires_grad={token_loss.requires_grad} grad_fn={token_loss.grad_fn} "
                f"dtype={token_loss.dtype}"
            )
            if token_loss.requires_grad:
                token_loss.register_hook(_make_bwd_hook("B04", "token_loss"))

        # Compute mass/tail metrics (for logging only, not used in loss)
        student_topk_prob = student_topk_logps.detach().exp()
        student_topk_mass = student_topk_prob.sum(dim=-1)
        student_tail = (1.0 - student_topk_mass).clamp(1e-12, 1.0)

        vocab_size = k  # for metrics; real vocab_size comes from *_for_metrics if provided

    else:
        # ---- Full-vocab path ----
        if student_logits is None or teacher_logits is None:
            raise ValueError("full_vocab SDPO mode requires student_logits and teacher_logits.")
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("student_logits and teacher_logits must have the same shape.")
        if student_logits.shape[:2] != response_mask.shape:
            raise ValueError("response_mask shape must match first two dims of logits.")

        vocab_size = student_logits.size(-1)
        k = vocab_size

        if debug_sdpo:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            print(
                "[RCA_SDPO_FULL_VOCAB_BRANCH] "
                f"rank={rank}/{world} student_logits_shape={tuple(student_logits.shape)} "
                f"teacher_logits_shape={tuple(teacher_logits.shape)} response_mask_shape={tuple(response_mask.shape)}"
            )

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if effective_alpha == 0.0:
            token_kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif effective_alpha == 1.0:
            token_kl = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Generalized Jensen-Shannon Divergence (full-vocab)
            alpha_t = torch.tensor(effective_alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_logps = torch.logsumexp(
                torch.stack([student_log_probs + torch.log(1 - alpha_t), teacher_log_probs + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_logps, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_logps, student_log_probs, reduction="none", log_target=True)
            token_kl = torch.lerp(kl_student, kl_teacher, alpha_t)
        token_loss = token_kl.sum(dim=-1)
        student_topk_mass = torch.ones_like(mask_f)
        student_tail = torch.zeros_like(mask_f)

    # ---- Loss ----
    masked_token_loss = token_loss * mask_f
    if _debug_shapes and _rank == 0:
        print(
            f"[SDPO_T_SHAPE] step=L10 PRE_LOSS_REDUCE "
            f"token_loss shape={tuple(token_loss.shape)} "
            f"requires_grad={token_loss.requires_grad} "
            f"masked_token_loss shape={tuple(masked_token_loss.shape)} "
            f"requires_grad={masked_token_loss.requires_grad} "
            f"grad_fn={masked_token_loss.grad_fn} "
            f"mask_f shape={tuple(mask_f.shape)} "
            f"valid_count={valid_count.item():.0f}"
        )
        if masked_token_loss.requires_grad:
            masked_token_loss.register_hook(_make_bwd_hook("B10", "masked_token_loss"))
    loss = masked_token_loss.sum() / valid_count
    if _debug_shapes and _rank == 0:
        print(
            f"[SDPO_T_SHAPE] step=L11 FINAL_LOSS shape={tuple(loss.shape)} "
            f"requires_grad={loss.requires_grad} grad_fn={loss.grad_fn} "
            f"value={loss.item():.6f}"
        )
        # Register hook on loss to trace backward entry
        if loss.requires_grad:
            loss.register_hook(_make_bwd_hook("B11", "sdpo_loss"))

    # ---- Metrics (all under no_grad) ----
    with torch.no_grad():
        if topk_mode and student_logits_for_metrics is not None:
            # Use detached full logits passed from the actor for rich metrics
            s_logits_m = student_logits_for_metrics
            t_logits_m = teacher_logits_for_metrics if teacher_logits_for_metrics is not None else None
            metrics_vocab = s_logits_m.size(-1)

            s_log_probs_m = F.log_softmax(s_logits_m, dim=-1)
            s_probs_m = s_log_probs_m.exp()
            student_entropy = -(s_probs_m * s_log_probs_m).sum(dim=-1)
            if t_logits_m is not None:
                t_log_probs_m = F.log_softmax(t_logits_m, dim=-1)
                t_probs_m = t_log_probs_m.exp()
                teacher_entropy = -(t_probs_m * t_log_probs_m).sum(dim=-1)
            else:
                teacher_entropy = torch.zeros_like(student_entropy)
            student_probs_for_sort = s_probs_m
        elif not topk_mode:
            # full-vocab mode: probs already computed
            s_log_probs_m = F.log_softmax(student_logits.detach(), dim=-1)
            s_probs_m = s_log_probs_m.exp()
            t_log_probs_m = F.log_softmax(teacher_logits.detach(), dim=-1)
            t_probs_m = t_log_probs_m.exp()
            student_entropy = -(s_probs_m * s_log_probs_m).sum(dim=-1)
            teacher_entropy = -(t_probs_m * t_log_probs_m).sum(dim=-1)
            student_probs_for_sort = s_probs_m
            metrics_vocab = vocab_size
        else:
            # top-k mode but no full logits for metrics — use top-k probs only
            student_entropy_topk = -(student_topk_prob.detach() * student_topk_logps.detach()).sum(dim=-1)
            teacher_entropy_topk = -(teacher_topk_prob.detach() * teacher_topk_logps.detach()).sum(dim=-1)
            student_entropy = student_entropy_topk
            teacher_entropy = teacher_entropy_topk
            student_probs_for_sort = None
            metrics_vocab = k

        # Mass-at-k and effective-topk metrics (require sorted full-vocab probs)
        student_mass_metrics: dict[str, float] = {}
        first_95 = torch.zeros_like(mask_f)
        first_99 = torch.zeros_like(mask_f)
        mass_ks = [1, 5, 10, 20, 50, 100]
        if student_probs_for_sort is not None:
            student_sorted_probs = torch.sort(student_probs_for_sort, dim=-1, descending=True).values
            for mass_k in mass_ks:
                cur_k = min(mass_k, metrics_vocab)
                mass_at_k = student_sorted_probs[..., :cur_k].sum(dim=-1)
                student_mass_metrics[f"mass@{mass_k}"] = (
                    (mass_at_k * mask_f).sum().item() / valid_count.item()
                )
            student_cdf = torch.cumsum(student_sorted_probs, dim=-1)
            first_95 = (student_cdf >= 0.95).float().argmax(dim=-1) + 1
            first_99 = (student_cdf >= 0.99).float().argmax(dim=-1) + 1
        else:
            # Approximate from top-k probs
            topk_mass = student_topk_mass.detach()
            for mass_k in mass_ks:
                if mass_k <= k:
                    student_topk_sorted = torch.sort(student_topk_prob.detach(), dim=-1, descending=True).values
                    cur_k = min(mass_k, k)
                    mass_at_k = student_topk_sorted[..., :cur_k].sum(dim=-1)
                    student_mass_metrics[f"mass@{mass_k}"] = (
                        (mass_at_k * mask_f).sum().item() / valid_count.item()
                    )
                else:
                    student_mass_metrics[f"mass@{mass_k}"] = (
                        (topk_mass * mask_f).sum().item() / valid_count.item()
                    )

    metrics = {
        "logit_loss": loss.detach().item(),
        "alpha": effective_alpha,
        "topk": float(k if topk_mode else metrics_vocab),
        "approx_mode": 0.0 if topk_mode else 1.0,
        "topk_mass_mean": (student_topk_mass.detach() * mask_f).sum().item() / valid_count.item(),
        "tail_mass_mean": (student_tail.detach() * mask_f).sum().item() / valid_count.item(),
        "student_entropy": (student_entropy * mask_f).sum().item() / valid_count.item(),
        "teacher_entropy": (teacher_entropy * mask_f).sum().item() / valid_count.item(),
        "effective_topk_for_95_mass": (first_95.float() * mask_f).sum().item() / valid_count.item(),
        "effective_topk_for_99_mass": (first_99.float() * mask_f).sum().item() / valid_count.item(),
        **student_mass_metrics,
    }
    return loss, metrics


def compute_sdpo_v_loss(
    good_topk_logps: torch.Tensor,
    bad_topk_logps: torch.Tensor,
    response_mask: torch.Tensor,
    margin: float = 0.1,
    use_tail: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SDPO-V visual separation loss.

    Computes a token-level margin hinge loss that encourages the good-video
    branch to assign higher average log-probability (on good-selected top-k
    tokens) than the bad-video branch.

    Loss per token:
        L_t = max(0, margin - (s_good_t - s_bad_t))

    where s_t = mean of top-k log-probs at position t.

    Args:
        good_topk_logps: (batch, resp_len, k) — proper log-probs from good branch.
        bad_topk_logps:  (batch, resp_len, k) — proper log-probs from bad branch,
                         gathered at the SAME good-selected top-k indices.
        response_mask:   (batch, resp_len) — boolean mask for valid response tokens.
        margin: Global constant margin.
        use_tail: Whether the log-probs include a tail bucket (last dim).

    Returns:
        loss: Scalar loss tensor.
        metrics: Dict of float metrics for logging.
    """
    mask_f = response_mask.float()
    valid_count = mask_f.sum().clamp_min(1.0)

    # Compute per-token scores: mean log-prob over the top-k dimension
    # If use_tail, we use all k+1 categories; otherwise just the k top tokens.
    # Score = mean(log_prob) over the token subset at each position.
    s_good = good_topk_logps.mean(dim=-1)  # (batch, resp_len)
    s_bad = bad_topk_logps.mean(dim=-1)    # (batch, resp_len)

    # Token-level delta and hinge loss
    delta = s_good - s_bad  # (batch, resp_len), positive means good > bad
    hinge = torch.clamp(margin - delta, min=0.0)  # (batch, resp_len)

    # Masked reduction
    masked_hinge = hinge * mask_f
    loss = masked_hinge.sum() / valid_count

    # Metrics (all detached)
    with torch.no_grad():
        delta_masked = delta * mask_f
        delta_mean = delta_masked.sum() / valid_count
        delta_sq = (delta ** 2) * mask_f
        delta_var = delta_sq.sum() / valid_count - delta_mean ** 2
        delta_std = delta_var.clamp_min(0.0).sqrt()

        # Fraction of tokens where hinge is active (margin not met)
        active_frac = ((hinge > 0).float() * mask_f).sum() / valid_count

        # Fraction of tokens where good > bad (delta > 0)
        positive_frac = ((delta > 0).float() * mask_f).sum() / valid_count

    metrics = {
        "v_loss": loss.detach().item(),
        "v_margin": margin,
        "v_delta_mean": delta_mean.item(),
        "v_delta_std": delta_std.item(),
        "v_active_frac": active_frac.item(),
        "v_positive_frac": positive_frac.item(),
        "v_s_good_mean": (s_good.detach() * mask_f).sum().item() / valid_count.item(),
        "v_s_bad_mean": (s_bad.detach() * mask_f).sum().item() / valid_count.item(),
    }
    return loss, metrics


def compute_sdpo_v_calibration_stats(
    good_topk_logps: torch.Tensor,
    bad_topk_logps: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Collect delta_t = s_good - s_bad statistics for margin calibration.

    Returns percentile-based summary statistics to help choose a good margin.

    Args:
        good_topk_logps: (batch, resp_len, k)
        bad_topk_logps:  (batch, resp_len, k) — gathered at same indices
        response_mask:   (batch, resp_len) — boolean mask

    Returns:
        Dict with count, mean, std, p10, p25, p50, p75, p90 of delta_t,
        plus a recommended_margin (p25 of delta).
    """
    with torch.no_grad():
        s_good = good_topk_logps.mean(dim=-1)  # (batch, resp_len)
        s_bad = bad_topk_logps.mean(dim=-1)
        delta = s_good - s_bad  # (batch, resp_len)

        # Flatten and select valid tokens
        valid_mask = response_mask.bool().view(-1)
        delta_flat = delta.view(-1)
        valid_deltas = delta_flat[valid_mask]

        if valid_deltas.numel() == 0:
            return {
                "calib_count": 0, "calib_mean": 0.0, "calib_std": 0.0,
                "calib_p10": 0.0, "calib_p25": 0.0, "calib_p50": 0.0,
                "calib_p75": 0.0, "calib_p90": 0.0,
                "calib_recommended_margin": 0.0,
            }

        count = valid_deltas.numel()
        mean = valid_deltas.mean().item()
        std = valid_deltas.std().item() if count > 1 else 0.0

        percentiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=valid_deltas.device)
        quantiles = torch.quantile(valid_deltas.float(), percentiles)
        p10, p25, p50, p75, p90 = [q.item() for q in quantiles]

        return {
            "calib_count": count,
            "calib_mean": mean,
            "calib_std": std,
            "calib_p10": p10,
            "calib_p25": p25,
            "calib_p50": p50,
            "calib_p75": p75,
            "calib_p90": p90,
            "calib_recommended_margin": p25,
        }


def compute_sdpo_v_softkl_loss(
    good_topk_logps: torch.Tensor,
    bad_ref_topk_logps: torch.Tensor,
    response_mask: torch.Tensor,
    tau: float = 1.0,
    use_tail: bool = False,
    kl_max: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """SDPO-V soft-capped forward KL loss.

    Computes a token-level forward KL divergence between the good-video branch
    (reference/anchor) and the bad-video EMA reference branch, then applies a
    soft cap to produce a bounded, smooth reward-like signal.

    For each valid token position t:
        KL_t = D_KL(p_good || p_bad_ref) = sum_k p_good_k * (log p_good_k - log p_bad_ref_k)
        KL_t = clamp(KL_t, 0, kl_max)   [if kl_max > 0]
        phi(KL_t) = tau * (1 - exp(-KL_t / tau))
        L = -mean_t phi(KL_t)

    Args:
        good_topk_logps:     (batch, resp_len, k) — log-probs from good branch
                             (proper log-probs normalized by full vocab logsumexp).
        bad_ref_topk_logps:  (batch, resp_len, k) — log-probs from bad-video EMA
                             reference branch, gathered at the SAME good-selected
                             top-k indices.
        response_mask:       (batch, resp_len) — boolean mask for valid response tokens.
        tau:  Soft-cap temperature. phi(x) = tau * (1 - exp(-x / tau)).
        use_tail: Whether the log-probs include a tail bucket (last dim).
        kl_max: Hard ceiling on per-token KL before the soft cap.
                When > 0, KL is clamped to [0, kl_max], which also zeros the
                gradient for tokens whose KL already exceeds the target.
                This prevents the mode-seeking KL-maximization objective
                from pushing logits to extremes.  Default 0 = no ceiling.

    Returns:
        loss: Scalar loss tensor (negative of mean soft-capped KL).
        metrics: Dict of float metrics for logging.
    """
    mask_f = response_mask.float()
    valid_count = mask_f.sum().clamp_min(1.0)

    # ---- Renormalize to local distributions on the top-k (or k+1) subset ----
    # Input logps are proper log-probs (logit - logsumexp over full vocab).
    # We need local distributions over just the k categories.
    good_local_logps = good_topk_logps - torch.logsumexp(good_topk_logps, dim=-1, keepdim=True)
    bad_local_logps = bad_ref_topk_logps - torch.logsumexp(bad_ref_topk_logps, dim=-1, keepdim=True)

    # ---- Token-level forward KL: D_KL(p_good || p_bad_ref) ----
    # = sum_k p_good_k * (log p_good_k - log p_bad_ref_k)
    good_probs = good_local_logps.exp()  # (batch, resp_len, k)
    kl_per_category = good_probs * (good_local_logps - bad_local_logps)  # (batch, resp_len, k)
    # Clamp individual terms to avoid NaN from 0 * (-inf)
    kl_per_category = kl_per_category.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    kl_t = kl_per_category.sum(dim=-1)  # (batch, resp_len)
    # Forward KL should be >= 0; clamp for numerical safety
    kl_t = kl_t.clamp_min(0.0)

    # ---- KL ceiling: prevent unbounded KL maximization ----
    # When kl_max > 0, tokens whose KL already exceeds the target receive
    # zero gradient (clamp kills the backward signal).  This is the key
    # safeguard against the mode-seeking collapse observed when the negative
    # loss drives logits to extremes within the first few steps.
    if kl_max > 0:
        kl_t = kl_t.clamp(max=kl_max)

    # ---- Soft cap: phi(x) = tau * (1 - exp(-x / tau)) ----
    tau_safe = max(tau, 1e-8)
    phi_t = tau_safe * (1.0 - torch.exp(-kl_t / tau_safe))  # (batch, resp_len)

    # ---- Masked reduction: L = -mean(phi(KL_t)) ----
    masked_phi = phi_t * mask_f
    loss = -(masked_phi.sum() / valid_count)

    # ---- Metrics (all detached) ----
    with torch.no_grad():
        kl_masked = kl_t * mask_f
        kl_mean = kl_masked.sum() / valid_count
        phi_mean = masked_phi.sum() / valid_count
        # Track how many tokens hit the ceiling (useful for tuning kl_max)
        kl_ceil_frac = 0.0
        if kl_max > 0:
            kl_ceil_frac = float(((kl_t >= kl_max - 1e-6) & response_mask.bool()).sum().item()) / max(valid_count.item(), 1.0)

    metrics = {
        "vsk_loss": loss.detach().item(),
        "vsk_tau": tau,
        "vsk_kl_raw_mean": kl_mean.item(),
        "vsk_kl_capped_mean": phi_mean.item(),
        "vsk_valid_tokens": int(valid_count.item()),
        "vsk_kl_max": kl_max,
        "vsk_kl_ceil_frac": kl_ceil_frac,
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


def compute_rlsd_token_advantages(
    advantages: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_eps_w_low: float = 0.2,
    rlsd_eps_w_high: float = 0.2,
    delta_clamp: float = 5.0,
    answer_span_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level advantage reweighting using teacher-student logprob difference.

    Uses the privileged teacher's per-token support to modulate the magnitude
    of the sequence-level advantage at each token position, without changing
    the update direction (which is still determined by the sequence-level
    DAPO/GRPO advantage).

    Formula:
        delta_t = clamp(logp_teacher(y_t) - logp_student(y_t), -C, C)
        w_t = exp(sign(A) * delta_t)
        A_hat_t = A * ((1 - lambda) + lambda * clip(w_t, 1 - eps_low, 1 + eps_high))

    All inputs to delta_t must be detached — no gradient flows through this
    reweighting path.  The GRPO/DAPO policy loss provides the only gradient
    through the student log-probs.

    Args:
        advantages: (bs, resp_len) — uniform sequence-level advantage broadcast to tokens
        teacher_logprobs: (bs, resp_len) — detached per-token log P_teacher(y_t)
        student_logprobs: (bs, resp_len) — detached per-token log P_student(y_t)
        response_mask: (bs, resp_len) — valid response token mask
        rlsd_lambda: interpolation weight (0 = uniform, 1 = fully reweighted)
        rlsd_eps_w_low: lower clip for w_t, clipped to [1 - eps_low, ...]
        rlsd_eps_w_high: upper clip for w_t, clipped to [..., 1 + eps_high]
        delta_clamp: clamp magnitude for delta_t before exponentiation
        answer_span_mask: (bs, resp_len) optional bool mask, True for answer-span tokens
            that should be excluded from reweighting (they keep uniform advantage)

    Returns:
        A_hat_t: (bs, resp_len) — token-level reweighted advantages
        metrics: diagnostic dict
    """
    # delta_t: positive when teacher is more confident than student on this token
    delta_t = torch.clamp(teacher_logprobs - student_logprobs, -delta_clamp, delta_clamp)

    # w_t > 1 when teacher support aligns with advantage sign:
    #   A > 0 and teacher more confident → upweight (encourage this token)
    #   A < 0 and teacher less confident → upweight (penalise this token more)
    w_t = torch.exp(torch.sign(advantages) * delta_t)
    w_t = torch.clamp(w_t, 1.0 - rlsd_eps_w_low, 1.0 + rlsd_eps_w_high)

    # Gate: only reweight negative-advantage tokens; non-negative → w_t = 1
    # (degenerates to plain DAPO for positively-rewarded rollouts)
    w_t = torch.where(advantages < 0, w_t, torch.ones_like(w_t))

    token_weight = (1.0 - rlsd_lambda) + rlsd_lambda * w_t

    # Exclude answer-span tokens from reweighting: keep uniform advantage
    # to prevent over-concentrating credit on final-answer tokens when the
    # teacher guideline includes a correct-option hint.
    if answer_span_mask is not None:
        token_weight = torch.where(answer_span_mask, torch.ones_like(token_weight), token_weight)

    A_hat_t = advantages * token_weight * response_mask

    # ---- diagnostics ----
    mask_f = response_mask.float()
    metrics = {
        "delta_t_mean": VF.masked_mean(delta_t, mask_f).detach().item(),
        "delta_t_abs_mean": VF.masked_mean(delta_t.abs(), mask_f).detach().item(),
        "w_t_mean": VF.masked_mean(w_t, mask_f).detach().item(),
        "token_weight_mean": VF.masked_mean(token_weight, mask_f).detach().item(),
        "token_weight_std": (token_weight * mask_f).std().detach().item(),
    }
    if answer_span_mask is not None:
        ans_count = (answer_span_mask & response_mask.bool()).sum().item()
        total_count = response_mask.sum().item()
        metrics["answer_span_frac"] = ans_count / max(total_count, 1)

    return A_hat_t, metrics
