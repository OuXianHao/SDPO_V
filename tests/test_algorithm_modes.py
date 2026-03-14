import torch

from verl.trainer.config import PPOConfig
from verl.trainer.core_algos import compute_grpo_outcome_advantage, compute_sdpo_logit_loss


def test_grpo_advantage_group_relative_shape_and_centering():
    token_level_rewards = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(token_level_rewards)
    index = torch.tensor([0, 0, 1, 1])

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert advantages.shape == token_level_rewards.shape
    assert returns.shape == token_level_rewards.shape
    assert torch.isclose(advantages[0, 0], torch.tensor(-0.7071), atol=1e-3)
    assert torch.isclose(advantages[1, 0], torch.tensor(0.7071), atol=1e-3)


def test_sdpo_logit_loss_topk_tail_finite():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 3, 16)
    teacher_logits = student_logits + 0.1 * torch.randn(2, 3, 16)
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    loss, metrics = compute_sdpo_logit_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=response_mask,
        topk=4,
        divergence="forward_kl",
        use_tail=True,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert metrics["topk"] == 4.0
    assert 0.0 <= metrics["topk_mass_mean"] <= 1.0
    assert 0.0 <= metrics["tail_mass_mean"] <= 1.0


def test_sdpo_logit_loss_reverse_kl_without_tail_finite():
    torch.manual_seed(1)
    student_logits = torch.randn(1, 2, 8)
    teacher_logits = torch.randn(1, 2, 8)
    response_mask = torch.tensor([[1, 1]], dtype=torch.bool)

    loss, metrics = compute_sdpo_logit_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        response_mask=response_mask,
        topk=3,
        divergence="reverse_kl",
        use_tail=False,
    )
    assert torch.isfinite(loss)
    assert metrics["topk"] == 3.0


def test_config_wires_algorithm_modes_into_actor_config():
    config = PPOConfig()
    config.algorithm.loss_mode = "sdpo_logit"
    config.algorithm.sdpo_topk = 32
    config.algorithm.sdpo_divergence = "reverse_kl"
    config.algorithm.sdpo_use_tail = False
    config.algorithm.sdpo_feedback_mode = "scalar_text"
    config.post_init()

    assert config.worker.actor.loss_mode == "sdpo_logit"
    assert config.worker.actor.sdpo_topk == 32
    assert config.worker.actor.sdpo_divergence == "reverse_kl"
    assert config.worker.actor.sdpo_use_tail is False
    assert config.worker.actor.sdpo_feedback_mode == "scalar_text"
