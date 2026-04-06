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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class AlgorithmConfig:
    loss_mode: str = "grpo_on_policy"
    """training objective mode, support `grpo_on_policy`, `sdpo_logit`, `dapo_with_sdpo`"""
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    sdpo_topk: int = 100
    sdpo_divergence: str = "forward_kl"
    sdpo_use_tail: bool = True
    sdpo_approx_mode: str = "topk"
    sdpo_feedback_mode: str = "successful_rollout"
    """SDPO-T feedback mode: 'successful_rollout', 'scalar_text', or 'guideline_mixed_rollouts'."""
    sdpo_alpha: Optional[float] = None
    """Alpha for divergence interpolation: 0.0=forward KL, 1.0=reverse KL, in-between=GJS. None=derive from sdpo_divergence."""
    sdpo_teacher_update_rate: float = 0.0
    """EMA update rate for teacher weights (0.0=frozen ref teacher, 0.05=original SDPO default)."""

    # ---- Combined DAPO+SDPO loss weights (dapo_with_sdpo mode) ----
    lambda_dapo: float = 1.0
    """Weight for the DAPO/GRPO main loss in combined mode."""
    lambda_sdpo_t: float = 0.1
    """Weight for the SDPO-T auxiliary loss in combined mode."""
    lambda_sdpo_v: float = 0.1
    """Weight for the SDPO-V auxiliary loss in combined mode."""

    # ---- SDPO-V (visual separation) settings ----
    sdpo_v_enabled: bool = False
    """Enable SDPO-V visual separation loss. When False, training is identical to SDPO-T only."""
    sdpo_v_weight: float = 1.0
    """Weight for the SDPO-V loss term in the joint objective."""
    sdpo_v_topk: int = 100
    """Number of top-k tokens for SDPO-V score computation."""
    sdpo_v_use_tail: bool = False
    """Whether to append tail bucket when computing SDPO-V top-k log-probs."""
    sdpo_v_margin: float = 0.1
    """Global margin for the SDPO-V hinge loss: L = max(0, margin - (s_good - s_bad))."""
    sdpo_v_margin_mode: str = "constant"
    """Margin mode for SDPO-V. Currently only 'constant' is supported."""
    sdpo_v_bad_video_mode: str = "blur"
    """Bad-video construction strategy: 'blur', 'drop', 'blur_and_drop', 'shuffle'."""
    sdpo_v_blur_sigma: float = 5.0
    """Gaussian blur sigma for bad-video blur mode."""
    sdpo_v_blur_fraction: float = 0.5
    """Fraction of frames to blur (0.0 to 1.0)."""
    sdpo_v_drop_fraction: float = 0.5
    """Fraction of frames to drop (replaced by zeros) in drop mode."""
    sdpo_v_shuffle_fraction: float = 0.2
    """Fraction of frames to temporally shuffle in 'shuffle' mode (0.0 to 1.0)."""
    sdpo_v_debug: bool = False
    """Enable extra SDPO-V debug logging."""
    sdpo_v_calibration: bool = False
    """Enable calibration mode: collect delta_t statistics instead of applying loss."""

    # ---- SDPO-V soft-capped forward KL settings ----
    sdpo_v_softkl_enabled: bool = False
    """Enable SDPO-V soft-capped forward KL loss. Independent of the hinge line."""
    sdpo_v_softkl_weight: float = 1.0
    """Weight for the SDPO-V soft-KL loss term in the joint objective."""
    sdpo_v_softkl_topk: int = 100
    """Number of top-k tokens for the soft-KL distribution."""
    sdpo_v_softkl_tau: float = 1.0
    """Soft-cap temperature: phi(x) = tau * (1 - exp(-x / tau))."""
    sdpo_v_softkl_use_tail: bool = False
    """Whether to append tail bucket for proper distribution in soft-KL."""
    sdpo_v_softkl_debug: bool = False
    """Enable extra debug logging for the soft-KL line."""
    sdpo_v_softkl_use_ema_bad_ref: bool = True
    """Use EMA teacher module (no_grad) for the bad-video reference branch."""
    sdpo_v_softkl_kl_max: float = 1.0
    """Hard ceiling on per-token KL before the soft cap.
    Tokens whose KL already exceeds this value receive zero gradient,
    preventing the mode-seeking KL-maximization objective from pushing
    logits to extremes.  Set to 0 to disable the ceiling."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    allow_tf32: bool = False
    """Enable TF32 for matmul and cuDNN operations. TF32 uses 19-bit precision
    (10-bit mantissa) and can significantly speed up training on Ampere+ GPUs
    with a small precision trade-off. Default False preserves the existing
    behavior (full fp32 precision for internal matmul accumulations)."""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.allow_tf32 = self.trainer.allow_tf32
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.loss_mode = self.algorithm.loss_mode
        self.worker.actor.sdpo_topk = self.algorithm.sdpo_topk
        self.worker.actor.sdpo_divergence = self.algorithm.sdpo_divergence
        self.worker.actor.sdpo_use_tail = self.algorithm.sdpo_use_tail
        self.worker.actor.sdpo_approx_mode = self.algorithm.sdpo_approx_mode
        self.worker.actor.sdpo_feedback_mode = self.algorithm.sdpo_feedback_mode
        self.worker.actor.sdpo_alpha = self.algorithm.sdpo_alpha
        self.worker.actor.sdpo_teacher_update_rate = self.algorithm.sdpo_teacher_update_rate
        # SDPO-V config propagation
        self.worker.actor.sdpo_v_enabled = self.algorithm.sdpo_v_enabled
        self.worker.actor.sdpo_v_weight = self.algorithm.sdpo_v_weight
        self.worker.actor.sdpo_v_topk = self.algorithm.sdpo_v_topk
        self.worker.actor.sdpo_v_use_tail = self.algorithm.sdpo_v_use_tail
        self.worker.actor.sdpo_v_margin = self.algorithm.sdpo_v_margin
        self.worker.actor.sdpo_v_margin_mode = self.algorithm.sdpo_v_margin_mode
        self.worker.actor.sdpo_v_bad_video_mode = self.algorithm.sdpo_v_bad_video_mode
        self.worker.actor.sdpo_v_blur_sigma = self.algorithm.sdpo_v_blur_sigma
        self.worker.actor.sdpo_v_blur_fraction = self.algorithm.sdpo_v_blur_fraction
        self.worker.actor.sdpo_v_drop_fraction = self.algorithm.sdpo_v_drop_fraction
        self.worker.actor.sdpo_v_shuffle_fraction = self.algorithm.sdpo_v_shuffle_fraction
        self.worker.actor.sdpo_v_debug = self.algorithm.sdpo_v_debug
        self.worker.actor.sdpo_v_calibration = self.algorithm.sdpo_v_calibration
        # SDPO-V soft-KL config propagation
        self.worker.actor.sdpo_v_softkl_enabled = self.algorithm.sdpo_v_softkl_enabled
        self.worker.actor.sdpo_v_softkl_weight = self.algorithm.sdpo_v_softkl_weight
        self.worker.actor.sdpo_v_softkl_topk = self.algorithm.sdpo_v_softkl_topk
        self.worker.actor.sdpo_v_softkl_tau = self.algorithm.sdpo_v_softkl_tau
        self.worker.actor.sdpo_v_softkl_use_tail = self.algorithm.sdpo_v_softkl_use_tail
        self.worker.actor.sdpo_v_softkl_debug = self.algorithm.sdpo_v_softkl_debug
        self.worker.actor.sdpo_v_softkl_use_ema_bad_ref = self.algorithm.sdpo_v_softkl_use_ema_bad_ref
        self.worker.actor.sdpo_v_softkl_kl_max = self.algorithm.sdpo_v_softkl_kl_max
        # Combined DAPO+SDPO lambda propagation
        self.worker.actor.lambda_dapo = self.algorithm.lambda_dapo
        self.worker.actor.lambda_sdpo_t = self.algorithm.lambda_sdpo_t
        self.worker.actor.lambda_sdpo_v = self.algorithm.lambda_sdpo_v

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
