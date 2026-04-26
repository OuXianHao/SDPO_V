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

    # ---- Teacher token-level advantage reweighting (experimental) ----
    teacher_reweight_enabled: bool = False
    """Enable teacher-guided token-level advantage reweighting.
    Uses the privileged guideline teacher's token logprobs to modulate
    per-token advantage magnitude in the DAPO/GRPO objective, without
    adding a KL/distillation loss.  Only effective in dapo_with_sdpo mode."""
    teacher_reweight_lambda: float = 0.5
    """Interpolation between uniform and reweighted advantage.
    0.0 = fully uniform (no reweighting), 1.0 = fully reweighted."""
    teacher_reweight_eps_w_low: float = 0.2
    """Lower clip for the token reweighting factor w_t.  w_t is clipped to
    [1 - eps_w_low, 1 + eps_w_high], allowing asymmetric clipping."""
    teacher_reweight_eps_w_high: float = 0.2
    """Upper clip for the token reweighting factor w_t."""
    teacher_reweight_delta_clamp: float = 5.0
    """Clamp magnitude for teacher-student logprob difference before exponentiation."""
    teacher_reweight_correct_hint: bool = False
    """Append the known correct final option to the teacher guideline for MC tasks.
    Only takes effect when sdpo_feedback_mode is 'guideline_mixed_rollouts'."""
    teacher_reweight_lambda_decay_to_zero_step: int = 0
    """If > 0, linearly decay teacher_reweight_lambda from its initial value to 0
    over this many training steps.  At step >= this value, lambda is 0 (no
    reweighting).  If <= 0, lambda is constant (no decay)."""
    teacher_reweight_skip_long_response: bool = False
    """Disable RLSD teacher_reweight for long responses at sample level.
    When enabled together with a positive min length threshold, matching
    samples keep uniform DAPO advantages (RLSD multiplier forced to 1)."""
    teacher_reweight_skip_min_response_len: int = 0
    """Minimum response length to skip RLSD teacher_reweight for a sample.
    Gate is disabled when <= 0."""
    # ---- Teacher reweight token-level dump (debug/analysis) ----
    teacher_reweight_dump_enabled: bool = False
    """Dump per-sample token-level teacher-reweight analysis to JSONL for
    external visualization.  Only active in teacher_reweight_enabled mode."""
    teacher_reweight_dump_path: str = ""
    """File path for the JSONL dump.  Required when dump_enabled is True."""
    teacher_reweight_dump_max_samples: int = 100
    """Maximum total number of sample records to write across the entire run."""
    teacher_reweight_dump_max_correct: int = 10
    """Maximum number of *correct* rollout records to dump (balanced sampling)."""
    teacher_reweight_dump_max_incorrect: int = 10
    """Maximum number of *incorrect* rollout records to dump (balanced sampling)."""

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

    # ---- Visual counterfactual branch for RLSD token reweighting ----
    visual_cf_enabled: bool = False
    """Enable visual counterfactual emphasis in RLSD token reweight path (dapo_with_sdpo + teacher_reweight only)."""
    visual_cf_base_gate_mode: str = "existing_rlsd_only"
    """Base gate mode: 'existing_rlsd_only', 'entropy_or_tsdelta', 'existing_rlsd_and_informative'."""
    visual_cf_base_gate_entropy_threshold: float = 0.0
    """Entropy threshold for informative-token base gate modes."""
    visual_cf_base_gate_tsdelta_threshold: float = 0.0
    """Teacher-student delta threshold for informative-token base gate modes."""
    visual_cf_visual_gate_threshold: float = 0.0
    """Hard visual gate threshold on visual_delta_t."""
    visual_cf_visual_weight: float = 0.0
    """Visual emphasis weight used in visual_factor_t = 1 + weight * gate * strength."""
    visual_cf_visual_delta_clip: float = 5.0
    """Clamp max for visual_strength_t."""
    visual_cf_visual_factor_clip_low: float = 1.0
    """Lower clip bound for visual_factor_t (v1 keeps it >= 1.0 by default)."""
    visual_cf_visual_factor_clip_high: float = 2.0
    """Upper clip bound for visual_factor_t."""
    visual_cf_final_weight_clip_enabled: bool = False
    """Enable final clip on composed token weight in visual_cf path."""
    visual_cf_final_weight_clip_low: float = 0.0
    """Lower clip bound for final token weight when final clip is enabled."""
    visual_cf_final_weight_clip_high: float = 10.0
    """Upper clip bound for final token weight when final clip is enabled."""
    visual_cf_bad_video_mode: str = "reuse_existing"
    """Bad-video mode for visual_cf: 'reuse_existing' or explicit mode supported by bad_video utility."""
    visual_cf_debug: bool = False
    """Enable extra debug metrics/logging for visual_cf path."""


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
        # Teacher token-level reweighting propagation
        self.worker.actor.teacher_reweight_enabled = self.algorithm.teacher_reweight_enabled
        self.worker.actor.teacher_reweight_lambda = self.algorithm.teacher_reweight_lambda
        self.worker.actor.teacher_reweight_eps_w_low = self.algorithm.teacher_reweight_eps_w_low
        self.worker.actor.teacher_reweight_eps_w_high = self.algorithm.teacher_reweight_eps_w_high
        self.worker.actor.teacher_reweight_delta_clamp = self.algorithm.teacher_reweight_delta_clamp
        self.worker.actor.teacher_reweight_correct_hint = self.algorithm.teacher_reweight_correct_hint
        self.worker.actor.teacher_reweight_lambda_decay_to_zero_step = self.algorithm.teacher_reweight_lambda_decay_to_zero_step
        self.worker.actor.teacher_reweight_skip_long_response = self.algorithm.teacher_reweight_skip_long_response
        self.worker.actor.teacher_reweight_skip_min_response_len = self.algorithm.teacher_reweight_skip_min_response_len
        # Teacher reweight dump propagation
        self.worker.actor.teacher_reweight_dump_enabled = self.algorithm.teacher_reweight_dump_enabled
        self.worker.actor.teacher_reweight_dump_path = self.algorithm.teacher_reweight_dump_path
        self.worker.actor.teacher_reweight_dump_max_samples = self.algorithm.teacher_reweight_dump_max_samples
        self.worker.actor.teacher_reweight_dump_max_correct = self.algorithm.teacher_reweight_dump_max_correct
        self.worker.actor.teacher_reweight_dump_max_incorrect = self.algorithm.teacher_reweight_dump_max_incorrect
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
        # visual_cf propagation
        self.worker.actor.visual_cf_enabled = self.algorithm.visual_cf_enabled
        self.worker.actor.visual_cf_base_gate_mode = self.algorithm.visual_cf_base_gate_mode
        self.worker.actor.visual_cf_base_gate_entropy_threshold = self.algorithm.visual_cf_base_gate_entropy_threshold
        self.worker.actor.visual_cf_base_gate_tsdelta_threshold = self.algorithm.visual_cf_base_gate_tsdelta_threshold
        self.worker.actor.visual_cf_visual_gate_threshold = self.algorithm.visual_cf_visual_gate_threshold
        self.worker.actor.visual_cf_visual_weight = self.algorithm.visual_cf_visual_weight
        self.worker.actor.visual_cf_visual_delta_clip = self.algorithm.visual_cf_visual_delta_clip
        self.worker.actor.visual_cf_visual_factor_clip_low = self.algorithm.visual_cf_visual_factor_clip_low
        self.worker.actor.visual_cf_visual_factor_clip_high = self.algorithm.visual_cf_visual_factor_clip_high
        self.worker.actor.visual_cf_final_weight_clip_enabled = self.algorithm.visual_cf_final_weight_clip_enabled
        self.worker.actor.visual_cf_final_weight_clip_low = self.algorithm.visual_cf_final_weight_clip_low
        self.worker.actor.visual_cf_final_weight_clip_high = self.algorithm.visual_cf_final_weight_clip_high
        self.worker.actor.visual_cf_bad_video_mode = self.algorithm.visual_cf_bad_video_mode
        self.worker.actor.visual_cf_debug = self.algorithm.visual_cf_debug
        # Combined DAPO+SDPO lambda propagation
        self.worker.actor.lambda_dapo = self.algorithm.lambda_dapo
        self.worker.actor.lambda_sdpo_t = self.algorithm.lambda_sdpo_t
        self.worker.actor.lambda_sdpo_v = self.algorithm.lambda_sdpo_v

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
