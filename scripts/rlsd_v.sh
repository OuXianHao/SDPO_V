#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------- W&B ----------
# 建议不要在脚本里明文写 WANDB_API_KEY。
# 运行前可以手动执行：
# export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="RLSD-V"
export WANDB_MODE="online"

# ---------- Runtime ----------
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- Debug ----------
export SDPO_DEBUG_DUMP=0
export SDPO_SKIP_DEBUG=1
export SDPO_DEBUG_MAX_SAMPLES=0
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=0

# frame-path debug 训练时关闭；如果再排查数据链路可以改成 1
export EASYR1_DEBUG_FRAME_PATH_VIDEO=0
export EASYR1_DEBUG_FRAME_PATH_VIDEO_MAX=0

export EASYR1_DEBUG_SDPO_UPDATE=0
export EASYR1_DEBUG_COLLECTIVE=0
export EASYR1_DEBUG_FSDP_HANDLES=0
export EASYR1_DEBUG_SPLIT_BACKWARD=0

export SDPO_T_DEBUG_SHAPES=0
export SDPO_DETECT_ANOMALY=0
export DEBUG_ANSWER_SPAN_MASK=0

# ---------- Ray ----------
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6388}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/data/xhou/outputs/ray_${USER}_${RAY_HEAD_PORT}}"

mkdir -p /data/xhou/checkpoints
mkdir -p /data/xhou/outputs
mkdir -p "${RAY_TEMP_DIR}"

echo "[INFO] Cleaning stale Ray processes..."
pkill -u "$USER" -f ray || true
sleep 2
ray stop --force || true

echo "[INFO] Starting Ray head..."
ray start --head \
  --port="${RAY_HEAD_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}"

# ---------- Paths ----------
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

MODEL_PATH="/data/xhou/models/Qwen3-VL-8B-Instruct"

TRAIN_DATA="/data/xhou/datasets/rlsd_v_frames32_train_14311.jsonl"
VAL_DATA="/data/xhou/datasets/rlsd_v_frames32_val200.jsonl"

EXP_NAME="qwen3vl8b_newdata_frames32_rlsd_vcf_w04_lam05_decay80"
CKPT_DIR="/data/xhou/checkpoints/${EXP_NAME}"

export WANDB_NAME="${EXP_NAME}"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.dataset_mode=rlsd_v_frames32_jsonl \
  data.frame_key=frame_paths \
  data.ground_frame_key=ground_frame_indices \
  data.target_num_frames=32 \
  data.prompt_key=prompt \
  data.answer_key=answer \
  data.video_key=videos \
  data.image_dir=null \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=64 \
  data.max_prompt_length=10240 \
  data.max_response_length=1024 \
  data.max_pixels=524288 \
  data.val_batch_size=8 \
  data.filter_overlong_prompts=true \
  data.filter_overlong_prompts_workers=1 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-5 \
  worker.actor.optim.lr_scheduler_type=cosine \
  worker.actor.optim.min_lr_ratio=0.1 \
  worker.actor.optim.lr_warmup_steps=10 \
  worker.actor.global_batch_size=64 \
  worker.actor.micro_batch_size_per_device_for_update=4 \
  worker.actor.micro_batch_size_per_device_for_experience=2 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.actor.ppo_epochs=1 \
  worker.actor.clip_ratio_low=0.2 \
  worker.actor.clip_ratio_high=0.28 \
  worker.actor.clip_ratio_dual=3.0 \
  worker.actor.loss_avg_mode=token \
  worker.rollout.n=5 \
  worker.rollout.temperature=1.0 \
  worker.rollout.top_p=1.0 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.65 \
  worker.rollout.max_model_len=10240 \
  worker.rollout.max_num_batched_tokens=65536 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  worker.ref.offload.offload_params=false \
  trainer.total_epochs=1 \
  trainer.max_steps=230 \
  trainer.val_before_train=true \
  trainer.val_freq=10 \
  trainer.save_freq=10 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=RLSD-V \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.save_checkpoint_path="${CKPT_DIR}" \
  trainer.find_last_checkpoint=true \
  trainer.n_gpus_per_node=8 \
  algorithm.loss_mode=dapo_with_sdpo \
  algorithm.disable_kl=false \
  algorithm.lambda_dapo=1.0 \
  algorithm.lambda_sdpo_t=0.0 \
  algorithm.lambda_sdpo_v=0.0 \
  algorithm.sdpo_v_enabled=false \
  algorithm.sdpo_v_softkl_enabled=false \
  algorithm.teacher_reweight_enabled=true \
  algorithm.teacher_reweight_lambda=0.5 \
  algorithm.teacher_reweight_lambda_decay_to_zero_step=70 \
  algorithm.teacher_reweight_eps_w_low=0.2 \
  algorithm.teacher_reweight_eps_w_high=0.2 \
  algorithm.teacher_reweight_delta_clamp=5.0 \
  algorithm.teacher_reweight_correct_hint=true \
  algorithm.teacher_reweight_skip_long_response=true \
  algorithm.teacher_reweight_skip_min_response_len=1024 \
  algorithm.teacher_reweight_dump_enabled=false \
  algorithm.teacher_reweight_dump_max_samples=0 \
  algorithm.visual_cf_enabled=true \
  algorithm.visual_cf_use_keyframe_mask=true \
  algorithm.visual_cf_bad_video_mode=keyframe_blackout \
  algorithm.visual_cf_visual_weight=0.2 \
  algorithm.visual_cf_visual_delta_clip=5.0 \
  algorithm.visual_cf_visual_gate_threshold=0.0 \
  algorithm.visual_cf_visual_factor_clip_low=1.0 \
  algorithm.visual_cf_visual_factor_clip_high=2.0 \
  algorithm.visual_cf_base_gate_mode=existing_rlsd_only \
  algorithm.visual_cf_base_gate_entropy_threshold=0.0 \
  algorithm.visual_cf_base_gate_tsdelta_threshold=0.0 \
  algorithm.visual_cf_final_weight_clip_enabled=false \
  algorithm.visual_cf_debug=false