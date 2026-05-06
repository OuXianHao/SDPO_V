#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------- Proxy ----------
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="http://127.0.0.1:7890"

# ---------- W&B ----------
export WANDB_MODE=disabled

export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=1

# ---------- Debug for RLSD token reweight probe ----------
export SDPO_DEBUG_DUMP=0
export SDPO_SKIP_DEBUG=1

export EASYR1_DEBUG_SDPO_UPDATE=1
export EASYR1_DEBUG_COLLECTIVE=0
export EASYR1_DEBUG_FSDP_HANDLES=0
export EASYR1_DEBUG_SPLIT_BACKWARD=0

export SDPO_T_DEBUG_SHAPES=0
export SDPO_DETECT_ANOMALY=0

# ---------- Ray ----------
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6388}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/data/xhou/outputs/ray_${USER}_${RAY_HEAD_PORT}}"

mkdir -p /data/xhou/checkpoints
mkdir -p /data/xhou/outputs
mkdir -p "${RAY_TEMP_DIR}"
mkdir -p /data/xhou/debug_dumps/rlsd_token_reweight_probe_lamda07

echo "[INFO] Cleaning stale Ray processes..."
pkill -u "$USER" -f ray || true
sleep 2
ray stop --force || true

echo "[INFO] Starting Ray head..."
ray start --head \
  --port="${RAY_HEAD_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}"

# ---------- Runtime ----------
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="decord"

MODEL_PATH="/data/xhou/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/data/xhou/datasets/Video-R1-data"

TRAIN_DATA="/data/xhou/datasets/Video-R1-data/parquet/train_aligned_split.train.parquet"
VAL_DATA="/data/xhou/datasets/Video-R1-data/parquet/train_aligned_split.val.parquet"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=path \
  data.image_dir="${DATA_ROOT}" \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=1536 \
  data.max_pixels=524288 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-5 \
  worker.actor.optim.lr_scheduler_type=cosine \
  worker.actor.optim.min_lr_ratio=0.1 \
  worker.actor.optim.lr_warmup_steps=10 \
  worker.actor.global_batch_size=32 \
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
  worker.rollout.max_model_len=8192 \
  worker.rollout.max_num_batched_tokens=16384 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  worker.ref.offload.offload_params=false \
  trainer.total_epochs=1 \
  trainer.max_steps=1 \
  trainer.val_before_train=false \
  trainer.val_freq=-1 \
  trainer.save_freq=1000000 \
  trainer.logger='["console"]' \
  trainer.project_name=EasyR1-RLSD \
  trainer.experiment_name=qwen3vl8b_rlsd_token_reweight_probe_lamda07 \
  trainer.save_checkpoint_path=/data/xhou/checkpoints/qwen3vl8b_rlsd_token_reweight_probe_lamda07 \
  trainer.find_last_checkpoint=false \
  trainer.n_gpus_per_node=8 \
  algorithm.loss_mode=dapo_with_sdpo \
  algorithm.disable_kl=false \
  algorithm.sdpo_feedback_mode=guideline_mixed_rollouts \
  algorithm.sdpo_teacher_update_rate=0.05 \
  algorithm.lambda_dapo=1.0 \
  algorithm.lambda_sdpo_t=0.0 \
  algorithm.sdpo_v_enabled=false \
  algorithm.lambda_sdpo_v=0.0 \
  algorithm.teacher_reweight_enabled=true \
  algorithm.teacher_reweight_lambda=0.7 \
  algorithm.teacher_reweight_lambda_decay_to_zero_step=156 \
  algorithm.teacher_reweight_eps_w_low=0.2 \
  algorithm.teacher_reweight_eps_w_high=0.2 \
  algorithm.teacher_reweight_delta_clamp=5.0 \
  algorithm.teacher_reweight_correct_hint=true \
  algorithm.teacher_reweight_dump_enabled=true \
  algorithm.teacher_reweight_dump_path=/data/xhou/debug_dumps/rlsd_token_reweight_probe_lamda07/reweight.jsonl \
  algorithm.teacher_reweight_dump_max_samples=500 \
  algorithm.teacher_reweight_dump_max_correct=200 \
  algorithm.teacher_reweight_dump_max_incorrect=200