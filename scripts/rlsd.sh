#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------- Proxy ----------
# 如果这台机器没有本地 7890 代理，就保持注释
# export http_proxy="http://127.0.0.1:7890"
# export https_proxy="http://127.0.0.1:7890"
# export all_proxy="http://127.0.0.1:7890"

# ---------- W&B ----------
# 建议不要在脚本里明文写 WANDB_API_KEY。
# 你可以运行脚本前手动 export WANDB_API_KEY=...
export WANDB_API_KEY='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'
export WANDB_PROJECT="EasyR1-RLSD"
export WANDB_NAME="qwen3vl8b_rlsd_t_rollout5_rl1e6_run7_8gpu_lamda05_skiplong1024"
export WANDB_MODE="online"

# ---------- Runtime ----------
# 正式训练不要开 CUDA_LAUNCH_BLOCKING=1，会明显变慢
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=1

# ---------- Debug ----------
# 正式训练关闭大规模通用 debug dump，避免 I/O 压力
export SDPO_DEBUG_DUMP=0
export SDPO_SKIP_DEBUG=1
export SDPO_DEBUG_MAX_SAMPLES=0
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=0

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

# ---------- Runtime ----------
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="decord"

MODEL_PATH="/data/xhou/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/data/xhou/datasets/Video-R1-data"

TRAIN_DATA="/data/xhou/datasets/Video-R1-data/parquet/train_avg5_filtered.train.parquet"
VAL_DATA="/data/xhou/datasets/Video-R1-data/parquet/train_avg5_filtered_split.val.parquet"

EXP_NAME="qwen3vl8b_rlsd_t_rollout5_rl1e6_run7_8gpu_lamda05_skiplong1024_zerostep50_newdata"
CKPT_DIR="/data/xhou/checkpoints/${EXP_NAME}"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=path \
  data.image_dir="${DATA_ROOT}" \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=128 \
  data.max_prompt_length=6144 \
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
  worker.actor.global_batch_size=128 \
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
  worker.rollout.max_num_batched_tokens=32768 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  worker.ref.offload.offload_params=false \
  trainer.total_epochs=1 \
  trainer.max_steps=156 \
  trainer.val_before_train=true \
  trainer.val_freq=5 \
  trainer.save_freq=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1-RLSD \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.save_checkpoint_path="${CKPT_DIR}" \
  trainer.find_last_checkpoint=true \
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
  algorithm.teacher_reweight_lambda=0.5 \
  algorithm.teacher_reweight_lambda_decay_to_zero_step=50 \
  algorithm.teacher_reweight_eps_w_low=0.2 \
  algorithm.teacher_reweight_eps_w_high=0.2 \
  algorithm.teacher_reweight_delta_clamp=5.0 \
  algorithm.teacher_reweight_correct_hint=true \
  algorithm.teacher_reweight_skip_long_response=true \
  algorithm.teacher_reweight_skip_min_response_len=1024 \
  algorithm.teacher_reweight_dump_enabled=false