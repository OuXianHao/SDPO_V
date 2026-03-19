#!/usr/bin/env bash
set -euo pipefail

# =========================
# 4x A800 SDPO-T 训练脚本
# Qwen3-VL-8B-Instruct + successful_rollout
# =========================

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---------- W&B ----------
export WANDB_PROJECT="EasyR1-SDPO"
export WANDB_NAME="qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_a800x4"
export WANDB_MODE="online"

# ---------- Ray ----------
echo "[INFO] Cleaning stale Ray processes..."
pkill -u "$USER" -f ray || true
sleep 2
ray stop --force || true

echo "[INFO] Starting Ray head..."
ray start --head --port=6388 --dashboard-port=8266

# ---------- Runtime ----------
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="torchvision"

# Optional debug: verify video sampling behavior in first run.
# 生产训练可设为 0。
export SDPO_VIDEO_SAMPLING_DEBUG=1
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=32

MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"
TRAIN_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_train/train.parquet"
VAL_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_val/train.parquet"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=videos \
  data.image_dir="${DATA_ROOT}" \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=8 \
  data.max_prompt_length=3500 \
  data.max_response_length=2048 \
  data.max_pixels=401408 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-6 \
  worker.actor.global_batch_size=8 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=true \
  worker.actor.offload.offload_optimizer=true \
  worker.actor.ppo_epochs=1 \
  worker.rollout.n=8 \
  worker.rollout.temperature=0.7 \
  worker.rollout.top_p=1.0 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.50 \
  worker.rollout.max_model_len=12288 \
  worker.rollout.max_num_batched_tokens=12288 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  trainer.total_epochs=1 \
  trainer.max_steps=120 \
  trainer.val_before_train=false \
  trainer.val_freq=20 \
  trainer.save_freq=60 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_a800x4 \
  trainer.n_gpus_per_node=4 \
  algorithm.loss_mode=sdpo_logit \
  algorithm.sdpo_approx_mode=full_vocab \
  algorithm.sdpo_topk=100 \
  algorithm.sdpo_divergence=forward_kl \
  algorithm.sdpo_use_tail=true \
  algorithm.sdpo_feedback_mode=successful_rollout
