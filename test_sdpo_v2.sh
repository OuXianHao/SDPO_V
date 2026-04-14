#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# W&B
# 请先在当前 shell 中 export WANDB_API_KEY='your_key_here'
export WANDB_PROJECT='EasyR1-SDPO'
export WANDB_NAME='qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_run9'
export WANDB_MODE=online

# SDPO debug dump
# 第一次跑先保留少量，重点检查 successful sibling rollout / teacher prompt / logits path
export SDPO_DEBUG_DUMP=1
export SDPO_DEBUG_DUMP_PATH="/ssd5/xhou/outputs/sdpo_debug_fullvocab_successrollout_run9.jsonl"
export SDPO_DEBUG_MAX_SAMPLES=150

# 检查 skip-group / no-sibling 逻辑
export SDPO_SKIP_DEBUG=1
export SDPO_SKIP_DEBUG_MAX_GROUPS=8

# 视频采样 debug
export SDPO_VIDEO_SAMPLING_DEBUG=1
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=32

# 如需开启 top-k 细粒度 debug，可手动打开
# export SDPO_TOPK_DEBUG=1
# export SDPO_TOPK_DEBUG_POSITIONS=3

# Ray
echo "Cleaning up old Ray processes..."
pkill -u "$USER" -f ray || true
sleep 2
ray stop --force || true

echo "Starting isolated Ray cluster..."
RAY_HEAD_PORT="${RAY_HEAD_PORT:-7388}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-7866}"
echo "[INFO] RAY_HEAD_PORT=${RAY_HEAD_PORT}"
echo "[INFO] RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}"
ray start --head --port="${RAY_HEAD_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}"

# Env
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="torchvision"
echo "Using Python at: $(which python)"

# Paths
MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"

TRAIN_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_train/train.parquet"
VAL_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_val/train.parquet"

# Quick env check
python -c "import decord; print('✨ Decord 完美加载，底层组件齐全！')"

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
  data.video_fps=0.5 \
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
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.50 \
  worker.rollout.max_model_len=12288 \
  worker.rollout.max_num_batched_tokens=12288 \
  worker.ref.fsdp.torch_dtype=bf16 \
  trainer.total_epochs=1 \
  trainer.max_steps=120 \
  trainer.val_before_train=false \
  trainer.val_freq=20 \
  trainer.save_freq=60 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_run9 \
  trainer.n_gpus_per_node=4 \
  algorithm.loss_mode=sdpo_logit \
  algorithm.sdpo_approx_mode=full_vocab \
  algorithm.sdpo_topk=100 \
  algorithm.sdpo_divergence=forward_kl \
  algorithm.sdpo_use_tail=true \
  algorithm.sdpo_feedback_mode=successful_rollout