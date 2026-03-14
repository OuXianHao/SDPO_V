#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

# W&B
# 真实 key 请在运行前于当前 shell 中 export
# export WANDB_API_KEY='your_key_here'
export WANDB_PROJECT='EasyR1-SDPO'
export WANDB_NAME='qwen3vl8b_perceptiontest_sdpot_routeA_run3'
export WANDB_MODE=online

# SDPO debug dump
export SDPO_DEBUG_DUMP=1
export SDPO_DEBUG_DUMP_PATH="/ssd5/xhou/outputs/sdpo_debug_run3.jsonl"
export SDPO_DEBUG_MAX_SAMPLES=20

# Ray
echo "Cleaning up old Ray processes..."
pkill -u $USER -f ray || true
sleep 2
ray stop --force || true

echo "Starting isolated Ray cluster..."
ray start --head --port=6388 --dashboard-port=8266

# Env
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="torchvision"
echo "Using Python at: $(which python)"

# Paths
MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"

TRAIN_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_train/train.parquet"
VAL_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_qiyuan_val/train.parquet"

python -c "import decord; print('✨ Decord 完美加载，底层组件齐全！')"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=videos \
  data.image_dir=${DATA_ROOT} \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path=${MODEL_PATH} \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.optim.lr=1e-6 \
  worker.actor.global_batch_size=8 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=true \
  worker.actor.offload.offload_optimizer=true \
  worker.rollout.n=2 \
  worker.rollout.temperature=0.7 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.35 \
  worker.rollout.max_model_len=12288 \
  worker.rollout.max_num_batched_tokens=12288 \
  data.rollout_batch_size=8 \
  data.max_prompt_length=5000 \
  data.max_response_length=4096 \
  data.video_fps=0.5 \
  data.max_pixels=602112 \
  trainer.total_epochs=1 \
  trainer.max_steps=500 \
  trainer.val_before_train=false \
  trainer.val_freq=50 \
  trainer.save_freq=-1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpot_routeA_run3 \
  trainer.n_gpus_per_node=4 \
  algorithm.use_sdpo_t=true \
  algorithm.sdpo_coef=0.1 \
  algorithm.sdpo_granularity=logits