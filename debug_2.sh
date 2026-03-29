#!/bin/bash
set -euo pipefail

# 现在改用空出来的 0,1 卡
export CUDA_VISIBLE_DEVICES=0,1

# 给这次任务一个独立的 Ray 端口，避免和别人撞
export MY_RAY_PORT=6399
export MY_RAY_DASHBOARD_PORT=8279

# W&B
export WANDB_PROJECT='EasyR1-SDPO'
export WANDB_NAME='qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_3gpu_debugrun_gpu01'
export WANDB_MODE=online
# 纯排障时也可改成 offline
# export WANDB_MODE=offline

# Debug
export SDPO_DEBUG_DUMP=1
export SDPO_DEBUG_DUMP_PATH="/ssd5/xhou/outputs/sdpo_debug_fullvocab_successrollout_3gpu_debugrun_gpu01.jsonl"
export SDPO_DEBUG_MAX_SAMPLES=40

export SDPO_SKIP_DEBUG=1
export SDPO_SKIP_DEBUG_MAX_GROUPS=4

# 建议先关掉视频采样 debug，减少刷屏和 I/O 干扰
export SDPO_VIDEO_SAMPLING_DEBUG=0
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=8

# Env
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="torchvision"

echo "Using Python at: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Ray port=${MY_RAY_PORT}, dashboard port=${MY_RAY_DASHBOARD_PORT}"

python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"logical cuda:{i} ->", torch.cuda.get_device_name(i))
PY

# 只清理你自己用户下的 Ray 进程
echo "Cleaning up my old Ray processes..."
pkill -u "$USER" -f raylet || true
pkill -u "$USER" -f gcs_server || true
pkill -u "$USER" -f dashboard || true
pkill -u "$USER" -f "ray::" || true
pkill -u "$USER" -f " ray " || true
pkill -u "$USER" -f ray || true
sleep 3
ray stop --force || true
sleep 2

echo "Checking remaining ray processes under my user..."
ps -u "$USER" -ef | grep ray | grep -v grep || true

unset RAY_ADDRESS

echo "Starting my isolated Ray cluster ..."
ray start \
  --head \
  --port="${MY_RAY_PORT}" \
  --dashboard-port="${MY_RAY_DASHBOARD_PORT}" \
  --num-gpus=2

export RAY_ADDRESS="localhost:${MY_RAY_PORT}"
ray status

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
  data.filter_overlong_prompts=true \
  data.rollout_batch_size=2 \
  data.max_prompt_length=3000 \
  data.max_response_length=512 \
  data.video_fps=0.5 \
  data.max_pixels=401408 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-6 \
  worker.actor.global_batch_size=2 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=true \
  worker.actor.offload.offload_optimizer=true \
  worker.actor.ppo_epochs=1 \
  worker.rollout.n=2 \
  worker.rollout.temperature=0.7 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.60 \
  worker.rollout.max_model_len=8192 \
  worker.rollout.max_num_batched_tokens=4096 \
  worker.ref.fsdp.torch_dtype=bf16 \
  trainer.total_epochs=1 \
  trainer.max_steps=100 \
  trainer.val_before_train=false \
  trainer.val_freq=1000 \
  trainer.save_freq=1000 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpo_logit_fullvocab_successrollout_3gpu_debugrun_gpu01 \
  trainer.n_gpus_per_node=2 \
  algorithm.loss_mode=sdpo_logit \
  algorithm.sdpo_approx_mode=full_vocab \
  algorithm.sdpo_topk=100 \
  algorithm.sdpo_divergence=forward_kl \
  algorithm.sdpo_use_tail=true \
  algorithm.sdpo_feedback_mode=successful_rollout