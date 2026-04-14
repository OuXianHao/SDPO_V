#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---------- Proxy ----------
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="http://127.0.0.1:7890"

# ---------- W&B ----------
export WANDB_API_KEY='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'
export WANDB_PROJECT="EasyR1-SDPO"
export WANDB_NAME="qwen3vl8b_perceptiontest_sdpo_v_softkl_run1"
export WANDB_MODE="online"
# 更稳一点也可以先改成 offline

# ---------- Debug ----------
export SDPO_DEBUG_DUMP=1
export SDPO_SKIP_DEBUG=0
export SDPO_DEBUG_DUMP_PATH="/ssd5/xhou/outputs/sdpo_v_softkl_run1.jsonl"
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=200

export EASYR1_DEBUG_SDPO_UPDATE=0
export EASYR1_DEBUG_COLLECTIVE=0
export EASYR1_DEBUG_FSDP_HANDLES=0
export EASYR1_DEBUG_SPLIT_BACKWARD=0

export SDPO_T_DEBUG_SHAPES=0
export SDPO_DETECT_ANOMALY=0

# ---------- Ray ----------
echo "[INFO] Cleaning stale Ray processes..."
pkill -u "$USER" -f ray || true
sleep 2
ray stop --force || true

echo "[INFO] Starting Ray head..."
RAY_HEAD_PORT="${RAY_HEAD_PORT:-7388}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-7866}"
echo "[INFO] RAY_HEAD_PORT=${RAY_HEAD_PORT}"
echo "[INFO] RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}"
ray start --head --port="${RAY_HEAD_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}"

# ---------- Runtime ----------
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export FORCE_QWENVL_VIDEO_READER="torchvision"

MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"

TRAIN_DATA="/ssd5/zhzhu/dataset/Video-R1-data/Video-R1-20k-video-multiple-choice_seq_len_lt_3000_mcprompt.train.parquet"
VAL_DATA="/ssd5/zhzhu/dataset/Video-R1-data/Video-R1-20k-video-multiple-choice_seq_len_lt_3000_mcprompt.val.parquet"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=videos \
  data.image_dir="${DATA_ROOT}" \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.max_pixels=524288 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-5 \
  worker.actor.global_batch_size=32 \
  worker.actor.micro_batch_size_per_device_for_update=4 \
  worker.actor.micro_batch_size_per_device_for_experience=2 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.actor.ppo_epochs=1 \
  worker.rollout.n=8 \
  worker.rollout.temperature=0.7 \
  worker.rollout.top_p=1.0 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.65 \
  worker.rollout.max_model_len=8192 \
  worker.rollout.max_num_batched_tokens=16384 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  trainer.total_epochs=1 \
  trainer.max_steps=80 \
  trainer.val_before_train=true \
  trainer.val_freq=10 \
  trainer.save_freq=20 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpo_v_softkl_run1 \
  trainer.save_checkpoint_path=/ssd5/xhou/outputs/checkpoints/qwen3vl8b_perceptiontest_sdpo_v_softkl_run1 \
  trainer.find_last_checkpoint=true \
  trainer.n_gpus_per_node=4 \
  algorithm.loss_mode=sdpo_logit \
  algorithm.disable_kl=false \
  algorithm.sdpo_feedback_mode=scalar_text \
  algorithm.sdpo_approx_mode=student_topk_tail \
  algorithm.sdpo_topk=100 \
  algorithm.sdpo_divergence=forward_kl \
  algorithm.sdpo_alpha=0.5 \
  algorithm.sdpo_teacher_update_rate=0.05 \
  algorithm.sdpo_use_tail=true \
  algorithm.sdpo_v_enabled=false \
  algorithm.sdpo_v_softkl_enabled=true \
  algorithm.sdpo_v_softkl_weight=1.0 \
  algorithm.sdpo_v_softkl_topk=100 \
  algorithm.sdpo_v_softkl_tau=1.0 \
  algorithm.sdpo_v_softkl_use_tail=false \
  algorithm.sdpo_v_softkl_use_ema_bad_ref=true \
  algorithm.sdpo_v_softkl_debug=true \
  algorithm.sdpo_v_bad_video_mode=blur \
  algorithm.sdpo_v_blur_sigma=5.0 \
  algorithm.sdpo_v_blur_fraction=0.5 \
  algorithm.sdpo_v_drop_fraction=0.5