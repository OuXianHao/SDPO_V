#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5

# ---------- W&B ----------
export WANDB_PROJECT="EasyR1-SDPO"
export WANDB_NAME="qwen3vl8b_perceptiontest_sdpo_logit_student_topk_tail_successrollout_a800x2_probe_37"
export WANDB_MODE="online"

# ---------- SDPO debug dump ----------
export SDPO_DEBUG_DUMP=1
export SDPO_DEBUG_DUMP_PATH="/ssd5/xhou/outputs/sdpo_debug_student_topk_tail_successrollout_a800x2_probe_37.jsonl"
export SDPO_DEBUG_MAX_SAMPLES=40

# ---------- Skip-group debug ----------
export SDPO_SKIP_DEBUG=1
export SDPO_SKIP_DEBUG_MAX_GROUPS=4

# ---------- Video sampling debug ----------
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=16

# ---------- New instrumentation from codex ----------
export EASYR1_DEBUG_SDPO_UPDATE=1
export EASYR1_DEBUG_COLLECTIVE=1
export EASYR1_DEBUG_FSDP_HANDLES=1
export EASYR1_DEBUG_SPLIT_BACKWARD=1

export SDPO_T_DEBUG_SHAPES=1
export SDPO_DETECT_ANOMALY=1
# export SDPO_EXP=1
# export SDPO_FWD_EXP=1

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

MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"
TRAIN_DATA="/ssd5/zhzhu/datasets/Video-R1-data/avgk_outputs/video_mcq_selected_top3000_clean_parquet/train_compat.parquet"
VAL_DATA="/ssd5/zhzhu/datasets/Video-R1-data/avgk_outputs/video_mcq_selected_top3000_clean_parquet/val_compat.parquet"

python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.prompt_key=problem \
  data.answer_key=answer \
  data.video_key=videos \
  data.image_dir="${DATA_ROOT}" \
  data.format_prompt=./examples/format_prompt/r1v.jinja \
  data.rollout_batch_size=4 \
  data.max_prompt_length=3500 \
  data.max_response_length=1024 \
  data.max_pixels=401408 \
  worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=0 \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.lr=1e-6 \
  worker.actor.global_batch_size=4 \
  worker.actor.micro_batch_size_per_device_for_update=1 \
  worker.actor.micro_batch_size_per_device_for_experience=1 \
  worker.actor.padding_free=false \
  worker.actor.dynamic_batching=false \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.actor.ppo_epochs=1 \
  worker.rollout.n=4 \
  worker.rollout.temperature=0.7 \
  worker.rollout.top_p=1.0 \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.gpu_memory_utilization=0.45 \
  worker.rollout.max_model_len=8192 \
  worker.rollout.max_num_batched_tokens=8192 \
  worker.rollout.val_override_config='{"temperature":0.7,"top_p":1.0,"n":1}' \
  worker.ref.fsdp.torch_dtype=bf16 \
  trainer.total_epochs=1 \
  trainer.max_steps=20 \
  trainer.val_before_train=false \
  trainer.val_freq=1000 \
  trainer.save_freq=1000 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=EasyR1_SDPO \
  trainer.experiment_name=qwen3vl8b_perceptiontest_sdpo_logit_student_topk_tail_successrollout_a800x2_probe_37 \
  trainer.n_gpus_per_node=2 \
  algorithm.loss_mode=sdpo_logit \
  algorithm.sdpo_approx_mode=student_topk_tail \
  algorithm.sdpo_topk=100 \
  algorithm.sdpo_divergence=forward_kl \
  algorithm.sdpo_alpha=0.5 \
  algorithm.sdpo_teacher_update_rate=0.05 \
  algorithm.sdpo_use_tail=true \
  algorithm.sdpo_feedback_mode=successful_rollout