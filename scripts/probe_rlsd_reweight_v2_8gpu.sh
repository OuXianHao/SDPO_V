#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RLSD teacher_reweight debug v3 probe: skip long responses
# 目的：
#   1. 只跑 1 step
#   2. 使用 8 张卡
#   3. 不上传 W&B
#   4. 不保存 checkpoint
#   5. dump token-level reweight debug
#   6. 验证 long-response gate 是否能跳过超长 response 的 RLSD reweight
# ============================================================

# ---------- GPU ----------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------- Proxy ----------
# 如果这台机器没有本地 7890 代理，可以注释掉这三行
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="http://127.0.0.1:7890"

# ---------- W&B ----------
# probe 不需要上传 wandb，避免污染正式曲线
export WANDB_MODE=disabled

# ---------- Runtime Debug ----------
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=1

# 关闭旧的通用 SDPO debug dump
# 这次重点是 algorithm.teacher_reweight_dump_*
export SDPO_DEBUG_DUMP=0
export SDPO_SKIP_DEBUG=1
export SDPO_DEBUG_MAX_SAMPLES=0
export SDPO_VIDEO_SAMPLING_DEBUG_MAX=0

# 打开 actor 更新侧 debug 日志
export EASYR1_DEBUG_SDPO_UPDATE=1
export EASYR1_DEBUG_COLLECTIVE=0
export EASYR1_DEBUG_FSDP_HANDLES=0
export EASYR1_DEBUG_SPLIT_BACKWARD=0

export SDPO_T_DEBUG_SHAPES=0
export SDPO_DETECT_ANOMALY=0

# 这次不是单独验证 answer span mask，先关掉
export DEBUG_ANSWER_SPAN_MASK=0

# ---------- Paths ----------
REPO_DIR="/data/xhou/frameworks/SDPO_V"
MODEL_PATH="/data/xhou/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/data/xhou/datasets/Video-R1-data"

TRAIN_DATA="${DATA_ROOT}/parquet/train_aligned_split.train.parquet"
VAL_DATA="${DATA_ROOT}/parquet/train_aligned_split.val.parquet"

# v3: 新 dump 目录，避免和 v2 混在一起
DUMP_DIR="/data/xhou/debug_dumps/rlsd_token_reweight_probe_v3_skiplong"
DUMP_FILE="${DUMP_DIR}/reweight.jsonl"

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6399}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8279}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/data/xhou/outputs/ray_${USER}_${RAY_HEAD_PORT}}"

# ---------- Prepare ----------
cd "${REPO_DIR}"

mkdir -p /data/xhou/checkpoints
mkdir -p /data/xhou/outputs
mkdir -p "${RAY_TEMP_DIR}"
mkdir -p "${DUMP_DIR}"

# 避免 append 旧结果
rm -f "${DUMP_FILE}"

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Dump file: ${DUMP_FILE}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------- Ray ----------
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

# ---------- Train / Probe ----------
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
  trainer.experiment_name=rlsd_token_reweight_probe_v3_skiplong_8gpu \
  trainer.save_checkpoint_path=/data/xhou/checkpoints/rlsd_token_reweight_probe_v3_skiplong_8gpu \
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
  algorithm.teacher_reweight_skip_long_response=true \
  algorithm.teacher_reweight_skip_min_response_len=1024 \
  algorithm.teacher_reweight_dump_enabled=true \
  algorithm.teacher_reweight_dump_path="${DUMP_FILE}" \
  algorithm.teacher_reweight_dump_max_samples=20 \
  algorithm.teacher_reweight_dump_max_correct=12 \
  algorithm.teacher_reweight_dump_max_incorrect=8

echo "[INFO] Probe finished."

# ---------- Check dump ----------
echo "[INFO] Dump file:"
ls -lh "${DUMP_FILE}" || true

echo "[INFO] Number of dumped records:"
wc -l "${DUMP_FILE}" || true

echo "[INFO] Summarizing reweight dump..."
python scripts/summarize_reweight.py "${DUMP_FILE}" --topk 30

# ---------- Inspect long-response skipped samples ----------
echo "[INFO] Inspect long-response skipped samples:"
python - <<'PY'
import json

path = "/data/xhou/debug_dumps/rlsd_token_reweight_probe_v3_skiplong/reweight.jsonl"

try:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    print(f"[WARN] dump file not found: {path}")
    raise SystemExit(0)

for r in rows:
    if r.get("skip_reweight_due_to_long_response") or r.get("hit_max_response_len"):
        print("=" * 100)
        print("sample_idx:", r.get("sample_idx"))
        print("is_correct:", r.get("is_correct"))
        print("response_len:", r.get("response_len"))
        print("max_response_length:", r.get("max_response_length"))
        print("hit_max_response_len:", r.get("hit_max_response_len"))
        print("skip_reweight_due_to_long_response:", r.get("skip_reweight_due_to_long_response"))
        print("inactive_reason:", r.get("inactive_reason"))
        print("num_reweighted_tokens_before_answer_mask:", r.get("num_reweighted_tokens_before_answer_mask"))
        print("num_reweighted_tokens_after_answer_mask:", r.get("num_reweighted_tokens_after_answer_mask"))
        print("repetition_4gram_ratio:", r.get("repetition_4gram_ratio"))
        print("parsed_prediction:", r.get("parsed_prediction"))
        print("ground_truth_answer:", r.get("ground_truth_answer"))
PY