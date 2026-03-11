#!/bin/bash

# 1. 隔离并安全启动 Ray (带自定义端口防冲突)
echo "Cleaning up old Ray processes..."
pkill -u $USER -f ray
sleep 2
ray stop --force

echo "Starting isolated Ray cluster..."
ray start --head --port=6388 --dashboard-port=8266

# 2. 环境变量设置
export PYTHONPATH=$PWD:$PYTHONPATH
echo "Using Python at: $(which python)"
export FORCE_QWENVL_VIDEO_READER="torchvision"
# 3. 路径配置 (使用你的本地真实路径)
MODEL_PATH="/ssd5/zhzhu/models/Qwen3-VL-8B-Instruct"
DATA_ROOT="/ssd5/zhzhu/datasets/Video-R1-data"
TRAIN_DATA="/ssd5/zhzhu/datasets/Video-R1-data/PerceptionTest_parquet_train_debug/train.fixed.parquet"
python -c "import decord; print('✨ Decord 完美加载，底层组件齐全！')"
# 4. 启动主训练流程 (注入 SDPO-T 开关)
python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${TRAIN_DATA} \
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
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.video_fps=0.5 \
  data.max_pixels=602112 \
  trainer.total_epochs=1 \
  trainer.max_steps=1 \
  trainer.val_before_train=false \
  trainer.val_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger='["console"]' \
  trainer.project_name=easy_r1_smoke \
  trainer.experiment_name=qwen3vl8b_videor1_sdpo_smoke \
  trainer.n_gpus_per_node=4 \
  algorithm.use_sdpo_t=true \
  algorithm.sdpo_coef=0.1 \
  algorithm.sdpo_granularity=logits