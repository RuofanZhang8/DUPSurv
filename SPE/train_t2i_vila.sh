#!/bin/bash

# 设置模型与数据集
export MODEL_NAME="<PATH_TO_PRETRAINED_MODEL>"
export DATASET_NAME="<DATASET_IDENTIFIER>"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT='PROJECT'
export WANDB_MODE="offline"

# 切换到工作目录
cd <WORK_DIR>

# 启动训练
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_ldm_vila.py \
  --variant=fp16 \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=40010 \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt "Validation Prompt" \
  --output_dir="<OUTPUT_DIR>" \
  --prototype_number_text=32 \
  --prototype_number_image=32 \
  --checkpointing_steps=5000 \
  --fold 2 \
  --text_encoder='clip' \
  --dataset='BLCA'
