#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Đổi tên thử nghiệm
EXPERIMENT_NAME="SOTA_MaxViT_Focal"

CODE_DIR="code/classification"
SERVER_SETUP="arc_fastscratch" 

MODEL_NAME="maxvit_t"
DATASET_NAME="fishair_processed"
# THAY ĐỔI LOSS Ở ĐÂY
LOSS_TYPE="Focal"
OPTIMIZER="AdamW"
LEARNING_RATE="3e-4"
LR_WARMUP="50"
EPOCHS="150"
BATCH_SIZE="16" # Giữ nguyên hoặc giảm xuống 64 nếu vẫn OOM
DECAY="0.1"
NUM_WORKERS="8"
SEED="9229"

echo "Bắt đầu huấn luyện baseline SOTA: ${MODEL_NAME} với ${LOSS_TYPE} Loss"

python ${CODE_DIR}/train.py \
    --model ${MODEL_NAME} \
    --dataset ${DATASET_NAME} \
    --loss_type ${LOSS_TYPE} \
    --optimizer ${OPTIMIZER} \
    --lr ${LEARNING_RATE} \
    --lr_warmup ${LR_WARMUP} \
    --epoch ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --decay ${DECAY} \
    --name ${EXPERIMENT_NAME} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS} \
    --server ${SERVER_SETUP} \
    --cosine_annealing \
    --wandb \
    --no_over

echo "Hoàn thành huấn luyện baseline SOTA."