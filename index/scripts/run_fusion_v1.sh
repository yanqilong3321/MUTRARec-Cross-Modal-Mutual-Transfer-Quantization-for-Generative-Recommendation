#!/bin/bash

# 1. 设置参数
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
OUTPUT_DIR=log/fusion_v1_experiment
Code_num=256
mkdir -p $OUTPUT_DIR

# 2. 配置V1特有参数
LORA_RANK=16  # 低秩维度，可调整: 4, 8, 16, 32
#INIT_METHOD="random"  # 初始化方法: "random" 或 "kmeans_svd"
INIT_METHOD="kmeans_svd"  # 初始化方法: "random" 或 "kmeans_svd"
echo "=========================================="
echo "FusionRQVAE V1: Low-Rank Shared Codebook"
echo "=========================================="
echo "LoRA Rank: $LORA_RANK"
echo "Init Method: $INIT_METHOD"
echo "=========================================="

# 3. 运行V1版本训练 (低秩共享码本)
python -u main_fusion_v1.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 \
  --lora_rank $LORA_RANK \
  --codebook_init_method $INIT_METHOD > $OUTPUT_DIR/train_v1.log 2>&1

echo "=========================================="
echo "Training completed! Check logs at:"
echo "  $OUTPUT_DIR/train_v1.log"
echo "=========================================="

