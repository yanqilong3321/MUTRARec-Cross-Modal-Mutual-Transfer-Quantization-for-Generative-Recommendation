#!/bin/bash

# 1. 设置参数 (与原版保持一致)
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
OUTPUT_DIR=log/fusion_v4_experiment
Code_num=256
mkdir -p $OUTPUT_DIR

# 2. 运行V4版本训练 (量化前融合 + Gating + 低秩共享码本)
python -u main_fusion_v4.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MQL4GRec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 \
  --codebook_lora_rank 16 \
  --fusion_lora_rank 16 \
  --codebook_init_method random > $OUTPUT_DIR/train_v4.log 2>&1

