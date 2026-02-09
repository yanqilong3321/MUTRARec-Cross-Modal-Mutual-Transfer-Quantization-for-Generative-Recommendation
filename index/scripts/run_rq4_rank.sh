#!/bin/bash

# 设置参数
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
Code_num=256
EPOCHS=500
ALPHA=0.1

# 1. Rank 8
OUTPUT_DIR_8=log/rq4_rank_8
mkdir -p $OUTPUT_DIR_8
echo "Starting Rank 8..."
python -u main_fusion_v3.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR_8 \
  --eval_step 10 \
  --batch_size 2048 \
  --epochs $EPOCHS \
  --codebook_lora_rank 8 \
  --fusion_lora_rank 16 \
  --fusion_lora_alpha $ALPHA \
  --codebook_init_method random > $OUTPUT_DIR_8/train.log 2>&1 &

# 2. Rank 32
OUTPUT_DIR_32=log/rq4_rank_32
mkdir -p $OUTPUT_DIR_32
echo "Starting Rank 32..."
python -u main_fusion_v3.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR_32 \
  --eval_step 10 \
  --batch_size 2048 \
  --epochs $EPOCHS \
  --codebook_lora_rank 32 \
  --fusion_lora_rank 16 \
  --fusion_lora_alpha $ALPHA \
  --codebook_init_method random > $OUTPUT_DIR_32/train.log 2>&1 &

# 3. Rank 64
OUTPUT_DIR_64=log/rq4_rank_64
mkdir -p $OUTPUT_DIR_64
echo "Starting Rank 64..."
python -u main_fusion_v3.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR_64 \
  --eval_step 10 \
  --batch_size 2048 \
  --epochs $EPOCHS \
  --codebook_lora_rank 64 \
  --fusion_lora_rank 16 \
  --fusion_lora_alpha $ALPHA \
  --codebook_init_method random > $OUTPUT_DIR_64/train.log 2>&1 &

echo "All jobs submitted to background."
