#!/bin/bash

# Script for timing comparison - Independent Quantization with detailed timing logs
# This script will NOT overwrite old logs

Model_Text=llama
Model_Vision=ViT-L-14
Code_num=256
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'

echo "============================================"
echo "Training Independent Quantization Models with Timing"
echo "============================================"

# Text model (LLaMA embeddings)
echo ""
echo ">>> Training Text Model (LLaMA)..."
OUTPUT_DIR=log_timing/$Datasets/${Model_Text}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --embedding_file .emb-llama-td.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 50 > $OUTPUT_DIR/train_timing.log 2>&1

echo "Text model training completed. Log saved to: $OUTPUT_DIR/train_timing.log"

# Vision model (ViT-L-14 embeddings)
echo ""
echo ">>> Training Vision Model (ViT-L-14)..."
OUTPUT_DIR=log_timing/$Datasets/${Model_Vision}_${Code_num}
mkdir -p $OUTPUT_DIR

python -u main_mul.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --embedding_file .emb-ViT-L-14.npy \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 50 > $OUTPUT_DIR/train_timing.log 2>&1

echo "Vision model training completed. Log saved to: $OUTPUT_DIR/train_timing.log"

echo ""
echo "============================================"
echo "All training completed!"
echo "Check logs in: log_timing/$Datasets/"
echo "============================================"
