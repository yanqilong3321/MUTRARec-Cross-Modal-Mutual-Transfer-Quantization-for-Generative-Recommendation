#!/bin/bash

# 版本B的超参数配置
# 可以自由修改以下参数进行调试

Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
OUTPUT_DIR=log/fusion_v3_B
Code_num=256
mkdir -p $OUTPUT_DIR

# 超参数配置（可修改）
CODEBOOK_LORA_RANK=16
FUSION_LORA_RANK=16
FUSION_ALPHA=0.2

# 运行V3版本训练
python -u main_fusion_v3.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:2 \
  --data_root /data/yql/workspace/MQL4GRec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --log_file train_v3_B.log \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 \
  --codebook_lora_rank $CODEBOOK_LORA_RANK \
  --fusion_lora_rank $FUSION_LORA_RANK \
  --fusion_lora_alpha $FUSION_ALPHA \
  --codebook_init_method random



