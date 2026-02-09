#!/bin/bash

Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
CKPT_DIR=log/fusion_v1_experiment

# 遍历所有数据集生成索引
for Dataset in Instruments Arts Games Pet Cell Automotive Tools Toys Sports
do
    echo "=========================================="
    echo "Processing dataset: $Dataset (V1 Low-Rank)"
    echo "=========================================="
    
    OUTPUT_DIR=../data/$Dataset
    
    # 使用 V1 版本的 best_collision_model_v1.pth
    python -u generate_fusion_indices_v1.py \
      --dataset $Dataset \
      --device cuda:0 \
      --ckpt_path ${CKPT_DIR}/best_collision_model_v1.pth \
      --output_dir $OUTPUT_DIR 
    
    echo "Completed: $Dataset"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="


