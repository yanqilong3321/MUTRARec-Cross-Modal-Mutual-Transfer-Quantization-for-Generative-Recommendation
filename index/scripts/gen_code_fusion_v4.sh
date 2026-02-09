#!/bin/bash

Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
CKPT_DIR=log/fusion_v4_experiment

for Dataset in Instruments Arts Games Pet Cell Automotive Tools Toys Sports
do
    echo "=========================================="
    echo "Processing dataset: $Dataset (V4)"
    echo "=========================================="
    
    OUTPUT_DIR=../data/$Dataset
    
    python -u generate_fusion_indices_v4.py \
      --dataset $Dataset \
      --device cuda:0 \
      --ckpt_path ${CKPT_DIR}/best_collision_model_v4.pth \
      --output_dir $OUTPUT_DIR 
    
    echo "Completed: $Dataset"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="

