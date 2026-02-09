Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
CKPT_DIR=log/fusion_experiment

# 遍历所有数据集生成索引
for Dataset in Instruments Arts Games Pet Cell Automotive Tools Toys Sports
do
    echo "=========================================="
    echo "Processing dataset: $Dataset"
    echo "=========================================="
    
    OUTPUT_DIR=../data/$Dataset
    
    # 使用 best_collision_model.pth (或 best_loss_model.pth)
    # 注意：这里我们只需要运行一次脚本，因为它会同时生成 Text 和 Vis 的 json
    python -u generate_fusion_indices.py \
      --dataset $Dataset \
      --device cuda:0 \
      --ckpt_path ${CKPT_DIR}/best_collision_model.pth \
      --output_dir $OUTPUT_DIR 
    
    echo "Completed: $Dataset"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="

