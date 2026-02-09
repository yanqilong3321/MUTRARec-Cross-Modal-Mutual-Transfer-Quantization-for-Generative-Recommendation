Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'

# 遍历所有数据集生成索引
for Dataset in Instruments Arts Games Pet Cell Automotive Tools Toys Sports
do
    echo "=========================================="
    echo "Processing dataset: $Dataset"
    echo "=========================================="
    
    OUTPUT_DIR=../data/$Dataset
    
    # 生成 LLaMA 文本 embedding 索引
    python -u generate_indices_distance.py \
      --dataset $Dataset \
      --device cuda:0 \
      --ckpt_path log/$Datasets/llama_256/best_collision_model.pth \
      --output_dir $OUTPUT_DIR \
      --output_file ${Dataset}.index_lemb.json

    # 生成 ViT 图像 embedding 索引
    python -u generate_indices_distance.py \
        --dataset $Dataset \
        --device cuda:0 \
        --ckpt_path log/$Datasets/ViT-L-14_256/best_collision_model.pth \
        --output_dir $OUTPUT_DIR \
        --output_file ${Dataset}.index_vitemb.json \
        --content image
    
    echo "Completed: $Dataset"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="

