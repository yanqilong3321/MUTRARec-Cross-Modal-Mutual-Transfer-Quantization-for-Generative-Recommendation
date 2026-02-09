
# 1. 设置参数 (完全对齐基准的 run.sh)
Datasets='Instruments,Arts,Games,Pet,Cell,Automotive,Tools,Toys,Sports'
OUTPUT_DIR=log/fusion_experiment
Code_num=256
mkdir -p $OUTPUT_DIR

# 2. 运行融合训练
# 注意：我们使用 main_fusion.py，它会同时优化 text 和 vis 的 RQ-VAE
# 所有超参数与 run.sh 保持一致，除了新增的 lora_rank 和 lora_alpha
python -u main_fusion.py \
  --num_emb_list $Code_num $Code_num $Code_num $Code_num \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:0 \
  --data_root /data/yql/workspace/MUTRARec/data/ \
  --datasets $Datasets \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --epochs 500 \
  --lora_rank 32 \
  --lora_alpha 0.1 > $OUTPUT_DIR/train.log 2>&1

