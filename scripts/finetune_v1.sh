
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

Per_device_batch_size=128
Num_beams=20

Index_file=.index_lemb_{version}.json
Image_index_file=.index_vitemb_{version}.json

Tasks='seqrec,seqimage,item2image,image2item,fusionseqrec'
Valid_task=seqrec

load_model_name=./log/Pet,Cell,Automotive,Tools,Toys,Sports/ckpt_b1024_lr1e-3_seqrec,seqimage/pretrain_v1/checkpoint-26100

# 循环运行三个数据集
for Datasets in Instruments Arts Games
do
    echo "=========================================="
    echo "Processing dataset: $Datasets"
    echo "=========================================="
    
    OUTPUT_DIR=./log/$Datasets
    mkdir -p $OUTPUT_DIR
    log_file=$OUTPUT_DIR/train.log

    torchrun --nproc_per_node=4 --master_port=2309 finetune.py \
        --index_version v1  \
        --data_path ./data/ \
        --dataset $Datasets \
        --output_dir $OUTPUT_DIR \
        --load_model_name $load_model_name \
        --per_device_batch_size $Per_device_batch_size \
        --learning_rate 5e-4 \
        --epochs 200 \
        --weight_decay 0.01 \
        --save_and_eval_strategy epoch \
        --logging_step 50 \
        --max_his_len 20 \
        --prompt_num 4 \
        --patient 10 \
        --index_file $Index_file \
        --image_index_file $Image_index_file \
        --tasks $Tasks \
        --valid_task $Valid_task > $log_file

    # Test seqrec task
    results_file=$OUTPUT_DIR/results_seqrec_${Num_beams}.json
    save_file=$OUTPUT_DIR/save_seqrec_${Num_beams}.json
    test_seqrec_log=$OUTPUT_DIR/test_seqrec.log

    torchrun --nproc_per_node=4 --master_port=2309 test_ddp_save.py \
        --index_version v1 \
        --ckpt_path $OUTPUT_DIR \
        --data_path ./data/ \
        --dataset $Datasets \
        --test_batch_size 64 \
        --num_beams 20 \
        --index_file $Index_file \
        --image_index_file $Image_index_file \
        --test_task seqrec \
        --results_file $results_file \
        --save_file $save_file \
        --filter_items > $test_seqrec_log

    # Test seqimage task
    results_file=$OUTPUT_DIR/results_seqimage_${Num_beams}.json
    save_file=$OUTPUT_DIR/save_seqimage_${Num_beams}.json
    test_seqimage_log=$OUTPUT_DIR/test_seqimage.log

    torchrun --nproc_per_node=4 --master_port=2309 test_ddp_save.py \
        --index_version v1 \
        --ckpt_path $OUTPUT_DIR \
        --data_path ./data/ \
        --dataset $Datasets \
        --test_batch_size 64 \
        --num_beams 20 \
        --index_file $Index_file \
        --image_index_file $Image_index_file \
        --test_task seqimage \
        --results_file $results_file \
        --save_file $save_file \
        --filter_items > $test_seqimage_log

    python ensemble.py \
        --index_version v1 \
        --output_dir $OUTPUT_DIR\
        --dataset $Datasets\
        --data_path ./data/\
        --index_file $Index_file\
        --image_index_file $Image_index_file\
        --num_beams 20
    
    echo "Completed: $Datasets"
    echo ""
done

echo "=========================================="
echo "All datasets processed successfully!"
echo "=========================================="

