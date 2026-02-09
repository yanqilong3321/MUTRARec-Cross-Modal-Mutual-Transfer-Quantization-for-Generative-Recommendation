#!/usr/bin/env python3
"""
NNI Trial Wrapper - 单数据集版本
只针对一个数据集进行训练和优化，找到该数据集的最佳参数
"""

import os
import sys
import json
import subprocess
import re
import nni
from pathlib import Path


def extract_metrics_from_results(results_file):
    """
    从测试结果 JSON 文件中提取所有 Hit 和 NDCG 指标
    """
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        metrics = {}
        
        # 如果结果包含 mean_results，优先使用它
        if 'mean_results' in results:
            results = results['mean_results']
        
        # 定义需要提取的指标及其可能的键名变体
        metric_mappings = {
            'hit@1': ['hit@1', 'Hit@1', 'hit_1', 'Hit_1'],
            'hit@5': ['hit@5', 'Hit@5', 'hit_5', 'Hit_5'],
            'hit@10': ['hit@10', 'Hit@10', 'hit_10', 'Hit_10'],
            'ndcg@5': ['ndcg@5', 'NDCG@5', 'ndcg_5', 'NDCG_5'],
            'ndcg@10': ['ndcg@10', 'NDCG@10', 'ndcg_10', 'NDCG_10'],
        }
        
        # 提取所有指标
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in results:
                    metrics[metric_name] = float(results[key])
                    break
        
        # 如果没有提取到任何指标，返回 None
        if not metrics:
            print(f"警告：未能从结果文件中提取到任何指标")
            return None
        
        return metrics
    except Exception as e:
        print(f"警告：无法从 {results_file} 提取指标: {e}")
        return None


def extract_metrics_from_log(log_file):
    """
    从测试日志文件中提取所有 Hit 和 NDCG 指标
    用于备选方案，如果 JSON 文件格式不符合预期
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        metrics = {}
        
        # 使用正则表达式匹配常见的指标格式
        metric_patterns = {
            'hit@1': [r'hit@1[:\s=]+([0-9.]+)', r'Hit@1[:\s=]+([0-9.]+)'],
            'hit@5': [r'hit@5[:\s=]+([0-9.]+)', r'Hit@5[:\s=]+([0-9.]+)'],
            'hit@10': [r'hit@10[:\s=]+([0-9.]+)', r'Hit@10[:\s=]+([0-9.]+)'],
            'ndcg@5': [r'ndcg@5[:\s=]+([0-9.]+)', r'NDCG@5[:\s=]+([0-9.]+)'],
            'ndcg@10': [r'ndcg@10[:\s=]+([0-9.]+)', r'NDCG@10[:\s=]+([0-9.]+)'],
        }
        
        for metric_name, patterns in metric_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    metrics[metric_name] = float(match.group(1))
                    break
        
        return metrics if metrics else None
    except Exception as e:
        print(f"警告：无法从 {log_file} 提取指标: {e}")
        return None


def calculate_score(metrics, strategy='hit1_focused'):
    """
    根据策略计算优化得分
    """
    if strategy == 'ndcg5_hit5':
        score = 0.5 * metrics.get('ndcg@5', 0.0) + 0.5 * metrics.get('hit@5', 0.0)
        strategy_desc = "NDCG@5 & Hit@5 均衡 (0.5×NDCG@5 + 0.5×Hit@5)"
    elif strategy == 'ndcg10_only':
        score = metrics.get('ndcg@10', 0.0)
        strategy_desc = "单一优化 NDCG@10"
    elif strategy == 'balanced':
        score = 0.6 * metrics.get('ndcg@10', 0.0) + 0.4 * metrics.get('hit@10', 0.0)
        strategy_desc = "平衡策略 (0.6×NDCG@10 + 0.4×Hit@10)"
    elif strategy == 'comprehensive':
        score = (0.4 * metrics.get('ndcg@10', 0.0) +
                0.2 * metrics.get('ndcg@5', 0.0) +
                0.3 * metrics.get('hit@10', 0.0) +
                0.1 * metrics.get('hit@5', 0.0))
        strategy_desc = "综合策略"
    elif strategy == 'hit1_focused':
        score = (0.4 * metrics.get('hit@1', 0.0) +
                0.3 * metrics.get('ndcg@5', 0.0) +
                0.3 * metrics.get('hit@5', 0.0))
        strategy_desc = "首位命中优先 (0.4×Hit@1 + 0.3×NDCG@5 + 0.3×Hit@5)"
    else:
        score = 0.5 * metrics.get('ndcg@5', 0.0) + 0.5 * metrics.get('hit@5', 0.0)
        strategy_desc = "默认策略"
    
    return score, strategy_desc


def run_training(dataset, params, output_dir, use_multi_gpu=True, num_gpus=5):
    """
    为指定数据集运行训练和测试
    
    Args:
        dataset: 数据集名称
        params: 超参数字典
        output_dir: 输出目录
        use_multi_gpu: 是否使用多GPU训练（默认True）
        num_gpus: 使用的GPU数量（默认5）
    """
    print(f"\n{'='*60}")
    print(f"数据集: {dataset}")
    print(f"超参数: {params}")
    print(f"训练模式: {'多GPU' if use_multi_gpu else '单GPU'}")
    if use_multi_gpu:
        print(f"GPU数量: {num_gpus}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/train.log"
    
    # FP16参数
    fp16_flag = "--fp16" if params.get('fp16', False) else ""
    
    # 获取随机端口用于分布式训练
    import socket
    s = socket.socket()
    s.bind(("", 0))
    master_port = s.getsockname()[1]
    s.close()
    
    # 构建训练命令
    if use_multi_gpu:
        # 多GPU训练：使用torchrun
        # 强制设置可见设备，确保 torchrun 可以访问所有 GPU
        # 注意：这里假设我们想用机器上的所有 GPU (0-4)
        gpu_indices = ",".join([str(i) for i in range(num_gpus)])
        
    train_cmd = f"""
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES={gpu_indices}

torchrun --nproc_per_node={num_gpus} --master_port={master_port} finetune.py \\
    --index_version v3 \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --output_dir {output_dir} \\
    --load_model_name ./log/Pet,Cell,Automotive,Tools,Toys,Sports/ckpt_b1024_lr1e-3_seqrec,seqimage/pretrain_v3/checkpoint-26100 \\
    --per_device_batch_size {params['per_device_batch_size']} \\
    --learning_rate {params['learning_rate']:.6f} \\
    --epochs 200 \\
    --warmup_ratio {params['warmup_ratio']:.3f} \\
    --weight_decay {params['weight_decay']:.6f} \\
    --save_and_eval_strategy epoch \\
    --logging_step 50 \\
    --max_his_len 20 \\
    --prompt_num {params['prompt_num']} \\
    --patient 10 \\
    --index_file .index_lemb_{{version}}.json \\
    --image_index_file .index_vitemb_{{version}}.json \\
    --tasks seqrec,seqimage,item2image,image2item,fusionseqrec \\
    --valid_task seqrec {fp16_flag} > {log_file} 2>&1
"""
    else:
        # 单GPU训练：直接调用python（向后兼容）
        # 自动分配 GPU：根据 trial_id (sequence_id) 进行轮询
        try:
            seq_id = int(os.environ.get('NNI_TRIAL_SEQ_ID', '0'))
            gpu_id = seq_id % num_gpus
        except:
            import random
            gpu_id = random.randint(0, num_gpus - 1)
            
        print(f"自动分配 GPU: {gpu_id} (Seq ID: {os.environ.get('NNI_TRIAL_SEQ_ID', 'N/A')})")
        
        train_cmd = f"""
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES={gpu_id}

python finetune.py \\
    --index_version v3 \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --output_dir {output_dir} \\
    --load_model_name ./log/Pet,Cell,Automotive,Tools,Toys,Sports/ckpt_b1024_lr1e-3_seqrec,seqimage/pretrain_v3/checkpoint-26100 \\
    --per_device_batch_size {params['per_device_batch_size']} \\
    --learning_rate {params['learning_rate']:.6f} \\
    --epochs 200 \\
    --warmup_ratio {params['warmup_ratio']:.3f} \\
    --weight_decay {params['weight_decay']:.6f} \\
    --save_and_eval_strategy epoch \\
    --logging_step 50 \\
    --max_his_len 20 \\
    --prompt_num {params['prompt_num']} \\
    --patient 10 \\
    --index_file .index_lemb_{{version}}.json \\
    --image_index_file .index_vitemb_{{version}}.json \\
    --tasks seqrec,seqimage,item2image,image2item,fusionseqrec \\
    --valid_task seqrec {fp16_flag} > {log_file} 2>&1
"""
    
    print("开始训练...")
    result = subprocess.run(train_cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"警告：训练过程返回非零退出码: {result.returncode}")
    
    # 运行测试
    results_file = f"{output_dir}/results_seqrec_20.json"
    test_log = f"{output_dir}/test_seqrec.log"
    
    # 获取新的随机端口用于测试阶段
    s = socket.socket()
    s.bind(("", 0))
    test_port = s.getsockname()[1]
    s.close()

    if use_multi_gpu:
        # 多GPU测试：使用torchrun
        gpu_indices = ",".join([str(i) for i in range(num_gpus)])
        test_cmd = f"""
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES={gpu_indices}

torchrun --nproc_per_node={num_gpus} --master_port={test_port} test_ddp_save.py \\
    --index_version v3 \\
    --ckpt_path {output_dir} \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --test_batch_size 128 \\
    --num_beams 20 \\
    --index_file .index_lemb_{{version}}.json \\
    --image_index_file .index_vitemb_{{version}}.json \\
    --test_task seqrec \\
    --results_file {results_file} \\
    --save_file {output_dir}/save_seqrec_20.json \\
    --filter_items > {test_log} 2>&1
"""
    else:
        # 单GPU测试：使用伪分布式环境
        # 自动分配 GPU：根据 trial_id (sequence_id) 进行轮询
        try:
            seq_id = int(os.environ.get('NNI_TRIAL_SEQ_ID', '0'))
            gpu_id = seq_id % num_gpus
        except:
            import random
            gpu_id = random.randint(0, num_gpus - 1)

    test_cmd = f"""
export CUDA_LAUNCH_BLOCKING=1
export MASTER_ADDR=localhost
export MASTER_PORT={test_port}
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES={gpu_id}

python test_ddp_save.py \\
    --index_version v3 \\
    --ckpt_path {output_dir} \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --test_batch_size 128 \\
    --num_beams 20 \\
    --index_file .index_lemb_{{version}}.json \\
    --image_index_file .index_vitemb_{{version}}.json \\
    --test_task seqrec \\
    --results_file {results_file} \\
    --save_file {output_dir}/save_seqrec_20.json \\
    --filter_items > {test_log} 2>&1
"""
    
    print("开始测试...")
    subprocess.run(test_cmd, shell=True, executable='/bin/bash')
    
    # 提取指标
    metrics = None
    
    if os.path.exists(results_file):
        metrics = extract_metrics_from_results(results_file)
    
    if not metrics and os.path.exists(test_log):
        metrics = extract_metrics_from_log(test_log)
    
    if not metrics and os.path.exists(log_file):
        metrics = extract_metrics_from_log(log_file)
    
    if not metrics:
        print(f"错误：无法从 {dataset} 提取评估指标")
        metrics = {
            'hit@1': 0.0, 'hit@5': 0.0, 'hit@10': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0
        }
    
    print(f"测试指标: {metrics}")
    return metrics


def main():
    # 从 NNI 获取超参数
    params = nni.get_next_parameter()
    print(f"收到 NNI 超参数: {params}")
    
    # 设置默认参数（如果参数为空）
    if not params:
        print("警告：未收到参数，使用默认值")
        params = {
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'warmup_ratio': 0.3,
            'per_device_batch_size': 128
        }
    
    # 固定参数（不参与调参）
    params['prompt_num'] = 4
    
    # 确保 per_device_batch_size 存在（如果 NNI 没有传入）
    if 'per_device_batch_size' not in params:
        params['per_device_batch_size'] = 128
    
    # 从环境变量获取数据集名称和训练配置
    dataset = os.environ.get('NNI_DATASET', 'Instruments')
    optimization_strategy = os.environ.get('NNI_OPTIMIZATION_STRATEGY', 'hit1_focused')
    
    # 多GPU训练配置（从环境变量读取，默认启用多GPU训练）
    use_multi_gpu = os.environ.get('NNI_USE_MULTI_GPU', 'true').lower() == 'true'
    num_gpus = int(os.environ.get('NNI_NUM_GPUS', '5'))
    
    # 保存参数到文件
    trial_id = os.environ.get('NNI_TRIAL_JOB_ID', 'unknown')
    output_dir = f"./log/nni_trials_{dataset}/trial_{trial_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    # 运行训练和测试
    try:
        metrics = run_training(dataset, params, output_dir, use_multi_gpu=use_multi_gpu, num_gpus=num_gpus)
    except Exception as e:
        print(f"错误：训练过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        metrics = {
            'hit@1': 0.0, 'hit@5': 0.0, 'hit@10': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0
        }
    
    # 计算最终得分
    final_score, strategy_desc = calculate_score(metrics, optimization_strategy)
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset}")
    print(f"优化策略: {strategy_desc}")
    print(f"\n指标:")
    print(f"  Hit@1:    {metrics['hit@1']:.4f}")
    print(f"  Hit@5:    {metrics['hit@5']:.4f}")
    print(f"  Hit@10:   {metrics['hit@10']:.4f}")
    print(f"  NDCG@5:   {metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10:  {metrics['ndcg@10']:.4f}")
    print(f"\n最终优化得分: {final_score:.4f}")
    print(f"{'='*60}\n")
    
    # 报告结果给 NNI
    nni.report_final_result(final_score)
    
    # 保存详细结果
    final_results = {
        'params': params,
        'dataset': dataset,
        'optimization_strategy': optimization_strategy,
        'strategy_description': strategy_desc,
        'metrics': metrics,
        'final_score': final_score
    }
    
    with open(f"{output_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"试验失败: {e}")
        import traceback
        traceback.print_exc()
        nni.report_final_result(0.0)
        sys.exit(1)

