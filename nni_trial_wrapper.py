#!/usr/bin/env python3
"""
NNI Trial Wrapper - 用于包装 finetune_v3.sh 脚本
接收 NNI 传递的超参数，运行训练和测试，并报告结果给 NNI
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
        # 示例: "ndcg@10: 0.1234" 或 "ndcg@10=0.1234"
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


def run_training_for_dataset(dataset, params, base_output_dir):
    """
    为单个数据集运行训练和测试
    """
    print(f"\n{'='*50}")
    print(f"处理数据集: {dataset}")
    print(f"超参数: {params}")
    print(f"{'='*50}\n")
    
    # 构建输出目录
    output_dir = f"{base_output_dir}/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/train.log"
    
    # 构建训练命令
    train_cmd = f"""
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=5 --master_port=2311 finetune.py \\
    --index_version v3 \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --output_dir {output_dir} \\
    --load_model_name ./log/Pet,Cell,Automotive,Tools,Toys,Sports/ckpt_b1024_lr1e-3_seqrec,seqimage/pretrain_v3/checkpoint-26100 \\
    --per_device_batch_size 2048 \\
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
    --valid_task seqrec > {log_file} 2>&1
"""
    
    # 运行训练
    print("开始训练...")
    result = subprocess.run(train_cmd, shell=True, executable='/bin/bash')
    if result.returncode != 0:
        print(f"警告：训练过程返回非零退出码: {result.returncode}")
    
    # 运行 seqrec 测试
    results_file = f"{output_dir}/results_seqrec_20.json"
    test_seqrec_log = f"{output_dir}/test_seqrec.log"
    
    test_cmd = f"""
export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=5 --master_port=2311 test_ddp_save.py \\
    --index_version v3 \\
    --ckpt_path {output_dir} \\
    --data_path ./data/ \\
    --dataset {dataset} \\
    --test_batch_size 64 \\
    --num_beams 20 \\
    --index_file .index_lemb_{{version}}.json \\
    --image_index_file .index_vitemb_{{version}}.json \\
    --test_task seqrec \\
    --results_file {results_file} \\
    --save_file {output_dir}/save_seqrec_20.json \\
    --filter_items > {test_seqrec_log} 2>&1
"""
    
    print("开始测试 seqrec...")
    subprocess.run(test_cmd, shell=True, executable='/bin/bash')
    
    # 提取指标
    metrics = None
    
    # 首先尝试从 JSON 结果文件提取
    if os.path.exists(results_file):
        metrics = extract_metrics_from_results(results_file)
    
    # 如果失败，尝试从日志文件提取
    if not metrics and os.path.exists(test_seqrec_log):
        metrics = extract_metrics_from_log(test_seqrec_log)
    
    # 如果还是失败，尝试从训练日志提取（可能包含验证集指标）
    if not metrics and os.path.exists(log_file):
        metrics = extract_metrics_from_log(log_file)
    
    if not metrics:
        print(f"错误：无法从 {dataset} 提取评估指标")
        # 返回最低分，让 NNI 知道这次试验失败
        metrics = {
            'hit@1': 0.0, 'hit@5': 0.0, 'hit@10': 0.0,
            'ndcg@5': 0.0, 'ndcg@10': 0.0
        }
    
    print(f"数据集 {dataset} 的指标: {metrics}")
    return metrics


def calculate_final_score(all_metrics, strategy='ndcg5_hit5'):
    """
    根据不同策略计算最终优化得分
    
    支持的策略:
    - 'ndcg5_hit5': NDCG@5 和 Hit@5 加权组合 (0.5×ndcg@5 + 0.5×hit@5) [默认]
    - 'ndcg10_only': 只优化 ndcg@10 (适合注重排序质量)
    - 'balanced': 平衡 ndcg@10 和 hit@10 (0.6×ndcg@10 + 0.4×hit@10)
    - 'comprehensive': 综合多个指标 (0.4×ndcg@10 + 0.2×ndcg@5 + 0.3×hit@10 + 0.1×hit@5)
    - 'hit_focused': 注重召回率 (0.4×ndcg@10 + 0.6×hit@10)
    """
    
    # 计算各指标的平均值（跨数据集）
    avg_metrics = {}
    metric_keys = ['hit@1', 'hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']
    
    for key in metric_keys:
        values = [m.get(key, 0.0) for m in all_metrics]
        avg_metrics[key] = sum(values) / len(values) if values else 0.0
    
    # 根据策略计算最终得分
    if strategy == 'ndcg5_hit5':
        # 用户指定：NDCG@5 和 Hit@5 相等权重
        final_score = 0.5 * avg_metrics['ndcg@5'] + 0.5 * avg_metrics['hit@5']
        strategy_desc = "NDCG@5 & Hit@5 均衡策略 (0.5×NDCG@5 + 0.5×Hit@5)"
        
    elif strategy == 'ndcg10_only':
        final_score = avg_metrics['ndcg@10']
        strategy_desc = "单一优化 NDCG@10"
        
    elif strategy == 'balanced':
        # 平衡排序质量(ndcg)和召回能力(hit)
        final_score = 0.6 * avg_metrics['ndcg@10'] + 0.4 * avg_metrics['hit@10']
        strategy_desc = "平衡策略 (0.6×NDCG@10 + 0.4×Hit@10)"
        
    elif strategy == 'comprehensive':
        # 综合考虑多个指标
        final_score = (0.4 * avg_metrics['ndcg@10'] +
                      0.2 * avg_metrics['ndcg@5'] +
                      0.3 * avg_metrics['hit@10'] +
                      0.1 * avg_metrics['hit@5'])
        strategy_desc = "综合策略 (0.4×NDCG@10 + 0.2×NDCG@5 + 0.3×Hit@10 + 0.1×Hit@5)"
        
    elif strategy == 'hit_focused':
        # 更注重召回率
        final_score = 0.4 * avg_metrics['ndcg@10'] + 0.6 * avg_metrics['hit@10']
        strategy_desc = "召回优先策略 (0.4×NDCG@10 + 0.6×Hit@10)"
    
    else:
        # 默认使用 ndcg5_hit5 策略
        final_score = 0.5 * avg_metrics['ndcg@5'] + 0.5 * avg_metrics['hit@5']
        strategy_desc = "NDCG@5 & Hit@5 均衡策略 (默认)"
    
    return final_score, avg_metrics, strategy_desc


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
            'warmup_ratio': 0.03,
            'prompt_num': 4
        }
    
    # 保存参数到文件，便于后续分析
    trial_id = os.environ.get('NNI_TRIAL_JOB_ID', 'unknown')
    base_output_dir = f"./log/nni_trials/trial_{trial_id}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    with open(f"{base_output_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    # 优化策略配置 - 可通过环境变量修改
    # 可选值: 'ndcg5_hit5', 'ndcg10_only', 'balanced', 'comprehensive', 'hit_focused'
    optimization_strategy = os.environ.get('NNI_OPTIMIZATION_STRATEGY', 'ndcg5_hit5')
    
    # 对三个数据集分别运行训练和测试
    datasets = ['Instruments', 'Arts', 'Games']
    all_metrics = []
    
    for dataset in datasets:
        try:
            metrics = run_training_for_dataset(dataset, params, base_output_dir)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"错误：处理数据集 {dataset} 时发生异常: {e}")
            all_metrics.append({
                'hit@1': 0.0, 'hit@5': 0.0, 'hit@10': 0.0,
                'ndcg@5': 0.0, 'ndcg@10': 0.0
            })
    
    # 计算最终得分
    final_score, avg_metrics, strategy_desc = calculate_final_score(
        all_metrics, optimization_strategy
    )
    
    print(f"\n{'='*60}")
    print(f"所有数据集完成！")
    print(f"\n优化策略: {strategy_desc}")
    print(f"\n平均指标 (跨 3 个数据集):")
    print(f"  Hit@1:    {avg_metrics['hit@1']:.4f}")
    print(f"  Hit@5:    {avg_metrics['hit@5']:.4f}")
    print(f"  Hit@10:   {avg_metrics['hit@10']:.4f}")
    print(f"  NDCG@5:   {avg_metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10:  {avg_metrics['ndcg@10']:.4f}")
    print(f"\n最终优化得分: {final_score:.4f}")
    print(f"{'='*60}\n")
    
    # 报告结果给 NNI（使用最终评分作为优化目标）
    nni.report_final_result(final_score)
    
    # 同时保存详细结果
    final_results = {
        'params': params,
        'optimization_strategy': optimization_strategy,
        'strategy_description': strategy_desc,
        'datasets': {
            dataset: metrics 
            for dataset, metrics in zip(datasets, all_metrics)
        },
        'avg_metrics': avg_metrics,
        'final_score': final_score
    }
    
    with open(f"{base_output_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"试验失败: {e}")
        import traceback
        traceback.print_exc()
        # 报告失败给 NNI
        nni.report_final_result(0.0)
        sys.exit(1)

