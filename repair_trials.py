#!/usr/bin/env python3
"""
修复 Arts 数据集中有 checkpoint 但无结果的 trials
"""

import os
import sys
import json
import subprocess
import glob
from pathlib import Path

# 配置
DATASET = "Arts"
BASE_DIR = f"./log/nni_trials_{DATASET}"
DATA_PATH = "./data/"
INDEX_FILE = ".index_lemb_{version}.json"
IMAGE_INDEX_FILE = ".index_vitemb_{version}.json"
INDEX_VERSION = "v3"
NUM_GPUS = 5

def get_running_trials():
    """获取正在运行的 trials 列表，避免冲突"""
    try:
        cmd = "curl -s http://localhost:8081/api/v1/nni/trial-jobs"
        result = subprocess.check_output(cmd, shell=True)
        trials = json.loads(result)
        running = [t['id'] for t in trials if t['status'] in ['RUNNING', 'WAITING']]
        # 还要获取 trial 目录名
        # NNI API 返回的是 trial ID (如 gR53E)，目录名通常是 trial_ID
        return [f"trial_{tid}" for tid in running]
    except:
        print("无法连接 NNI Manager，假设没有正在运行的任务")
        return []

def scan_trials_to_repair(running_trials):
    to_repair = []
    for trial_dir in glob.glob(os.path.join(BASE_DIR, "trial_*")):
        trial_name = os.path.basename(trial_dir)
        
        # 跳过正在运行的
        if trial_name in running_trials:
            continue
            
        # 检查是否有 checkpoint
        checkpoints = glob.glob(os.path.join(trial_dir, "checkpoint-*"))
        if not checkpoints:
            continue
        
        # 获取最新的 checkpoint
        # 排序规则：按数字后缀
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_ckpt = checkpoints[-1]
        
        # 检查是否有有效结果
        result_file = os.path.join(trial_dir, "results_seqrec_20.json")
        final_result_file = os.path.join(trial_dir, "final_results.json")
        
        needs_repair = False
        
        if not os.path.exists(result_file):
            needs_repair = True
        elif os.path.exists(final_result_file):
            try:
                with open(final_result_file) as f:
                    data = json.load(f)
                    if data.get('final_score', 0) == 0:
                        needs_repair = True
            except:
                needs_repair = True
        
        if needs_repair:
            to_repair.append({
                "dir": trial_dir,
                "ckpt": latest_ckpt,
                "name": trial_name
            })
            
    return to_repair

def repair_trial(trial_info):
    print(f"\n正在修复: {trial_info['name']} (Checkpoint: {os.path.basename(trial_info['ckpt'])})")
    
    trial_dir = trial_info['dir']
    ckpt_path = trial_info['ckpt']
    
    results_file = os.path.join(trial_dir, "results_seqrec_20.json")
    save_file = os.path.join(trial_dir, "save_seqrec_20.json")
    log_file = os.path.join(trial_dir, "repair_test.log")
    
    # 获取随机端口
    import socket
    s = socket.socket()
    s.bind(("", 0))
    master_port = s.getsockname()[1]
    s.close()
    
    # 构建测试命令
    cmd = f"""
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

torchrun --nproc_per_node={NUM_GPUS} --master_port={master_port} test_ddp_save.py \\
    --index_version {INDEX_VERSION} \\
    --ckpt_path {ckpt_path} \\
    --data_path {DATA_PATH} \\
    --dataset {DATASET} \\
    --test_batch_size 128 \\
    --num_beams 20 \\
    --index_file {INDEX_FILE} \\
    --image_index_file {IMAGE_INDEX_FILE} \\
    --test_task seqrec \\
    --results_file {results_file} \\
    --save_file {save_file} \\
    --filter_items > {log_file} 2>&1
"""
    
    print("  执行测试中...")
    subprocess.run(cmd, shell=True)
    
    # 检查结果
    if os.path.exists(results_file):
        print("  测试完成，正在更新 final_results.json")
        update_final_results(trial_dir, results_file)
    else:
        print("  ❌ 测试失败，请查看日志: " + log_file)

def update_final_results(trial_dir, results_file):
    final_file = os.path.join(trial_dir, "final_results.json")
    
    if not os.path.exists(final_file):
        print("  警告: final_results.json 不存在，跳过更新")
        return

    try:
        # 读取测试结果
        with open(results_file) as f:
            res_data = json.load(f)
            # 处理嵌套结构 (mean_results)
            if 'mean_results' in res_data:
                metrics_data = res_data['mean_results']
            else:
                metrics_data = res_data
        
        # 读取原始 final_results
        with open(final_file) as f:
            final_data = json.load(f)
        
        # 更新指标
        metrics = {
            'hit@1': metrics_data.get('hit@1', 0),
            'hit@5': metrics_data.get('hit@5', 0),
            'hit@10': metrics_data.get('hit@10', 0),
            'ndcg@5': metrics_data.get('ndcg@5', 0),
            'ndcg@10': metrics_data.get('ndcg@10', 0)
        }
        
        # 计算分数 (hit1_focused)
        score = 0.4 * metrics['hit@1'] + 0.3 * metrics['ndcg@5'] + 0.3 * metrics['hit@5']
        
        final_data['metrics'] = metrics
        final_data['final_score'] = score
        
        # 写回
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"  ✅ 修复成功! Score: {score:.4f}")
        
    except Exception as e:
        print(f"  ❌ 更新结果失败: {e}")

def main():
    print("开始扫描待修复 trial...")
    running = get_running_trials()
    print(f"跳过正在运行的 trials: {running}")
    
    targets = scan_trials_to_repair(running)
    print(f"找到 {len(targets)} 个待修复 trial")
    
    for t in targets:
        repair_trial(t)
        
    print("\n所有修复任务完成。")

if __name__ == "__main__":
    main()
