#!/usr/bin/env python3
"""
串行修复脚本 - 稳健版
"""

import os
import sys
import json
import subprocess
import glob
import time
import random
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
        return [f"trial_{tid}" for tid in running]
    except:
        return []

def scan_trials_to_repair(running_trials):
    to_repair = []
    print(f"正在扫描 {BASE_DIR} ...")
    
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
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_ckpt = checkpoints[-1]
        
        # 检查结果
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

def get_random_port():
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def update_final_results(trial_dir, results_file):
    final_file = os.path.join(trial_dir, "final_results.json")
    if not os.path.exists(final_file):
        return

    try:
        with open(results_file) as f:
            res_data = json.load(f)
            if 'mean_results' in res_data:
                metrics_data = res_data['mean_results']
            else:
                metrics_data = res_data
        
        with open(final_file) as f:
            final_data = json.load(f)
        
        metrics = {
            'hit@1': metrics_data.get('hit@1', 0),
            'hit@5': metrics_data.get('hit@5', 0),
            'hit@10': metrics_data.get('hit@10', 0),
            'ndcg@5': metrics_data.get('ndcg@5', 0),
            'ndcg@10': metrics_data.get('ndcg@10', 0)
        }
        
        # hit1_focused
        score = 0.4 * metrics['hit@1'] + 0.3 * metrics['ndcg@5'] + 0.3 * metrics['hit@5']
        
        final_data['metrics'] = metrics
        final_data['final_score'] = score
        
        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"  ✅ 结果已更新! Score: {score:.4f}")
        
    except Exception as e:
        print(f"  ❌ 更新结果文件失败: {e}")

def repair_trial(trial_info):
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始修复: {trial_info['name']}")
    print(f"  Checkpoint: {os.path.basename(trial_info['ckpt'])}")
    
    trial_dir = trial_info['dir']
    ckpt_path = trial_info['ckpt']
    results_file = os.path.join(trial_dir, "results_seqrec_20.json")
    save_file = os.path.join(trial_dir, "save_seqrec_20.json")
    log_file = os.path.join(trial_dir, "repair_serial.log")
    
    # 随机端口避免冲突
    master_port = get_random_port()
    
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
    
    print(f"  启动测试 (Port: {master_port})...")
    subprocess.run(cmd, shell=True)
    
    if os.path.exists(results_file):
        update_final_results(trial_dir, results_file)
    else:
        print(f"  ❌ 修复失败，日志: {log_file}")
    
    # 稍微等待，确保资源释放
    time.sleep(5)

def main():
    running = get_running_trials()
    print(f"正在运行的 trials (跳过): {running}")
    
    targets = scan_trials_to_repair(running)
    
    if not targets:
        print("没有需要修复的 trial。")
        return

    print(f"共发现 {len(targets)} 个待修复 trial。")
    print("开始串行修复...")
    
    for i, t in enumerate(targets):
        print(f"\n>>> 进度: {i+1}/{len(targets)}")
        repair_trial(t)
        
    print("\n所有修复任务完成。")

if __name__ == "__main__":
    main()
