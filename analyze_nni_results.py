#!/usr/bin/env python3
"""分析 NNI 调参结果"""
import json
import os
from pathlib import Path
import pandas as pd

def analyze_nni_trials(trials_dir):
    """分析 NNI 试验结果"""
    results = []
    
    trials_path = Path(trials_dir)
    if not trials_path.exists():
        print(f"错误：目录不存在 {trials_dir}")
        return None
    
    trial_dirs = sorted([d for d in trials_path.iterdir() if d.is_dir() and d.name.startswith('trial_')])
    
    print(f"找到 {len(trial_dirs)} 个试验目录")
    print("=" * 80)
    
    for trial_dir in trial_dirs:
        trial_id = trial_dir.name
        
        # 读取参数
        params_file = trial_dir / 'params.json'
        final_results_file = trial_dir / 'final_results.json'
        results_seqrec_file = trial_dir / 'results_seqrec_20.json'
        
        if not params_file.exists():
            print(f"⚠️  {trial_id}: 缺少 params.json")
            continue
            
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # 读取最终结果
        metrics = {}
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                final_results = json.load(f)
                metrics.update(final_results)
        elif results_seqrec_file.exists():
            with open(results_seqrec_file, 'r') as f:
                results_seqrec = json.load(f)
                metrics.update(results_seqrec)
        else:
            print(f"⚠️  {trial_id}: 缺少结果文件")
            continue
        
        # 合并结果
        result = {
            'trial_id': trial_id,
            **params,
            **metrics
        }
        results.append(result)
    
    if not results:
        print("没有找到有效的试验结果")
        return None
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 识别指标列（通常包含 recall, ndcg, hit 等）
    metric_cols = [col for col in df.columns if any(x in col.lower() for x in ['recall', 'ndcg', 'hit', 'mrr', 'precision'])]
    param_cols = [col for col in df.columns if col not in metric_cols and col != 'trial_id']
    
    print(f"\n📊 共有 {len(df)} 个有效试验")
    print(f"参数列: {param_cols}")
    print(f"指标列: {metric_cols}")
    print("=" * 80)
    
    # 统计每个指标的最佳值
    print("\n🏆 最佳指标统计:\n")
    for metric in sorted(metric_cols):
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_value = df.loc[best_idx, metric]
            best_trial = df.loc[best_idx, 'trial_id']
            
            print(f"{metric:30s}: {best_value:.6f}  (Trial: {best_trial})")
    
    # 找出综合表现最好的试验（基于主要指标）
    main_metrics = [col for col in metric_cols if 'recall@20' in col.lower() or 'ndcg@20' in col.lower()]
    
    if main_metrics:
        print("\n" + "=" * 80)
        print(f"\n🎯 基于主要指标 {main_metrics} 的 Top 5 试验:\n")
        
        # 计算平均排名
        df['avg_rank'] = 0
        for metric in main_metrics:
            if metric in df.columns:
                df['avg_rank'] += df[metric].rank(ascending=False)
        df['avg_rank'] /= len(main_metrics)
        
        top5 = df.nsmallest(5, 'avg_rank')
        
        for idx, row in top5.iterrows():
            print(f"\n{row['trial_id']}:")
            print("  参数:")
            for param in param_cols:
                if param in row:
                    print(f"    {param:25s}: {row[param]}")
            print("  指标:")
            for metric in sorted(metric_cols):
                if metric in row:
                    print(f"    {metric:25s}: {row[metric]:.6f}")
    
    # 保存详细结果
    output_file = trials_path.parent / f'{trials_path.name}_analysis.csv'
    cols_order = ['trial_id'] + param_cols + sorted(metric_cols)
    df[cols_order].to_csv(output_file, index=False)
    print(f"\n\n💾 详细结果已保存到: {output_file}")
    
    return df

if __name__ == '__main__':
    trials_dir = '/data/yql/workspace/MQL4GRec_v1/log/nni_trials_Arts'
    df = analyze_nni_trials(trials_dir)
