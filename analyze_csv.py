import csv
import json
import ast

def parse_metrics(metrics_str):
    try:
        # 尝试解析 JSON 格式
        return json.loads(metrics_str.replace("'", '"'))
    except:
        try:
            # 尝试解析 Python 字典格式
            return ast.literal_eval(metrics_str)
        except:
            return {}

results = []
csv_path = '/data/yql/workspace/MQL4GRec_v1/log/nni_trials_Arts_analysis.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metrics = parse_metrics(row['metrics'])
        # 提取指标
        ndcg10 = float(metrics.get('ndcg@10', 0))
        hit10 = float(metrics.get('hit@10', 0))
        hit1 = float(metrics.get('hit@1', 0))
        ndcg5 = float(metrics.get('ndcg@5', 0))
        
        # 提取参数
        params = {
            'learning_rate': float(row['learning_rate']),
            'weight_decay': float(row['weight_decay']),
            'warmup_ratio': float(row['warmup_ratio']),
            'per_device_batch_size': int(row['per_device_batch_size']),
            'fp16': row['fp16']
        }
        
        results.append({
            'trial_id': row['trial_id'],
            'ndcg@10': ndcg10,
            'hit@10': hit10,
            'hit@1': hit1,
            'ndcg@5': ndcg5,
            'params': params
        })

# 按 ndcg@10 排序
sorted_by_ndcg = sorted(results, key=lambda x: x['ndcg@10'], reverse=True)

print(f"Top 3 Trials by NDCG@10:")
for i, res in enumerate(sorted_by_ndcg[:3]):
    print(f"{i+1}. {res['trial_id']}: NDCG@10={res['ndcg@10']:.6f}, Hit@10={res['hit@10']:.6f}")
    print(f"   Params: {res['params']}")

print("\nBest Trial Details:")
best = sorted_by_ndcg[0]
print(f"Trial ID: {best['trial_id']}")
print(f"NDCG@10: {best['ndcg@10']:.6f}")
print(f"Hit@10: {best['hit@10']:.6f}")
print(f"Hit@1: {best['hit@1']:.6f}")
print(f"NDCG@5: {best['ndcg@5']:.6f}")
print("Parameters:")
for k, v in best['params'].items():
    print(f"  {k}: {v}")
