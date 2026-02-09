#!/usr/bin/env python3
"""
简单的NNI测试脚本 - 用于验证NNI基本功能
"""
import nni
import time

print("=" * 60)
print("NNI 简单测试脚本")
print("=" * 60)

try:
    # 获取参数
    params = nni.get_next_parameter()
    print(f"✓ 成功接收参数: {params}")
    
    # 模拟训练
    print("模拟训练中...")
    time.sleep(2)
    
    # 计算一个简单的分数
    score = params.get('learning_rate', 0.0001) * 1000
    print(f"计算得分: {score}")
    
    # 报告结果
    nni.report_final_result(score)
    print("✓ 成功报告结果")
    
    print("=" * 60)
    print("测试成功！")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    nni.report_final_result(0.0)

