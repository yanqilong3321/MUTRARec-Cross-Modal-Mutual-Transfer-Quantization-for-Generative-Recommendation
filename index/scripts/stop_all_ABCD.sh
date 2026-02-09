#!/bin/bash

# 停止所有A、B、C、D版本的训练进程
# 通过匹配命令行参数来找到实际的Python训练进程

echo "================================"
echo "停止所有训练进程..."
echo "================================"

# 函数：通过ckpt_dir查找并终止进程
kill_by_ckpt_dir() {
    local version=$1
    local ckpt_dir=$2
    
    # 查找匹配的Python进程
    PIDS=$(ps -ef | grep "python.*main_fusion_v3.py" | grep "$ckpt_dir" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "版本$version: 未找到运行中的训练进程 (ckpt_dir=$ckpt_dir)"
    else
        for PID in $PIDS; do
            echo "终止版本$version进程 (PID: $PID)..."
            kill $PID
        done
        echo "版本$version: 已发送终止信号"
    fi
}

# 终止各版本进程
kill_by_ckpt_dir "A" "log/fusion_v3_A"
kill_by_ckpt_dir "B" "log/fusion_v3_B"
kill_by_ckpt_dir "C" "log/fusion_v3_C"
kill_by_ckpt_dir "D" "log/fusion_v3_D"

echo ""
echo "================================"
echo "完成！等待2秒后检查进程状态..."
echo "================================"

sleep 2

# 检查是否还有残留进程
REMAINING=$(ps -ef | grep "python.*main_fusion_v3.py" | grep -E "fusion_v3_[ABCD]" | grep -v grep)

if [ -z "$REMAINING" ]; then
    echo "✓ 所有训练进程已成功终止"
else
    echo "⚠ 以下进程仍在运行："
    echo "$REMAINING"
    echo ""
    echo "如需强制终止，请使用："
    echo "  kill -9 \$(ps -ef | grep 'python.*main_fusion_v3.py' | grep -E 'fusion_v3_[ABCD]' | grep -v grep | awk '{print \$2}')"
fi

echo "================================"

