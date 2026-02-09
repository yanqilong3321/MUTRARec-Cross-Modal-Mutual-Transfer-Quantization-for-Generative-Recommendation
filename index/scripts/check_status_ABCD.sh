#!/bin/bash

# 查看A、B、C、D四个版本的运行状态

echo "================================"
echo "训练进程状态检查"
echo "================================"

# 检查PID文件是否存在
if [ ! -f log/pid_A.txt ] || [ ! -f log/pid_B.txt ] || [ ! -f log/pid_C.txt ] || [ ! -f log/pid_D.txt ]; then
    echo "警告: 部分或全部PID文件不存在，可能尚未启动训练"
    echo ""
fi

# 检查各版本进程状态
check_process() {
    local version=$1
    local pid_file="log/pid_${version}.txt"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "✓ 版本${version}: 运行中 (PID: $pid)"
        else
            echo "✗ 版本${version}: 已停止 (PID: $pid)"
        fi
    else
        echo "? 版本${version}: 未找到PID文件"
    fi
}

check_process "A"
check_process "B"
check_process "C"
check_process "D"

echo ""
echo "================================"
echo "日志文件最后更新时间"
echo "================================"

for ver in A B C D; do
    log_file="log/fusion_v3_${ver}/train_v3_${ver}.log"
    if [ -f "$log_file" ]; then
        last_modified=$(stat -c "%y" "$log_file" 2>/dev/null || stat -f "%Sm" "$log_file" 2>/dev/null)
        file_size=$(du -h "$log_file" | cut -f1)
        echo "版本${ver}: ${last_modified} (大小: ${file_size})"
    else
        echo "版本${ver}: 日志文件不存在"
    fi
done

echo ""
echo "================================"
echo "快捷命令"
echo "================================"
echo "查看实时日志:"
echo "  tail -f log/fusion_v3_A/train_v3_A.log"
echo "  tail -f log/fusion_v3_B/train_v3_B.log"
echo "  tail -f log/fusion_v3_C/train_v3_C.log"
echo "  tail -f log/fusion_v3_D/train_v3_D.log"
echo ""
echo "GPU使用情况:"
echo "  nvidia-smi"
echo "  watch -n 1 nvidia-smi"

