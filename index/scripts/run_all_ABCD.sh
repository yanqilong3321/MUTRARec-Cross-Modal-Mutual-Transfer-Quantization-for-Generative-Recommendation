#!/bin/bash

# 并行运行A、B、C、D四个版本的训练脚本
# A -> GPU 1, B -> GPU 2, C -> GPU 3, D -> GPU 4
# 超参数调整请直接修改各自的训练脚本（A.sh、B.sh、C.sh、D.sh）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "================================"
echo "开始并行运行A、B、C、D四个版本"
echo "A -> GPU 1"
echo "B -> GPU 2"
echo "C -> GPU 3"
echo "D -> GPU 4"
echo "================================"

# 创建log目录
mkdir -p log/fusion_v3_A log/fusion_v3_B log/fusion_v3_C log/fusion_v3_D

# 函数：通过ckpt_dir获取Python进程PID
get_python_pid() {
    local ckpt_dir=$1
    local max_wait=10
    local count=0
    
    while [ $count -lt $max_wait ]; do
        local pid=$(ps -ef | grep "python.*main_fusion_v3.py" | grep "$ckpt_dir" | grep -v grep | awk '{print $2}' | head -1)
        if [ -n "$pid" ]; then
            echo "$pid"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    echo ""
    return 1
}

# 启动版本A
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动版本A (GPU 1)..."
echo "  读取超参数配置: scripts/A.sh"
nohup bash $SCRIPT_DIR/A.sh > /dev/null 2>&1 &
sleep 2
PID_A=$(get_python_pid "log/fusion_v3_A")
if [ -n "$PID_A" ]; then
    echo "✓ 版本A已启动，PID: $PID_A"
else
    echo "✗ 版本A启动失败或PID获取失败"
fi

# 启动版本B
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动版本B (GPU 2)..."
echo "  读取超参数配置: scripts/B.sh"
nohup bash $SCRIPT_DIR/B.sh > /dev/null 2>&1 &
sleep 2
PID_B=$(get_python_pid "log/fusion_v3_B")
if [ -n "$PID_B" ]; then
    echo "✓ 版本B已启动，PID: $PID_B"
else
    echo "✗ 版本B启动失败或PID获取失败"
fi

# 启动版本C
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动版本C (GPU 3)..."
echo "  读取超参数配置: scripts/C.sh"
nohup bash $SCRIPT_DIR/C.sh > /dev/null 2>&1 &
sleep 2
PID_C=$(get_python_pid "log/fusion_v3_C")
if [ -n "$PID_C" ]; then
    echo "✓ 版本C已启动，PID: $PID_C"
else
    echo "✗ 版本C启动失败或PID获取失败"
fi

# 启动版本D
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动版本D (GPU 4)..."
echo "  读取超参数配置: scripts/D.sh"
nohup bash $SCRIPT_DIR/D.sh > /dev/null 2>&1 &
sleep 2
PID_D=$(get_python_pid "log/fusion_v3_D")
if [ -n "$PID_D" ]; then
    echo "✓ 版本D已启动，PID: $PID_D"
else
    echo "✗ 版本D启动失败或PID获取失败"
fi

# 等待所有进程确认
sleep 1

# 汇总结果
echo ""
echo "================================"
echo "所有版本启动完成！"
echo "================================"

# 将PID保存到文件，方便后续管理
[ -n "$PID_A" ] && echo "$PID_A" > log/pid_A.txt
[ -n "$PID_B" ] && echo "$PID_B" > log/pid_B.txt
[ -n "$PID_C" ] && echo "$PID_C" > log/pid_C.txt
[ -n "$PID_D" ] && echo "$PID_D" > log/pid_D.txt

# 显示进程信息
echo ""
echo "进程信息："
[ -n "$PID_A" ] && echo "  版本A: PID=$PID_A, GPU=1, 日志=log/fusion_v3_A/train_v3_A.log"
[ -n "$PID_B" ] && echo "  版本B: PID=$PID_B, GPU=2, 日志=log/fusion_v3_B/train_v3_B.log"
[ -n "$PID_C" ] && echo "  版本C: PID=$PID_C, GPU=3, 日志=log/fusion_v3_C/train_v3_C.log"
[ -n "$PID_D" ] && echo "  版本D: PID=$PID_D, GPU=4, 日志=log/fusion_v3_D/train_v3_D.log"

echo ""
echo "超参数配置（在各脚本中定义）："
echo "  版本A: 查看 scripts/A.sh"
echo "  版本B: 查看 scripts/B.sh"
echo "  版本C: 查看 scripts/C.sh (推荐配置)"
echo "  版本D: 查看 scripts/D.sh"

echo ""
echo "查看日志命令："
echo "  tail -f log/fusion_v3_A/train_v3_A.log"
echo "  tail -f log/fusion_v3_B/train_v3_B.log"
echo "  tail -f log/fusion_v3_C/train_v3_C.log"
echo "  tail -f log/fusion_v3_D/train_v3_D.log"

echo ""
echo "查看进程状态："
if [ -n "$PID_A" ] && [ -n "$PID_B" ] && [ -n "$PID_C" ] && [ -n "$PID_D" ]; then
    echo "  ps -p $PID_A,$PID_B,$PID_C,$PID_D"
else
    echo "  ps -ef | grep 'python.*main_fusion_v3.py' | grep 'fusion_v3_[ABCD]'"
fi

echo ""
echo "终止所有进程："
echo "  bash $SCRIPT_DIR/stop_all_ABCD.sh"

echo ""
echo "================================"
echo "✨ 提示：调整超参数请直接修改 A.sh、B.sh、C.sh、D.sh"
echo "================================"
