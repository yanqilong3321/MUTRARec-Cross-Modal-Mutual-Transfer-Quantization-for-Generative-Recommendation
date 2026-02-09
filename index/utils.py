
import datetime
import os
import sys


def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    """
    给文本添加颜色（仅在终端显示时生效）
    
    Args:
        log: 要着色的文本
        color: 颜色名称
        highlight: 是否使用粗体
    
    Returns:
        str: 带颜色代码的文本（终端）或原始文本（文件）
    """
    # 自动检测：输出到文件时禁用颜色，避免ANSI转义序列
    if not sys.stdout.isatty():
        return log
    
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur



