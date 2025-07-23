#!/bin/bash
set -e

# 检查pandoc是否已安装
if ! command -v pandoc &> /dev/null; then
    echo "[INFO] pandoc 未检测到，正在尝试自动安装..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y pandoc
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install pandoc
    else
        echo "[WARN] 未知系统，请手动安装pandoc！"
    fi
else
    echo "[INFO] pandoc 已安装。"
fi

# 激活虚拟环境（如存在）
if [ -d "../.venv" ]; then
    echo "[INFO] 激活虚拟环境 .venv"
    source ../.venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[INFO] 激活虚拟环境 .venv"
    source .venv/bin/activate
fi

# 强制使用uv安装依赖
if ! command -v uv &> /dev/null; then
    echo "[ERROR] 未检测到uv，请先用 'pip install uv' 安装！"
    exit 1
fi
uv sync

# 运行主流程
python app/run.py
