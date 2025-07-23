FROM python:3.10-slim

# 安装pandoc
RUN apt-get update \
    && apt-get install -y pandoc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装Python依赖
RUN pip install --upgrade pip \
    && pip install uv \
    && uv pip install --system --no-cache-dir .

# 创建输出目录
RUN mkdir -p /app/reports

# 默认命令
CMD ["python", "app/run.py"]
