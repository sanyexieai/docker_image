# 金融研报自动生成系统

本项目为智能体赋能的金融多模态报告自动化生成系统，支持公司、行业、宏观多类型研报自动生成。

## 主要特性
- 支持多种数据源自动采集与分析
- 支持多种大模型与RAG融合
- 支持自动生成Word/Markdown格式研报
- 支持Docker一键部署

## 依赖环境
- Python >=3.10,<3.11
- 推荐使用虚拟环境（venv/conda/poetry/uv）
- 依赖详见 `pyproject.toml`

### 关键依赖
- `pypandoc`：用于文档格式转换（md <-> docx 等）
- 其它依赖见 `pyproject.toml`

## pypandoc 安装说明

`pypandoc` 依赖外部的 `pandoc` 可执行文件，需**单独安装 pandoc**。

### 1. Windows
- 推荐直接下载安装包：[Pandoc Releases](https://github.com/jgm/pandoc/releases)
- 或用 scoop/choco：
  ```powershell
  scoop install pandoc
  # 或
  choco install pandoc
  ```
- 安装后请确保 `pandoc.exe` 在 PATH 中

### 2. macOS
- 推荐用 Homebrew：
  ```bash
  brew install pandoc
  ```
- 或官网下载 dmg 安装包

### 3. Linux (Ubuntu/Debian)
- 推荐：
  ```bash
  sudo apt-get update
  sudo apt-get install pandoc
  ```
- 或下载二进制包解压到 `/usr/local/bin`

### 4. Docker 环境
- Dockerfile 已自动安装 pandoc，无需手动操作。

## 快速开始

```bash
# 安装依赖
uv sync  # 或 pip install -r requirements.txt

# 运行主流程
python app/run.py
```

## Docker 一键部署

```bash
docker build -t ai-report .
docker run --rm -it -v $PWD/reports:/app/reports ai-report
```

## 目录结构
- `app/`         主程序代码
- `run_company_research_report.py`  公司研报主流程
- `run_industry_research_report.py` 行业研报主流程
- `run_marco_research_report.py`    宏观研报主流程
- `reports/`     生成的研报输出

## 常见问题
- **pypandoc报错找不到pandoc**：请参考上方各平台安装方法，确保pandoc可执行文件在PATH中。
- **CUDA/torch相关报错**：如无GPU可忽略，RAG相关功能会自动降级。

## 联系方式
如有问题请提交issue或联系作者。
