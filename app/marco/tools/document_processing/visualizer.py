# -*- coding: utf-8 -*-
"""
通用研报数据可视化插件 (AI-Powered Report Visualizer)

功能:
1. 调用大模型从任意研报文本中提取数据
2. 让大模型生成可视化图表代码
3. 执行生成的代码并保存图表
4. 支持多种图表类型和数据格式

作者: AI助手
版本: 1.1 (合并为单文件)
"""

import os
import re
import json
import tempfile
import importlib
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import openai
from typing import List, Dict, Any, Union, Optional
from io import StringIO

# 加载环境变量
load_dotenv()

class AIReportVisualizer:
    """AI驱动的研报数据可视化工具"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 model: str = "deepseek-chat",
                 output_dir: str = "report_visuals",
                 dataframe_dir: str = "report_dataframes"):
        """
        初始化AI研报可视化工具
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            base_url: API基础URL，如果为None则从环境变量获取
            model: 使用的大模型名称
            output_dir: 图表输出目录
            dataframe_dir: 提取的数据表格（DataFrame）的输出目录
        """
        # 设置API参数
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "deepseek-chat")
        
        # 初始化OpenAI客户端
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"成功初始化AI客户端，使用模型: {self.model}")
        except Exception as e:
            print(f"初始化AI客户端失败: {e}")
            print("将使用本地模式，不调用AI接口")
            self.client = None
        
        # 设置输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置DataFrame输出目录
        self.dataframe_dir = dataframe_dir
        os.makedirs(self.dataframe_dir, exist_ok=True)
        
        # 设置临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 确保matplotlib支持中文
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """
        设置matplotlib支持中文显示，自动查找可用字体。
        """
        import platform
        try:
            from matplotlib.font_manager import FontManager
        except ImportError:
            print("警告: 无法导入 matplotlib.font_manager，将使用默认字体。中文可能无法正常显示。")
            return

        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

        system = platform.system()
        try:
            fm = FontManager()
            # 定义常见的中文字体
            if system == 'Darwin':  # macOS
                font_names = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Arial Unicode MS']
            elif system == 'Windows':
                font_names = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
            else:  # Linux
                font_names = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Droid Sans Fallback']
            
            # 查找并设置字体
            for font_name in font_names:
                if font_name in fm.get_font_names():
                    plt.rcParams['font.sans-serif'] = [font_name]
                    print(f"成功设置中文字体为: {font_name}")
                    return
            
            print("警告: 未找到预设的中文字体，将尝试使用系统备用字体。")
            # 最后的尝试
            if 'sans-serif' not in plt.rcParams['font.sans-serif']:
                 plt.rcParams['font.sans-serif'] = ['sans-serif']

        except Exception as e:
            print(f"设置中文字体时出错: {e}。图表中文可能显示异常。")
            plt.rcParams['font.sans-serif'] = ['sans-serif']
    
    def call_llm(self, prompt: str) -> str:
        """
        调用大语言模型
        
        Args:
            prompt: 提示词
            
        Returns:
            str: 模型响应
        """
        if not self.client:
            print("警告: AI客户端未初始化，无法调用大模型")
            return "AI客户端未初始化，请检查API密钥设置"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # 降低温度以获得更确定性的回答
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"调用大模型时出错: {e}")
            return f"调用大模型失败: {e}"
    
    def extract_data_to_dataframe(self, text: str) -> pd.DataFrame:
        """
        使用大模型从文本中提取所有数据，并将其整合到一个Pandas DataFrame中。
        
        Args:
            text: 研报文本
            
        Returns:
            pd.DataFrame: 包含所有提取数据的DataFrame
        """
        prompt = f"""
作为一名专业的金融数据分析师，请仔细分析以下研报文本。您的任务是提取所有可用于可视化的结构化数据，并将其整理成一个统一的、符合逻辑的CSV格式。

**核心要求:**
1.  **统一格式:** 将所有提取的数据转换为一个单一的表格。使用以下列标题：`Metric`, `Entity`, `Year`, `Value`, `Unit`, `Source`, `Category`。
    *   `Metric`: 数据的具体指标名称，例如 "GDP增速", "核心CPI同比", "财政赤字率"。
    *   `Entity`: 数据所属的实体或分类，例如 "美国", "中国", "欧元区", "高科技产业"。
    *   `Year`: 数据对应的年份。如果是时间段，请使用 "YYYY-YYYY" 格式，如 "2023-2026"。对于单个年份，使用 "YYYY"。
    *   `Value`: 数值。请只保留数字，不要包含 "%" 或单位。
    *   `Unit`: 数据的单位，例如 "%", "亿美元", "个基点", "倍"。
    *   `Source`: 数据来源，如 "IMF", "世界银行"。如果原文未提及，请注明"来源未提及"。
    *   `Category`: 一个更高层级的分类，用于对指标进行分组，例如 "宏观经济", "产业投资", "金融市场"。
2.  **数据完整性:** 提取所有关键数据点，包括时间序列、分类比较、百分比构成等。
3.  **精确性:** 严格忠于原文的数据和逻辑。

**研报文本:**
---
{text[:10000]}
---

请以严格的CSV格式返回结果，使用逗号作为分隔符，并包含表头。不要返回任何额外的解释性文字。
"""
        response = self.call_llm(prompt)
        
        # 提取CSV部分
        csv_match = re.search(r'```csv\s*([\s\S]*?)\s*```', response)
        if csv_match:
            csv_str = csv_match.group(1)
        else:
            # 假设如果模型没有返回markdown块，它直接返回了CSV内容
            csv_str = response

        # 检查是否包含表头
        if not any(header in csv_str for header in ['Metric', 'Entity', 'Year', 'Value']):
            print("警告: 模型返回的CSV数据可能缺少表头，尝试添加默认表头。")
            print(f"原始响应: {csv_str}")
            return pd.DataFrame()

        try:
            # 使用StringIO将字符串读作文件，然后用pandas解析
            data = StringIO(csv_str)
            df = pd.read_csv(data)
            
            # 清理数据: Value列转换为数字
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                df.dropna(subset=['Value'], inplace=True)
            return df
        except Exception as e:
            print(f"从LLM响应解析CSV失败: {e}")
            print(f"原始响应:\n{response}")
            return pd.DataFrame()

    def generate_code_from_dataframe(self, df: pd.DataFrame) -> str:
        """
        根据DataFrame生成可视化代码。
        
        Args:
            df: 包含数据的Pandas DataFrame。
            
        Returns:
            str: 可视化代码
        """
        if df.empty:
            return ""
            
        dataframe_as_csv = df.to_csv(index=False)
        
        prompt = f"""
作为一名专业的数据可视化工程师，请分析以下Pandas DataFrame，并生成专业、美观、信息清晰的Python可视化代码，为数据中包含的有意义的洞察创建图表。

**数据 (以CSV格式表示):**
```
{dataframe_as_csv}
```

**核心编码要求:**
1.  **智能分组和绘图:** 分析DataFrame的内容，识别可以一起可视化的相关数据。例如，按`Metric`或`Category`对数据进行分组，为每个组创建一个图表。
2.  **选择合适的图表:** 根据每个数据组的性质（例如，时间序列数据使用折线图，分类比较数据使用柱状图）自动选择最合适的图表类型。
3.  **库使用:** 必须使用 `matplotlib` 和 `seaborn` 进行绘图。
4.  **独立运行:** 生成的代码必须是完整的、可独立运行的脚本，包含所有必要的 `import` 语句。
5.  **中文支持:** 代码已在支持中文的环境中运行，直接使用中文设置标题、标签等。
6.  **专业美学设计:**
    *   使用 `seaborn` 预设样式，例如 `sns.set_style("whitegrid")`。
    *   选择合适的调色板。
    *   在图表上清晰地标注数据值。
7.  **信息标注:**
    *   根据DataFrame中的`Metric`, `Entity`, `Unit`等信息，生成清晰的图表标题和坐标轴标签。
    *   如果`Source`列有信息，在图表下标注数据来源。
8.  **文件保存和输出:**
    *   **必须** 使用 `plt.savefig()` 保存图表，不要使用 `plt.show()`。
    *   图表应保存到`'./output'`目录中。文件名应根据图表内容生成（例如，`GDP增速对比.png`）。
    *   **关键: 每保存一个文件后，必须立即使用 `print()` 语句输出所保存图表的完整路径。**

请只返回一个完整的Python代码块，该代码块能够根据DataFrame创建并保存多个有意义的图表。
"""
        response = self.call_llm(prompt)
        
        # 提取代码部分
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        return code

    def extract_data_from_text(self, text: str, data_type: str = "all") -> Dict:
        """
        使用大模型从文本中提取数据
        
        Args:
            text: 研报文本
            data_type: 要提取的数据类型，可选值: "all", "numeric", "time_series", "comparison"
            
        Returns:
            Dict: 提取的数据
        """
        # 构建提示词
        prompt = f"""
作为一名专业的金融数据分析师，请仔细分析以下研报文本。您的任务是提取所有可用于可视化的结构化数据，并将其组织成高度精确和逻辑一致的JSON格式。

**核心要求:**
1.  **逻辑一致性:** 提取的数据和图表建议必须严格忠于原文的逻辑和意图。例如，如果文本在比较不同国家的GDP，图表的标题和坐标轴就应明确反映这一点。
2.  **数据完整性:** 提取所有关键数据点，包括时间序列、分类比较、百分比构成等。特别注意从文本、表格和列表中提取数据。
3.  **精确元数据:**
    *   `title`: 创建一个清晰、具体、信息丰富的图表标题，例如"2020-2023年主要经济体GDP年增长率对比"。
    *   `type`: 准确判断数据类型: `time_series` (时间序列), `comparison` (分类比较), `composition` (构成/占比), `correlation` (相关性)。
    *   `chart_type`: 根据数据类型和原文逻辑，推荐最合适的图表类型: `line` (折线图), `bar` (柱状图), `pie` (饼图), `scatter` (散点图), `area` (面积图), `table` (表格)。
    *   `x_label` 和 `y_label`: 提供精确的横轴和纵轴标签，必须包含单位。例如 `y_label: "GDP增长率 (%)"`。
    *   `unit`: 明确数据的核心单位，如"%"、"亿元"、"个基点"。
    *   `data`: 将数据整理成干净的格式，例如 `[{{"x": "2022", "y": 5.2}}, {{"x": "2023", "y": 2.5}}]`。
    *   `source`: **必须提取数据来源**，如"数据来源：国家统计局"。如果原文未提及，请注明"来源未提及"。
    *   `description`: 简要解释数据的核心观点或其在报告中的上下文。

**研报文本:**
---
{text[:10000]}  # 限制文本长度以避免超过token限制
---

请以有效的JSON数组格式返回结果，确保可以直接被Python的`json.loads()`函数解析。不要返回任何额外的解释性文字。
"""
        
        # 调用大模型
        response = self.call_llm(prompt)
        
        # 尝试解析JSON
        try:
            # 提取JSON部分(防止模型返回额外文本)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # 解析JSON
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response}")
            return {"error": "数据提取失败，无法解析JSON", "raw_response": response}
    
    def generate_visualization_code(self, data: Dict) -> str:
        """
        使用大模型生成可视化代码
        
        Args:
            data: 提取的数据
            
        Returns:
            str: 可视化代码
        """
        # 构建提示词
        prompt = f"""
作为一名专业的数据可视化工程师，请根据以下JSON格式的数据数组，为**每一组数据**生成专业、美观、信息清晰的Python可视化代码。

**数据详情 (一个包含多组数据的JSON数组):**
```json
{json.dumps(data, ensure_ascii=False, indent=2)}
```

**核心编码要求:**
1.  **循环处理:** 你必须遍历输入的数据数组，为数组中的**每个对象**生成一个独立的图表。
2.  **库使用:** 必须使用 `matplotlib` 和 `seaborn` 进行绘图。
3.  **独立运行:** 生成的代码必须是完整的、可独立运行的脚本，包含所有必要的 `import` 语句。
4.  **中文支持:** 代码已在支持中文的环境中运行，直接使用中文设置标题、标签等。
5.  **图表类型:** 严格按照每组数据中指定的 `chart_type` 生成图表。
6.  **专业美学设计 (对每个图表都适用):**
    *   使用 `seaborn` 预设样式，例如 `sns.set_style("whitegrid")`。
    *   选择合适的调色板（例如 `seaborn.color_palette`）以增强可读性。
    *   在图表的条形、点或线上方清晰地标注数据值。
    *   坐标轴刻度应清晰、合理，避免拥挤。
    *   如果包含图例，确保其位置不会遮挡任何数据。
7.  **信息标注 (对每个图表都适用):**
    *   **必须** 使用 `plt.title()` `plt.xlabel()` `plt.ylabel()` 设置清晰的图表标题和坐标轴标签。
    *   **必须** 在图表的左下角或右下角使用 `plt.figtext()` 标注数据来源 (`source` 字段)。
8.  **文件保存和输出 (对每个图表都适用):**
    *   **必须** 使用 `plt.savefig()` 保存图表，不要使用 `plt.show()`。
    *   图表应保存到`'./output'`目录中。主程序会自动将此路径替换为正确的输出目录。文件名应根据图表标题生成，并确保文件名合法（例如，替换非法字符）。
    *   **关键: 每保存一个文件后，必须立即使用 `print()` 语句输出所保存图表的完整路径。** 这是脚本与主程序通信的唯一方式。

请只返回一个完整的Python代码块，该代码块能够处理所有数据对象并为每个对象生成和保存一个图表。
"""
        
        # 调用大模型
        response = self.call_llm(prompt)
        
        # 提取代码部分
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        return code
    
    def execute_visualization_code(self, code: str) -> List[str]:
        """
        执行可视化代码并返回生成的图表路径
        
        Args:
            code: 可视化代码
            
        Returns:
            List[str]: 生成的图表路径列表
        """
        # 修改代码中的输出路径，确保所有引用都被替换
        modified_code = code.replace("'./output'", f"'{self.output_dir}'").replace('"./output"', f'"{self.output_dir}"')

        # 创建临时Python文件
        temp_file = os.path.join(self.temp_dir, "visualization_code.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        # 执行代码
        try:
            # 使用subprocess执行，捕获输出
            result = subprocess.run(
                ['python', temp_file], 
                capture_output=True, 
                text=True, 
                check=False,
                encoding='utf-8' # 确保正确解码标准输出
            )
            
            if result.returncode != 0:
                print(f"代码执行失败:\n--- STDERR ---\n{result.stderr}\n--- STDOUT ---\n{result.stdout}")
                return []
            
            # 从stdout中提取文件路径
            output_paths = result.stdout.strip().split('\n')
            generated_files = []
            for path in output_paths:
                clean_path = path.strip()
                if clean_path and os.path.exists(clean_path):
                    generated_files.append(clean_path)
                elif clean_path:
                    print(f"警告: 脚本输出路径 '{clean_path}' 但文件不存在或路径不正确。")
            
            return generated_files
        except Exception as e:
            print(f"执行代码时出错: {e}")
            return []
    
    def improve_visualization(self, code: str, feedback: str) -> str:
        """
        根据反馈改进可视化代码
        
        Args:
            code: 原始可视化代码
            feedback: 改进反馈
            
        Returns:
            str: 改进后的代码
        """
        # 构建提示词
        prompt = f"""
请根据以下反馈和我们最新的代码生成标准，改进给定的Python可视化代码。

**原始代码:**
```python
{code}
```

**改进反馈:**
{feedback}

**核心代码标准 (请确保改进后的代码遵循这些标准):**
1.  **循环处理:** 如果代码处理多组数据，确保循环是正确的。
2.  **专业美学:** 使用`seaborn`样式、专业调色板、清晰的数据标签。
3.  **信息标注:** 必须包含准确的标题、坐标轴标签和数据来源注释 (`plt.figtext`)。
4.  **文件保存和输出:** 每个图表保存后，必须紧接着用`print()`输出其完整文件路径。例如: `plt.savefig(path)`, `print(path)`.

请只返回改进后的、完整的Python代码块，不要包含任何其他解释或文本。
"""
        
        # 调用大模型
        response = self.call_llm(prompt)
        
        # 提取代码部分
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            improved_code = code_match.group(1)
        else:
            improved_code = response
        
        return improved_code
    
    def visualize_report(self, report_text: str) -> List[Dict[str, Any]]:
        """
        从研报文本中提取数据并生成可视化图表。
        现在返回一个包含元数据和文件路径的字典列表。
        
        Args:
            report_text: 研报文本
            
        Returns:
            List[Dict[str, Any]]: 一个列表，每个元素是一个字典，包含 "metadata" 和 "chart_file"。
        """
        print("开始从研报文本中提取数据...")
        all_data = self.extract_data_from_text(report_text)

        if not isinstance(all_data, list) or "error" in all_data:
            print(f"数据提取失败或格式不正确: {all_data}")
            return []
        
        print(f"成功提取 {len(all_data)} 组数据")
        
        generated_charts = []
        for data_item in all_data:
            item_title = data_item.get('title', 'Untitled')
            print(f"正在为 '{item_title}' 生成可视化...")
            
            # 为单个数据项生成代码
            code = self.generate_visualization_code([data_item])
            
            # 执行代码
            generated_files = self.execute_visualization_code(code)
            
            if generated_files:
                chart_file = generated_files[0]
                generated_charts.append({
                    "metadata": data_item,
                    "chart_file": chart_file
                })
                print(f"  > 成功生成图表: {chart_file}")
            else:
                print(f"  > 首次执行失败，尝试修复代码并重新执行...")
                improved_code = self.improve_visualization(
                    code, 
                    "代码执行失败，请修复错误并确保使用plt.savefig保存图表，并用print()输出文件路径。"
                )
                generated_files = self.execute_visualization_code(improved_code)
                if generated_files:
                    chart_file = generated_files[0]
                    generated_charts.append({
                        "metadata": data_item,
                        "chart_file": chart_file
                    })
                    print(f"  > 修复后成功生成图表: {chart_file}")
                else:
                    print(f"警告: 未能为 '{item_title}' 生成图表。")

        print(f"总共生成了 {len(generated_charts)} 个可视化图表")
        return generated_charts
    
    def visualize_report_file(self, report_file: str) -> List[str]:
        """
        从研报文件中提取数据并生成可视化图表
        
        Args:
            report_file: 研报文件路径
            
        Returns:
            List[str]: 生成的图表路径列表
        """
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_text = f.read()
            return self.visualize_report(report_text)
        except FileNotFoundError:
            print(f"错误: 研报文件未找到于路径 {report_file}")
            return []
        except Exception as e:
            print(f"读取研报文件失败: {e}")
            return []
    
    def visualize_specific_data(self, report_text: str, data_type: str, chart_type: str) -> List[str]:
        """
        从研报中提取特定类型的数据并生成指定类型的图表
        
        Args:
            report_text: 研报文本
            data_type: 数据类型，如"GDP", "CPI", "投资"
            chart_type: 图表类型，如"bar", "line", "pie", "heatmap"
            
        Returns:
            List[str]: 生成的图表路径列表
        """
        # 构建提取特定数据的提示词
        extract_prompt = f"""
请从以下研报文本中仅提取与 **"{data_type}"** 相关的所有数据，并将其组织成JSON格式。
要求:
1. 强相关性：只提取与"{data_type}"主题紧密相关的数据。
2. 结构清晰：对于每组数据，提供标题、单位、数据点和来源(如有)。
3. 返回格式必须是有效的JSON数组，每个数据集包含以下字段:
   - "title": 数据标题
   - "type": 数据类型(如"time_series", "comparison", "single_value")
   - "chart_type": 推荐的图表类型, 优先使用用户指定的 `{chart_type}`
   - "unit": 数据单位(如"%", "亿美元")
   - "data": 数据点数组或对象
   - "source": 数据来源(如有)
   - "description": 简短描述

研报文本:
{report_text[:10000]}  # 限制文本长度以避免超过token限制

只返回JSON格式的数据，不要包含任何其他解释或文本。
"""
        
        # 调用大模型提取数据
        response = self.call_llm(extract_prompt)
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 构建生成特定图表的提示词
            viz_prompt = f"""
请使用以下数据生成 **{chart_type}** 图表的Python代码。

数据:
{json.dumps(data, ensure_ascii=False, indent=2)}

要求:
1. 必须生成`{chart_type}`类型的图表。
2. 遵循所有专业可视化标准：使用`seaborn`和`matplotlib`，确保中文显示，添加标题、标签、数据来源，并美化图表。
3. 使用 `plt.savefig()` 保存图表到 `'./output'` 目录, 不要使用 `plt.show()`。
4. **关键：** 保存图表后，用 `print()` 输出完整的文件路径。

只返回Python代码，不要包含任何其他解释或文本。
"""
            
            # 调用大模型生成可视化代码
            code = self.call_llm(viz_prompt)
            
            # 提取代码部分
            code_match = re.search(r'```python\s*([\s\S]*?)\s*```', code)
            if code_match:
                code = code_match.group(1)
            
            # 执行代码
            return self.execute_visualization_code(code)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return []
    
    def generate_interactive_chart(self, report_text: str, data_type: str) -> str:
        """
        生成交互式图表(HTML)
        
        Args:
            report_text: 研报文本
            data_type: 数据类型
            
        Returns:
            str: 生成的HTML文件路径
        """
        # 提取数据
        data = self.extract_data_from_text(report_text, data_type)
        
        if "error" in data:
            print(f"数据提取失败: {data['error']}")
            return ""
        
        # 构建生成交互式图表的提示词
        prompt = f"""
请使用以下数据生成交互式图表的Python代码。

数据:
{json.dumps(data, ensure_ascii=False, indent=2)}

要求:
1. 使用 **Plotly** 或 **Bokeh** 库生成交互式图表。
2. 确保中文标签和标题显示正常。
3. 添加悬停提示、缩放功能等交互元素。
4. 将图表保存为HTML文件，路径为`'./output/interactive_{data_type}.html'`。
5. 代码必须能独立运行，包含所有必要的导入语句。
6. **关键：** 代码执行后，使用`print()`打印出生成的HTML文件路径。

只返回Python代码，不要包含任何其他解释或文本。
"""
        
        # 调用大模型
        response = self.call_llm(prompt)
        
        # 提取代码部分
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        # 修改代码中的输出路径
        modified_code = code.replace('./output/', f'{self.output_dir}/')
        
        # 创建临时Python文件
        temp_file = os.path.join(self.temp_dir, "interactive_chart.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        # 执行代码
        try:
            result = subprocess.run(
                ['python', temp_file], 
                capture_output=True, 
                text=True, 
                check=False,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                print(f"交互式图表代码执行失败: {result.stderr}")
                return ""

            # 从stdout获取html文件路径
            html_file_path = result.stdout.strip()
            if os.path.exists(html_file_path):
                return html_file_path
            else:
                print(f"警告: 脚本输出路径 '{html_file_path}' 但文件不存在。")
                # 尝试在输出目录中寻找
                for file in os.listdir(self.output_dir):
                    if file.endswith('.html'):
                        return os.path.join(self.output_dir, file)
                return ""

        except Exception as e:
            print(f"执行交互式图表代码时出错: {e}")
            return ""
    
    def create_dashboard(self, report_file: str, output_file: str = "dashboard.html") -> str:
        """
        为研报创建完整的可视化仪表盘
        
        Args:
            report_file: 研报文件路径
            output_file: 输出HTML文件名
            
        Returns:
            str: 生成的仪表盘HTML文件路径
        """
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_text = f.read()
        except FileNotFoundError:
            print(f"错误: 研报文件未找到于路径 {report_file}")
            return ""
        except Exception as e:
            print(f"读取研报文件失败: {e}")
            return ""
        
        # 提取数据
        data = self.extract_data_from_text(report_text)
        
        if "error" in data or not data:
            print(f"为创建仪表盘提取数据失败: {data.get('error', '未提取到数据')}")
            return ""
        
        # 构建生成仪表盘的提示词
        prompt = f"""
作为一名专业的数据可视化工程师，请根据以下数据生成一个完整的可视化仪表盘的Python代码。

数据:
{json.dumps(data, ensure_ascii=False, indent=2)}

要求:
1. 使用Dash、Streamlit或Flask+Plotly创建交互式仪表盘。
2. 包含多个图表，展示不同类型的数据。
3. 添加标题、说明文字和数据来源。
4. 确保中文显示正常。
5. 代码必须能独立运行，包含所有必要的导入语句。
6. 将仪表盘应用运行或保存为HTML文件。如果保存为文件，路径为`'./output/dashboard.html'`。
7. **关键:** 如果生成了文件，请在最后用`print()`输出文件的完整路径。

只返回Python代码，不要包含任何其他解释或文本。
"""
        
        # 调用大模型
        response = self.call_llm(prompt)
        
        # 提取代码部分
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        # 修改代码中的输出路径
        modified_code = code.replace('./output/', f'{self.output_dir}/')
        
        # 创建临时Python文件
        temp_file = os.path.join(self.temp_dir, "dashboard.py")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_code)
        
        # 执行代码
        try:
            result = subprocess.run(
                ['python', temp_file], 
                capture_output=True, 
                text=True, 
                check=False,
                encoding='utf-8'
            )

            if result.returncode != 0:
                print(f"仪表盘代码执行失败: {result.stderr}")
                return ""
            
            # 从stdout获取文件路径
            dashboard_file_path = result.stdout.strip()
            if dashboard_file_path and os.path.exists(dashboard_file_path):
                return dashboard_file_path
            else:
                 # 如果没有打印路径，则按原逻辑查找
                dashboard_file = os.path.join(self.output_dir, output_file)
                if os.path.exists(dashboard_file):
                    return dashboard_file
                # 查找任何生成的HTML文件
                for file in os.listdir(self.output_dir):
                    if file.endswith('.html'):
                        return os.path.join(self.output_dir, file)
                print("警告: 仪表盘脚本已执行，但未找到输出的HTML文件。")
                return ""
        except Exception as e:
            print(f"执行仪表盘代码时出错: {e}")
            return ""

# --- 辅助函数 (原__init__.py中的导出内容) ---

def visualize_report(report_file, output_dir="report_visuals"):
    """
    从研报文件中提取数据并生成可视化图表
    
    Args:
        report_file: 研报文件路径
        output_dir: 图表输出目录
        
    Returns:
        List[Dict[str, Any]]: 生成的图表对象列表，包含元数据和路径
    """
    visualizer = AIReportVisualizer(output_dir=output_dir)
    return visualizer.visualize_report_file(report_file)

def create_interactive_dashboard(report_file, output_dir="report_visuals"):
    """
    为研报创建交互式仪表盘
    
    Args:
        report_file: 研报文件路径
        output_dir: 输出目录
        
    Returns:
        str: 生成的仪表盘HTML文件路径
    """
    visualizer = AIReportVisualizer(output_dir=output_dir)
    return visualizer.create_dashboard(report_file)

def visualize_specific_data(report_file, data_type, chart_type, output_dir="report_visuals"):
    """
    从研报中提取特定类型的数据并生成指定类型的图表
    
    Args:
        report_file: 研报文件路径
        data_type: 数据类型，如"GDP", "CPI", "投资"
        chart_type: 图表类型，如"bar", "line", "pie", "heatmap"
        output_dir: 图表输出目录
        
    Returns:
        List[str]: 生成的图表路径列表
    """
    try:
        visualizer = AIReportVisualizer(output_dir=output_dir)
        with open(report_file, 'r', encoding='utf-8') as f:
            report_text = f.read()
        return visualizer.visualize_specific_data(report_text, data_type, chart_type)
    except FileNotFoundError:
        print(f"错误: 研报文件未找到于路径 {report_file}")
        return []
    except Exception as e:
        print(f"可视化特定数据时发生错误: {e}")
        return []

# --- 示例使用 (原example.py的内容) ---
if __name__ == "__main__":
    # 创建一个模拟的研报文件
    report_content = """
    2023年全球宏观经济分析报告

    摘要：
    2023年全球经济在波动中复苏。主要经济体GDP增长呈现分化态势，
    美国GDP增长率为2.5%，中国为5.2%，欧元区为0.5%。
    通货膨胀方面，美国CPI同比上涨3.4%，欧元区为5.5%。
    
    投资结构方面，高科技产业投资占比达到40%，传统能源占比30%，其他产业占比30%。
    数据来源：世界银行、IMF。
    """
    
    # 将模拟内容写入文件
    if not os.path.exists("example_report.md"):
        with open("example_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)

    report_file = "Macro_Research_Report.md"
    if not os.path.exists(report_file):
        report_file = "example_report.md" # 如果主报告不存在，则使用示例报告

    output_dir = "report_visuals"
    
    print("--- 1. 使用AIReportVisualizer类 (新版DataFrame流程) ---")
    # 创建可视化器
    visualizer = AIReportVisualizer(output_dir=output_dir)
    
    # 生成所有图表
    charts = visualizer.visualize_report_file(report_file)
    if charts:
        print(f"已生成 {len(charts)} 个图表:")
        for chart in charts:
            print(f"- {chart}")
    else:
        print("未能生成任何图表。")
    
    print("\n--- 2. 使用独立的辅助函数 (旧版兼容/特定图表) ---")
    # 生成特定类型的图表
    gdp_charts = visualize_specific_data(report_file, "GDP", "bar", output_dir)
    if gdp_charts:
        print(f"已生成 {len(gdp_charts)} 个GDP柱状图")
    else:
        print("未能生成特定的GDP图表。")
    
    # 创建交互式仪表盘
    dashboard = create_interactive_dashboard(report_file, output_dir)
    if dashboard:
        print(f"交互式仪表盘已生成: {dashboard}")
    else:
        print("未能生成交互式仪表盘") 