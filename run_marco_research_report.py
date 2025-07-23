# 导入必要的库和模块
from datetime import datetime
import os  # 操作系统接口，用于文件路径操作
import shutil
import time  # 时间相关功能，用于延时和计时
import yaml  # YAML格式数据处理
import openai  # OpenAI API客户端
import re  # 正则表达式，用于文本模式匹配
import subprocess  # 子进程管理，用于执行外部命令
import sys  # 系统相关参数和函数
import pandas as pd
import matplotlib.pyplot as plt
from ddgs import DDGS  # DuckDuckGo搜索引擎接口

from app.document_conversion_pipeline import DocumentConversionPipeline
from app.llm.config import LLMConfig
from app.llm.llm_helper import LLMHelper
from app.marco.frameworks.pocketflow import Flow, Node
from app.marco.tools.document_processing.doc_converter import convert_to_docx_with_indent
# 导入项目自定义模块
 # 工作流节点和流程管理
from dotenv import load_dotenv  # 环境变量加载工具

# --- 图表生成与环境配置 ---

# 配置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 添加项目根目录到Python路径，确保能正确导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
sys.path.insert(0, project_root)  # 将项目根目录添加到Python模块搜索路径



# --- 环境与配置初始化 ---

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
try:
    llm_config = LLMConfig(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model = os.getenv("OPENAI_MODEL", "gpt-4"),
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens= int(os.getenv("OPENAI_MAX_TOKENS", "8192")),
    )
    client = LLMHelper(llm_config)
except openai.OpenAIError as e:
    print(f"OpenAI客户端初始化失败，请检查您的.env文件或环境变量: {e}")
    exit()


# --- 核心功能函数 ---

def call_llm(prompt: str) -> str:
    """
    调用大语言模型 (LLM) 并返回其响应

    Args:
        prompt (str): 发送给LLM的提示文本

    Returns:
        str: LLM的响应内容，如果调用失败则返回错误信息

    Raises:
        openai.APIError: 当API调用出现问题时抛出
    """
    try:

        response = client.ask([{"role": "user", "content": prompt}], temperature=0.3)
        return response
        # # 创建聊天完成请求，使用环境变量中的模型配置
        # response = client.chat.completions.create(
        #     model=os.getenv("OPENAI_MODEL", "gpt-4"),  # 默认使用gpt-4模型
        #     messages=[{"role": "user", "content": prompt}],  # 用户角色消息
        #     temperature=0.6,  # 控制输出的随机性，0.6提供适度的创造性
        #     max_tokens=8192  # deepseek最大tokens值
        # )
        # return response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"LLM API调用时发生错误: {e}")
        return f"错误：LLM调用失败 - {e}"


def search_web(term: str):
    """
    使用 DuckDuckGo 执行网络搜索
    """
    print(f"  > 正在搜索关键词: {term}")
    try:
        with DDGS() as ddgs:
            # 使用中文区域设置，限制返回3个结果以提高效率
            results = ddgs.text(query=term, region="cn-zh", max_results=3)
        return list(results)
    except Exception as e:
        print(f"  > 搜索 '{term}' 时出错: {e}")
        return []


def generate_chart_from_table(md_table: str, chart_type: str, output_path: str, title: str):
    """从Markdown表格数据生成图表并保存为图片。"""
    try:
        # 将Markdown表格转换为二维列表
        lines = [line.strip() for line in md_table.strip().split('\n')]
        header = [h.strip() for h in lines[0].strip('|').split('|')]
        data = []
        for line in lines[2:]:
            data.append([d.strip() for d in line.strip('|').split('|')])

        df = pd.DataFrame(data, columns=header)

        # 数据清洗，尝试将数值列转换为数字
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type.lower() == 'bar':
            df.plot(kind='bar', x=df.columns[0], ax=ax)
        elif chart_type.lower() == 'line':
            df.plot(kind='line', x=df.columns[0], ax=ax)
        elif chart_type.lower() == 'pie':
            # 饼图需要指定数值列
            numeric_col = df.select_dtypes(include=['number']).columns[0]
            df.plot(kind='pie', y=numeric_col, labels=df[df.columns[0]], ax=ax, autopct='%1.1f%%')
            ax.set_ylabel('')  # 饼图不需要y轴标签
        else:
            print(f"不支持的图表类型: {chart_type}")
            return

        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"  > 图表已生成: {output_path}")

    except Exception as e:
        print(f"  > 生成图表 '{title}' 时出错: {e}")


# --- 工作流节点类 ---

class MacroResearchFlow(Node):
    """宏观经济研究流程的决策节点"""

    def prep(self, shared):
        context = shared.get("context", [])
        context_str = yaml.dump(context, allow_unicode=True)
        topic = shared["topic"]  # 研究主题
        focus_areas = shared.get("focus_areas", ["GDP", "CPI", "利率", "汇率", "财政政策", "货币政策"])
        generated_sections = shared.get("generated_sections", [])
        return topic, focus_areas, context_str, generated_sections

    def exec(self, inputs):
        topic, focus_areas, context, generated_sections = inputs
        print(f"\n正在分析 {topic} 的研究进度...")

        # 提取已生成章节的名称
        generated_section_names = [section['name'] for section in generated_sections]

        # 设置终止条件
        # 1. 章节数量达到6个以上时，倾向于完成研报
        max_sections = 6
        section_count = len(generated_sections)

        # 2. 定义标准章节类型，检查是否已覆盖所有核心章节类型
        standard_section_types = ["宏观经济", "政策分析", "行业影响", "风险预测"]
        covered_types = []

        for section_name in generated_section_names:
            section_name_lower = section_name.lower()
            if any(typ.lower() in section_name_lower for typ in ["宏观", "经济", "gdp"]):
                covered_types.append("宏观经济")
            if any(typ.lower() in section_name_lower for typ in ["政策", "监管"]):
                covered_types.append("政策分析")
            if any(typ.lower() in section_name_lower for typ in ["行业", "影响", "投资"]):
                covered_types.append("行业影响")
            if any(typ.lower() in section_name_lower for typ in ["风险", "预测", "展望"]):
                covered_types.append("风险预测")

        covered_types = list(set(covered_types))  # 去重
        coverage_ratio = len(covered_types) / len(standard_section_types)

        # 基于终止条件做预判决策
        predecision = None
        predecision_reason = None

        if section_count >= max_sections:
            predecision = "complete"
            predecision_reason = f"已生成{section_count}个章节，达到推荐的最大章节数{max_sections}"
        elif coverage_ratio >= 0.75:  # 覆盖了至少75%的标准章节类型
            predecision = "complete"
            predecision_reason = f"已覆盖{len(covered_types)}/{len(standard_section_types)}个核心章节类型，报告结构已相对完整"

        # 构建决策提示词，加入预判决策信息
        prompt = f"""
已生成的章节列表: {generated_section_names}

针对 "{topic}" 的宏观经济研究，需要做出下一步行动决策。

当前状态分析:
- 已生成章节数量: {section_count}个
- 已覆盖的核心章节类型: {', '.join(covered_types) if covered_types else '尚未覆盖标准章节'}
- 系统预判决策: {predecision or '尚无预判'} (原因: {predecision_reason or 'N/A'})

请综合判断下一步最合适的行动：
1) 搜索更多信息 (如果当前信息不足以覆盖关键领域)
2) 开始生成某个新章节内容 (如果还有重要章节未生成)
3) 完成研报生成 (如果关键章节已足够，可以整合报告了)

附加考虑因素:
- 章节数量建议控制在5-7个，当前已有{section_count}个
- 宏观经济研究应重点关注：{', '.join(focus_areas)}
- 需要对政策进行解读，分析宏观变量间交互影响，提供全球视野和风险预警

请以 YAML 格式输出：
```yaml
action: search/generate/complete  # search表示继续搜索，generate表示生成章节，complete表示完成
reason: 做出此判断的原因
search_terms: # 如果是search，列出要搜索的关键词列表
  - 关键词1 
  - 关键词2
section: # 如果是generate，指定要生成的章节名称
  name: 章节名称 # 例如：全球宏观经济背景/货币政策分析等
  focus: 重点关注内容 # 具体要分析的宏观经济指标和政策关联
```"""
        resp = call_llm(prompt)
        yaml_str = resp.split("```yaml")[1].split("```", 1)[0].strip() if "```yaml" in resp else resp
        result = yaml.safe_load(yaml_str)

        # 打印决策结果
        print(f"决策结果: {result['action']}")
        print(f"决策原因: {result['reason']}")
        if result['action'] == 'search':
            print("需要搜索的关键词:", result['search_terms'])
        elif result['action'] == 'generate':
            print(f"即将生成章节: {result['section']['name']}")

        # 如果有预判决策为complete，并且LLM决定继续generate，且章节数已达到最大值，
        # 则强制转为complete决策以确保终止
        if predecision == "complete" and result["action"] == "generate" and section_count >= max_sections:
            print(f"注意: 已达到最大章节数({max_sections})，强制完成报告")
            result["action"] = "complete"
            result["reason"] = f"{result['reason']} (系统判断: {predecision_reason})"

        return result

    def post(self, shared, prep_res, exec_res):
        action = exec_res.get("action")
        if action == "search":
            shared["search_terms"] = exec_res.get("search_terms", [])
            print("\n=== 开始信息搜索阶段 ===")
        elif action == "generate":
            shared["current_section"] = exec_res.get("section", {})
            print("\n=== 开始章节生成阶段 ===")
        elif action == "complete":
            print("\n=== 开始完成研报阶段 ===")
        return action


class SearchKeywordsGenerator(Node):
    """根据研究主题生成搜索关键词的节点"""

    def prep(self, shared):
        return (
            shared.get("topic", ""),
            shared.get("focus_areas", ["GDP", "CPI", "利率", "汇率", "财政政策", "货币政策"])
        )

    def exec(self, inputs):
        topic, focus_areas = inputs
        print("\n正在为研究主题生成搜索关键词...")

        prompt = f"""
请为以下宏观经济研究主题生成5-7个具体的搜索关键词：

研究主题：{topic}
重点关注的宏观经济指标：{', '.join(focus_areas)}

搜索关键词应该覆盖：
1. 全球和主要经济体的宏观指标数据与预测
2. 相关的政策报告与权威解读
3. 行业特定影响和趋势（如AI基建与算力投资）
4. 潜在风险因素和"灰犀牛"事件

请直接返回一个Python列表格式，每个元素为一个搜索关键词字符串，不要包含任何其他解释。例如：
['全球GDP增长预测 2023-2026', '中国CPI趋势与预测', '美联储利率政策对全球资本流动影响']
"""
        keywords_str = call_llm(prompt)
        try:
            # 尝试解析返回的列表格式
            import ast
            keywords = ast.literal_eval(keywords_str)
            if not isinstance(keywords, list):
                raise ValueError("返回值不是列表")
        except:
            # 如果解析失败，尝试提取可能包含在代码块中的内容
            if "```" in keywords_str:
                code_block = re.search(r'```(?:python)?\s*(.*?)```', keywords_str, re.DOTALL)
                if code_block:
                    try:
                        keywords = ast.literal_eval(code_block.group(1).strip())
                    except:
                        keywords = []
            else:
                # 如果以上方法都失败，则使用一些默认值
                keywords = []

            if not keywords:
                print("无法解析LLM返回的关键词列表，使用默认关键词")
                keywords = [
                    f"全球主要经济体GDP增长预测 {topic}",
                    f"主要国家CPI趋势与货币政策 {topic}",
                    f"中国财政政策对{topic}的影响",
                    f"国际贸易关系与{topic}",
                    f"{topic}的经济影响分析"
                ]

        print(f"已生成 {len(keywords)} 个搜索关键词")
        return keywords

    def post(self, shared, prep_res, exec_res):
        shared["search_terms"] = exec_res
        return "search_ready"


class WebSearcher(Node):
    """执行网络搜索的节点"""

    def prep(self, shared):
        return shared.get("search_terms", [])

    def exec(self, search_terms):
        all_results = []
        total = len(search_terms)
        for i, term in enumerate(search_terms, 1):
            print(f"\n[{i}/{total}] 搜索关键词: {term}")
            results = search_web(term)
            print(f"找到 {len(results)} 条相关信息")
            all_results.append({"term": term, "results": results})
            
            if i < total:
                print(f"进度: {i}/{total} 完成，等待10秒后继续...")
                for countdown in range(10, 0, -1):
                    print(f"\r剩余等待时间: {countdown}秒 (按 Ctrl+C 可中断)", end="", flush=True)
                    time.sleep(1)
                print()  # 换行
        return all_results

    def post(self, shared, prep_res, exec_res):
        # 将本次搜索结果暂存，交由下一个节点处理
        shared["latest_search_results"] = exec_res
        print("\n信息搜索完成，即将进行上下文总结...")
        return "search_done"


class ContextSummarizer(Node):
    """对搜索结果进行总结，同时保留结构化数据"""

    def prep(self, shared):
        # 获取最近一次的搜索结果
        return shared.get("latest_search_results", [])

    def exec(self, latest_search_results):
        if not latest_search_results:
            return "没有新的搜索结果需要总结。"

        print("\n正在总结新的搜索结果...")
        search_results_str = yaml.dump(latest_search_results, allow_unicode=True)

        prompt = f"""
请总结以下网络搜索结果，以便将其作为宏观经济报告的上下文。

**关键要求**:
1.  **提取核心信息**: 提炼出与研究主题最相关的事实、数据、观点和预测。
2.  **保留结构化数据**: **此项至关重要**。如果搜索结果中包含任何表格、JSON、YAML、列表或任何其他格式的结构化数据，必须将其**原封不动地**复制到摘要中。不允许对这些结构化数据进行任何形式的简化、概括或省略。这是为了保留后续数据可视化的原始数据。
3.  **形成连贯段落**: 将非结构化的信息整合成流畅、连贯的摘要段落。
4.  **关注数据来源**: 如果信息来源可靠（如政府报告、知名研究机构），请在摘要中提及。

**待总结的搜索结果**:
```
{search_results_str}
```

请生成简洁、信息密集的摘要。
"""
        summary = call_llm(prompt)
        print(f"上下文总结完成，长度: {len(summary)} 字符")
        return summary

    def post(self, shared, prep_res, exec_res):
        # 将总结后的上下文添加到总的上下文中
        context_list = shared.get("context", [])
        context_list.append({"summary_from_search": exec_res})
        shared["context"] = context_list
        # 清理掉临时的搜索结果，避免重复处理
        if "latest_search_results" in shared:
            del shared["latest_search_results"]
        print("上下文已更新，返回决策节点...")
        return "summary_done"


class SectionGenerator(Node):
    """生成研报章节的节点"""

    def prep(self, shared):
        return (
            shared.get("topic"),
            shared.get("current_section", {}),
            shared.get("context", []),
            shared.get("focus_areas", ["GDP", "CPI", "利率", "汇率", "财政政策", "货币政策"]),
            shared.get("generated_sections", [])
        )

    def exec(self, inputs):
        topic, section, context, focus_areas, generated_sections = inputs
        print(f"\n开始生成 {section['name']} 章节...")

        # 确定章节序号
        section_number = len(generated_sections) + 1

        # 确定标题格式
        section_title = f"{section_number}. {section['name']}"

        # --- 增强上下文连贯性 ---
        report_outline = ""
        last_section_summary = ""
        # 如果已有生成的章节，则创建报告大纲和上一章节的摘要
        if generated_sections:
            report_outline += "报告已有大纲：\n"
            for i, sec in enumerate(generated_sections):
                report_outline += f"{i + 1}. {sec['name']}\n"

            # 提取上一章节内容作为参考
            last_section_content = generated_sections[-1]['content']
            last_section_summary = f"作为参考，这是上一章节 '{generated_sections[-1]['name']}' 的内容摘要：\n{last_section_content[:300]}...\n"
        else:
            last_section_summary = "这是报告的第一个章节。"
        # --- 结束 ---

        context_str = yaml.dump(context, allow_unicode=True)
        prompt = f"""
请为宏观经济研究报告 "{topic}" 生成以下章节内容：

**上下文参考**:
{report_outline}
{last_section_summary}

**当前任务**:
章节标题：{section_title}
章节重点：{section['focus']}
需关注的宏观指标：{', '.join(focus_areas)}

**撰写要求**:
1. **结构格式与内容长度**: 
   - 本章节已有标题为 "{section_title}"，请勿重复生成标题
   - 生成3-5个二级标题，格式为 "### {section_number}.1 标题名" 这种带序号的格式
   - **内容要详实丰富，本章节总字数应在1200-2000字之间。**
   - 使用自然段落撰写内容，段落之间应有良好的逻辑衔接，避免使用编号列表或分点格式

2. **内容要求**:
   - 数据与事实：提供具体的经济数据、政策文件引用与实际案例
   - 宏观分析：深入分析宏观经济指标间的关联影响与传导机制
   - 比较视角：提供国际间政策、经济表现的对比分析
   - 风险预警：识别潜在的"灰犀牛"事件与系统性风险指标

3. **专业性**:
   - 使用专业经济学术语
   - 保持客观中立的表述风格
   - 提供多维度、辩证的分析视角
   - 满足中国证券业协会《发布证券研究报告暂行规定》排版与披露要求，论点-论据链完整，章节衔接流畅。

4. **数据驱动与图表生成**:
   - **数据溯源**: 引用数据或事实时，**必须**在句末用括号注明来源，格式为 `(来源：[来源名称], [年份])`。
   - **图表生成**: 当内容涉及数据对比(历年变化)、趋势变化、结构占比时，**必须**在段落后插入相应数据的Markdown表格，并在表格下方添加图表生成标记。
   - **标记格式**: `[CHART-图表类型:./reports/charts/图表名.png:图表标题]` (图表类型支持: `bar`, `line`, `pie`)
   - **示例**: `[CHART-bar:./reports/charts/gdp_growth_rate.png:2020-2025年全球主要经济体GDP增长率]`

**参考资料**:
{context_str[:4000]}
"""
        content = call_llm(prompt)
        print(f"章节 {section['name']} 生成完成!")
        print(f"内容长度: {len(content)} 字符")
        print(f"内容预览: {content[:100]}...")

        # 标准化标题格式
        # 处理二级标题和三级标题的编号
        lines = content.split("\n")
        new_lines = []

        for line in lines:
            # 检测到三级标题 (####)
            if line.startswith("#### "):
                # 提取标题文本
                title_text = line.replace("#### ", "").strip()
                # 如果标题不已经包含编号格式，则添加编号
                if not re.match(r'^\d+\.\d+\.\d+\s+', title_text):
                    new_lines.append(
                        f"#### {section_number}.{len(new_lines) // 10 + 1}.{len(new_lines) % 10 + 1} {title_text}")
                else:
                    new_lines.append(line)
            # 检测到二级标题 (###)
            elif line.startswith("### "):
                # 提取标题文本
                title_text = line.replace("### ", "").strip()
                # 如果标题不已经包含编号格式，则添加编号
                if not re.match(r'^\d+\.\d+\s+', title_text):
                    subsection_number = sum(1 for l in new_lines if l.startswith("### ")) + 1
                    new_lines.append(f"### {section_number}.{subsection_number} {title_text}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        content = "\n".join(new_lines)

        return {
            "name": section['name'],
            "title": section_title,
            "content": content
        }

    def post(self, shared, prep_res, exec_res):
        sections = shared.get("generated_sections", [])
        sections.append(exec_res)
        shared["generated_sections"] = sections
        print("\n返回决策节点，继续分析下一步...")
        return "continue"


class ReportCompleter(Node):
    """完成研报整合的节点"""

    def prep(self, shared):
        return (
            shared.get("topic"),
            shared.get("generated_sections", []),
            shared.get("focus_areas", ["GDP", "CPI", "利率", "汇率", "财政政策", "货币政策"])
        )

    def exec(self, inputs):
        topic, sections, focus_areas = inputs
        print(f"\n=== 开始整合最终研报 ===")

        # 生成研报标题
        content = f"# {topic}宏观经济分析报告\n\n"

        # 生成摘要
        summary_prompt = f"""
请为一份关于"{topic}"的宏观经济分析报告撰写摘要。
摘要应该概述报告的主要发现，包括：
1. 核心宏观经济指标（{', '.join(focus_areas)}）的趋势
2. 政策环境的关键变化
3. 对{topic}的主要影响
4. 潜在风险因素

摘要长度控制在300-400字之间。
"""
        summary = call_llm(summary_prompt)
        content += f"## 摘要\n\n{summary}\n\n"

        # 添加目录
        content += "## 目录\n\n"
        for section in sections:
            content += f"- {section['title']}\n"
        content += "\n"

        # 逐个添加章节内容
        for section in sections:
            print(f"添加章节: {section['title']}")
            content += f"\n## {section['title']}\n\n{section['content']}\n"

        # 添加结论部分
        conclusion_prompt = f"""
请为一份关于"{topic}"的宏观经济分析报告撰写结论部分。
该结论应总结报告的核心发现，并提出前瞻性建议。
结论格式应使用标题"## 结论与展望"，并包含400-500字的内容。
"""
        conclusion = call_llm(conclusion_prompt)
        content += "\n" + conclusion + "\n"

        return content

    def post(self, shared, prep_res, exec_res):
        print(f"\n=== 宏观经济研报生成完成！===")

        # 保存原始报告内容
        shared["report"] = exec_res

        # --- 处理图表并保存最终文件 ---
        # reports_dir = "reports"
        # charts_dir = "charts"
        # os.makedirs(charts_dir, exist_ok=True)
        final_content = shared["report"]

        # # 正则表达式查找图表标记和Markdown表格
        # chart_tags = re.findall(r'\[CHART-(.*?):(.*?:.*?)\]', final_content)
        # md_tables = re.findall(r'((?:\|.*\|[\r\n]+)+(?:\|-.*-+\|[\r\n]+)+(?:\|.*\|[\r\n]?)+)', final_content)

        # # 确保图表标记和表格数量匹配，以避免错位
        # if len(chart_tags) <= len(md_tables):
        #     for i, tag_parts in enumerate(chart_tags):
        #         tag_str = f"[CHART-{tag_parts[0]}:{tag_parts[1]}]"
        #         table_data = md_tables[i]

        #         # 解析图表标记
        #         try:
        #             chart_type, relative_path, chart_title = tag_parts[1].split(':', 2)
        #             full_path = os.path.join(project_root, relative_path.strip('./'))

        #             # 生成图表
        #             generate_chart_from_table(table_data, chart_type, full_path, chart_title)

        #             # 将表格和标记替换为图片链接
        #             replacement = f"![{chart_title}]({relative_path})\n\n"
        #             final_content = final_content.replace(table_data, "", 1)
        #             final_content = final_content.replace(tag_str, replacement, 1)
        #         except Exception as e:
        #             print(f"[警告] 处理图表 '{tag_str}' 时出错: {e}")
        # else:
        #     print(f"[警告] 找到 {len(chart_tags)} 个图表标记，但仅匹配到 {len(md_tables)} 个表格，图表生成可能不完整。")

        # 保存处理完图表的最终Markdown文件
        md_file = "Macro_Research_Report.md"
        #copy to reports
        os.makedirs("reports", exist_ok=True)
        try:
            shutil.copy(md_file, f"reports/Macro_Research_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        except Exception as e:
            print(f"[警告] 复制文件失败: {e}")
        # 删除原始文件
        if os.path.exists(md_file):
            os.remove(md_file)
        # 删除word文件
        if os.path.exists(f"Macro_Research_Report.docx"):
            os.remove(f"Macro_Research_Report.docx")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"最终报告已保存到 '{md_file}' 文件中")

        # 转换为Word文档
        # docx_file = os.path.join(reports_dir, "Macro_Research_Report.docx")
        # convert_to_docx_with_indent(md_file, docx_file)
        # 创建文档转换实例
        pipeline = DocumentConversionPipeline()
        
        # 运行文档转换流程
        result = pipeline.run_conversion(md_file,f"Macro_Research_Report.docx")
        return "report_saved"


# --- 主执行代码 ---
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='研报生成流程')
    parser.add_argument('--marco_name', default='国家级\"人工智能+\"政策效果评估', help='报告名称')
    parser.add_argument('--time', default='2023-2025', help='时间范围')


    args = parser.parse_args()

    # 构建工作流
    macro_flow = MacroResearchFlow()
    keyword_gen = SearchKeywordsGenerator()
    searcher = WebSearcher()
    summarizer = ContextSummarizer()  # 新增：上下文总结节点
    generator = SectionGenerator()
    completer = ReportCompleter()

    # 设置节点转换关系
    # 1. 流程从关键词生成开始，然后搜索
    keyword_gen - "search_ready" >> searcher

    # 2. 搜索完成后，进行上下文总结，然后进入主要的决策循环节点 macro_flow
    searcher - "search_done" >> summarizer
    summarizer - "summary_done" >> macro_flow

    # 3. 在决策循环中，如果LLM再次决定搜索，则直接跳到 searcher 节点
    #    （因为关键词在第一次已经生成，通常不需要反复生成）
    macro_flow - "search" >> searcher

    # 4. 其他路径保持不变
    macro_flow - "generate" >> generator
    generator - "continue" >> macro_flow
    macro_flow - "complete" >> completer

    # 5. 流程的起始节点为关键词生成
    flow = Flow(start=keyword_gen)

    # 设置初始共享状态
    shared_state = {
        "topic": f"{args.marco_name} ({args.time})",
        "focus_areas": ["GDP", "CPI", "利率", "汇率", "财政政策", "货币政策", "国际贸易关系"]
    }

    # 运行工作流
    result = flow.run(shared_state)

    print("宏观经济报告生成工作流程已完成")