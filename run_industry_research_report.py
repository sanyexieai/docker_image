import os
import shutil
import time
import yaml
import openai
import logging
from datetime import datetime
from duckduckgo_search import DDGS
from app.document_conversion_pipeline import DocumentConversionPipeline
from app.pocketflow import Node, Flow
from dotenv import load_dotenv
import argparse

from app.utils.search_engine import SearchEngine
# 加载环境变量
load_dotenv()

# 从环境变量中初始化 OpenAI API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 配置日志
def setup_logging():
    """配置日志记录"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/industry_workflow_{timestamp}.log"
    
    logger = logging.getLogger('IndustryWorkflow')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"📝 行业研究工作流日志已启动: {log_filename}")
    return logger

logger = setup_logging()

class IndustryResearchFlow(Node):  # 研报生成的决策节点
    def prep(self, shared):
        context = shared.get("context", [])
        generated_sections = shared.get("generated_sections", [])
        context_str = yaml.dump(context, allow_unicode=True)
        industry = shared["industry"]  # 行业名称
        # 记录已生成的章节名称
        generated_section_names = [section.get('name', '') for section in generated_sections]
        return industry, context_str, generated_section_names, shared

    def exec(self, inputs):
        industry, context, generated_section_names, shared = inputs
        logger.info(f"\n正在分析 {industry} 行业的研究进度...")
        logger.info(f"已生成的章节: {generated_section_names}")
        max_loops = 10
        loop_count = 0
        while True:
            if loop_count > max_loops:
                logger.warning("已达最大循环次数，自动完成。")
                return {"action": "complete", "reason": "已达最大循环次数，自动完成。"}
            loop_count += 1
            prompt = f"""
针对 {industry} 行业研究，分析已有信息：{context}

已生成的章节：{generated_section_names}

请判断下一步应该：
1) 搜索更多信息 - 如果信息不足
2) 开始生成某个章节内容 - 如果信息充足且还有重要章节未生成
3) 完成研报生成 - 如果所有重要章节都已生成

重要章节清单：
- 行业概述/行业概览
- 市场规模分析
- 竞争格局分析
- 技术发展趋势
- 政策环境分析
- 风险与挑战
- 发展前景预测

请以 YAML 格式输出：
```yaml
action: search/generate/complete  # search表示继续搜索，generate表示生成章节，complete表示完成
reason: 做出此判断的原因
search_terms: # 如果是search，列出要搜索的关键词列表
  - 关键词1 
  - 关键词2
section: # 如果是generate，指定要生成的单个章节
  name: 章节名称 # 如：行业概述/市场规模分析/竞争格局分析等
  focus: 重点关注内容 # 具体要分析的要点
```

注意：
- 如果某个章节已经生成过，不要重复生成
- 如果信息不足，优先选择search
- 如果所有重要章节都已生成，选择complete
- section字段必须是一个单个章节的字典，包含name和focus字段
- 不要返回章节列表，只返回一个要生成的章节
"""
            resp = call_llm(prompt)
            try:
                yaml_str = resp.split("```yaml")[1].split("```", 1)[0].strip()
                result = yaml.safe_load(yaml_str)
            except Exception as e:
                logger.error(f"解析YAML失败: {e}")
                logger.error(f"原始响应: {resp}")
                # 默认完成
                result = {"action": "complete", "reason": "解析失败，默认完成"}
            logger.info(f"决策结果: {result['action']}")
            logger.info(f"决策原因: {result['reason']}")
            if result['action'] == 'generate':
                section = result.get('section', {})
                if isinstance(section, dict) and 'name' in section:
                    if section['name'] in generated_section_names:
                        logger.warning(f"章节 {section['name']} 已生成，自动跳过，重新决策...")
                        important_sections = [
                            "行业概述/行业概览",
                            "市场规模分析",
                            "竞争格局分析",
                            "技术发展趋势",
                            "政策环境分析",
                            "风险与挑战",
                            "发展前景预测"
                        ]
                        for s in important_sections:
                            if s not in generated_section_names:
                                result['section'] = {'name': s, 'focus': '重点分析'}
                                logger.info(f"自动切换到未生成章节: {s}")
                                break
                        else:
                            result['action'] = 'complete'
                            result['reason'] = '所有重要章节都已生成，自动完成。'
                elif not section or 'name' not in section:
                    result['action'] = 'complete'
                    result['reason'] = '章节信息格式错误，自动完成。'
            if result['action'] == 'search':
                break
            elif result['action'] == 'generate':
                break
            elif result['action'] == 'complete':
                break
        return result

    def post(self, shared, prep_res, exec_res):
        action = exec_res.get("action")
        if action == "search":
            search_terms = exec_res.get("search_terms", [])
            if isinstance(search_terms, list):
                shared["search_terms"] = search_terms
                logger.info("\n=== 开始信息搜索阶段 ===")
            else:
                logger.error(f"搜索关键词格式错误: {search_terms}")
                return "complete"  # 出错时直接完成
        elif action == "generate":
            section = exec_res.get("section", {})
            if isinstance(section, dict) and 'name' in section:
                # 单个章节的情况
                shared["current_section"] = section
                logger.info("\n=== 开始章节生成阶段 ===")
            elif isinstance(section, list) and len(section) > 0:
                # 章节列表的情况，选择第一个未生成的章节
                generated_sections = shared.get("generated_sections", [])
                generated_names = [s.get('name', '') for s in generated_sections]
                
                for s in section:
                    if isinstance(s, dict) and 'name' in s and s['name'] not in generated_names:
                        shared["current_section"] = s
                        logger.info(f"\n=== 开始章节生成阶段: {s['name']} ===")
                        break
                else:
                    # 所有章节都已生成
                    logger.info("所有章节都已生成，转为完成状态")
                    return "complete"
            else:
                logger.error(f"章节信息格式错误: {section}")
                return "complete"  # 出错时直接完成
        elif action == "complete":
            logger.info("\n=== 开始完成研报阶段 ===")
        return action

class SearchInfo(Node):  # 信息搜索节点
    def prep(self, shared):
        return shared.get("search_terms", [])

    def exec(self, search_terms):
        all_results = []
        total = len(search_terms)
        for i, term in enumerate(search_terms, 1):
            logger.info(f"\n搜索关键词 ({i}/{total}): {term}")
            results = search_web(term)
            logger.info(f"找到 {len(list(results))} 条相关信息")
            all_results.append({"term": term, "results": results})
            time.sleep(15)  # 避免请求过快
        return all_results

    def post(self, shared, prep_res, exec_res):
        context_list = shared.get("context", [])
        context_list.extend(exec_res)
        shared["context"] = context_list
        logger.info("\n信息搜索完成，返回决策节点...")
        return "search_done"

class GenerateSection(Node):  # 章节生成节点
    def prep(self, shared):
        return (
            shared.get("industry"),
            shared.get("current_section", {}),
            shared.get("context", [])
        )

    def exec(self, inputs):
        industry, section, context = inputs
        # 安全检查section格式
        if not isinstance(section, dict) or 'name' not in section:
            logger.error(f"章节信息格式错误: {section}")
            return {
                "name": "错误章节",
                "content": f"章节信息格式错误: {section}"
            }
        logger.info(f"\n开始生成 {section['name']} 章节...")
        context_str = yaml.dump(context, allow_unicode=True)
        focus = section.get('focus', '综合分析')
        prompt = f"""
行业：{industry}
章节：{section['name']}
重点：{focus}
参考资料：{context_str}

请生成一个专业、详实的研报章节。要求：
1. 数据支撑充分
2. 逻辑严谨
3. 分析深入
4. 结构清晰
5. 语言专业
"""
        content = call_llm(prompt)
        logger.info(f"章节 {section['name']} 生成完成!")
        logger.info(f"内容长度: {len(content)} 字符")
        logger.info(f"内容预览: {content[:100]}...")
        return {
            "name": section["name"],
            "content": content
        }

    def post(self, shared, prep_res, exec_res):
        sections = shared.get("generated_sections", [])
        generated_names = [s.get('name', '') for s in sections]
        if exec_res['name'] in generated_names:
            logger.warning(f"章节 {exec_res['name']} 已生成，跳过")
            return "continue"
        sections.append(exec_res)
        shared["generated_sections"] = sections
        logger.info(f"\n章节 {exec_res['name']} 已添加到生成列表")
        logger.info(f"当前已生成 {len(sections)} 个章节")
        logger.info("\n返回决策节点，继续分析下一步...")
        return "continue"

class CompleteReport(Node):  # 研报完成节点
    def prep(self, shared):
        return (
            shared.get("industry"),
            shared.get("generated_sections", [])
        )

    def exec(self, inputs):
        industry, sections = inputs
        logger.info(f"\n=== 开始整合最终研报 ===")
        # 整合所有章节内容
        content = f"# {industry}行业研究报告\n\n"
        for section in sections:
            logger.info(f"添加章节: {section['name']}")
            content += f"\n## {section['name']}\n\n{section['content']}\n"
        return content

    def post(self, shared, prep_res, exec_res):
        logger.info(f"\n=== 研报生成完成！===")
        shared["report"] = exec_res
        return exec_res  # 返回研报内容而不是None

def call_llm(prompt: str) -> str:
    try:
        logger.info("🤖 正在调用LLM...")
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"✅ LLM调用成功，返回长度: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"❌ LLM调用失败: {e}")
        return ""

def search_web(term: str, force_refresh: bool = False):
    # with DDGS() as ddgs:
    #     results = ddgs.text(keywords=term, region="cn-zh", max_results=20)
    multi_engine = SearchEngine()
    results = multi_engine.search(term, max_results=10, force_refresh=force_refresh)
    return results

def test_workflow():
    """测试工作流基本功能"""
    logger.info("🧪 开始测试工作流...")
    
    # 测试LLM调用
    test_prompt = "请简单回答：1+1等于几？"
    result = call_llm(test_prompt)
    if result:
        logger.info(f"✅ LLM测试成功: {result}")
    else:
        logger.error("❌ LLM测试失败")
        return False
    
    # 测试搜索功能
    try:
        results = search_web("测试搜索", force_refresh=True)
        logger.info(f"✅ 搜索测试成功，找到 {len(list(results))} 条结果")
    except Exception as e:
        logger.error(f"❌ 搜索测试失败: {e}")
        return False
    
    logger.info("✅ 所有测试通过")
    return True

"""
示例用法
"""
if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='行业研究工作流')
    parser.add_argument('--industry_name',default="中国智能服务机器人产业", help='目标行业名称（如：智能风控&大数据征信服务）')
    parser.epilog = '示例：python run_industry_research_report.py --industry_name "智能风控&大数据征信服务"'
    parser.add_argument('--output-file', default=None,
                       help='输出文件名 (默认: 自动生成带时间戳的文件名)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='最大迭代次数，防止无限循环')
    parser.add_argument('--force-refresh', action='store_true',
                       help='强制刷新搜索缓存')
    parser.add_argument('--test', action='store_true',
                       help='仅运行测试，不执行完整工作流')
    
    args = parser.parse_args()
    
    # 如果只是测试
    if args.test:
        test_workflow()
        exit(0)
    
    # 构建工作流
    research = IndustryResearchFlow()
    search = SearchInfo()
    generate = GenerateSection()
    complete = CompleteReport()
    
    # 设置转换关系
    research - "search" >> search
    research - "generate" >> generate
    research - "complete" >> complete
    search - "search_done" >> research
    generate - "continue" >> research
    
    # 运行工作流
    flow = Flow(start=research)
    shared_state = {
        "industry": args.industry_name,
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "force_refresh": args.force_refresh
    }
    
    logger.info(f"🚀 开始行业研究工作流")
    logger.info(f"📊 目标行业: {args.industry_name}")
    logger.info(f"🔄 最大迭代次数: {args.max_iterations}")
    if args.force_refresh:
        logger.info("🔄 强制刷新搜索缓存")
    
    result = flow.run(shared_state)
    
    # 保存结果
    if result:
        output_filename = f"Industry_Research_Report.md"
        #copy to reports
        os.makedirs("reports", exist_ok=True)
        shutil.copy(output_filename, f"reports/Industry_Research_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        # 删除原始文件
        if os.path.exists(output_filename):
            os.remove(output_filename)
        # 删除word文件
        if os.path.exists(f"Industry_Research_Report.docx"):
            os.remove(f"Industry_Research_Report.docx")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(result)
        
        # 创建文档转换实例
        pipeline = DocumentConversionPipeline()
        
        # 运行文档转换流程
        result = pipeline.run_conversion(output_filename)
        logger.info(f"\n✅ 行业研报生成完成！")
        logger.info(f"📁 输出文件: {output_filename}")
    else:
        logger.error("❌ 研报生成失败")
        logger.error("可能的原因：")
        logger.error("1. LLM调用失败")
        logger.error("2. YAML解析失败")
        logger.error("3. 工作流逻辑错误")
        logger.error("建议运行 --test 参数进行诊断")
