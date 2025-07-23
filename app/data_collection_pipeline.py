#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集流程
负责收集财务数据、公司信息、行业信息等，并将数据向量化存储到PostgreSQL数据库
"""

import os
import glob
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from app.data_analysis_agent.config.llm_config import LLMConfig
from app.data_analysis_agent.utils.llm_helper import LLMHelper
from app.utils.get_shareholder_info import get_shareholder_info, get_table_content
from app.utils.get_financial_statements import get_all_financial_statements, save_financial_statements_to_csv
from app.utils.identify_competitors import identify_competitors_with_ai
from app.utils.get_stock_intro import get_stock_intro, save_stock_intro_to_txt
from app.utils.search_engine import SearchEngine
from app.utils.rag_postgres import RAGPostgresHelper
from app.config.database_config import db_config

class DataCollectionPipeline:
    """数据收集流程类"""
    
    def __init__(self, target_company="商汤科技", target_company_code="00020", target_company_market="HK", search_engine="all"):
        # 配置日志记录
        self.setup_logging()
        
        # 环境变量与全局配置
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        self.logger.info(f"🔧 使用的模型: {self.model}")
        self.target_company = target_company
        self.target_company_code = target_company_code
        self.target_company_market = target_company_market
        
        # 搜索引擎配置
        self.search_engine = SearchEngine()
        if search_engine and search_engine != "all":
            self.logger.info(f"🔍 搜索引擎已配置为: {search_engine.upper()}")
        else:
            self.logger.info(f"🔍 搜索引擎默认全部使用")
        
        # 目录配置
        self.data_dir = "./download_financial_statement_files"
        self.company_info_dir = "./company_info"
        self.industry_info_dir = "./industry_info"
        
        # 创建必要的目录
        for dir_path in [self.data_dir, self.company_info_dir, self.industry_info_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # LLM配置
        self.llm_config = LLMConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.7,
            max_tokens=8192,
        )
        self.llm = LLMHelper(self.llm_config)
        
        # 初始化PostgreSQL RAG助手
        self.use_local_mode = False
        self.local_rag_backup_path = "local_rag_backup.json"
        try:
            self.logger.info("🔗 初始化PostgreSQL RAG助手...")
            self.rag_helper = RAGPostgresHelper(
                db_config=db_config.get_postgres_config(),
                rag_config=db_config.get_rag_config()
            )
            self.logger.info("✅ PostgreSQL RAG助手初始化成功")
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL RAG助手初始化失败: {e}")
            self.logger.warning("⚠️ 启用本地文件处理模式，所有数据仅保存到local_rag_backup.json。")
            self.use_local_mode = True
            self.rag_helper = None
            if not os.path.exists(self.local_rag_backup_path):
                with open(self.local_rag_backup_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
    
    def setup_logging(self):
        """配置日志记录"""
        # 创建logs目录
        os.makedirs("logs", exist_ok=True)
        
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/data_collection_{timestamp}.log"
        
        # 配置日志记录器
        self.logger = logging.getLogger('DataCollection')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 日志记录已启动，日志文件: {log_filename}")
    
    def _add_search_results(self, results, tag):
        """根据模式写入数据库或本地json"""
        if not self.use_local_mode and self.rag_helper:
            self.rag_helper.add_search_results(results, tag)
        else:
            try:
                with open(self.local_rag_backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = []
            for result in results:
                data.append({"tag": tag, "result": result})
            with open(self.local_rag_backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[本地模式] 数据已写入 {self.local_rag_backup_path}，tag={tag}")

    def _get_statistics(self):
        if not self.use_local_mode and self.rag_helper:
            return self.rag_helper.get_statistics()
        else:
            try:
                with open(self.local_rag_backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    "total_documents": len(data),
                    "total_chunks": len(data),
                    "last_updated": os.path.getmtime(self.local_rag_backup_path)
                }
            except Exception:
                return {"total_documents": 0, "total_chunks": 0, "last_updated": "无"}
    
    def collect_competitors(self):
        """收集竞争对手信息"""
        self.logger.info("🔍 识别竞争对手...")
        other_companies = identify_competitors_with_ai(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model,
            company_name=self.target_company
        )
        listed_companies = [company for company in other_companies if company.get('market') != "未上市"]
        
        # 将竞争对手信息存储到数据库
        competitors_text = f"目标公司: {self.target_company}\n竞争对手列表:\n"
        for company in listed_companies:
            competitors_text += f"- {company.get('name', '')} ({company.get('code', '')}) - {company.get('market', '')}\n"
        
        # 添加到RAG数据库
        self._add_search_results([{
            'title': f'{self.target_company}竞争对手分析',
            'description': competitors_text,
            'url': 'internal://competitors'
        }], f"{self.target_company}竞争对手")
        
        self.logger.info(f"✅ 识别到 {len(listed_companies)} 个竞争对手")
        return listed_companies
    
    def collect_financial_data(self, listed_companies):
        """收集财务数据"""
        all_companies = [(self.target_company, self.target_company_code, self.target_company_market)]
        
        # 添加竞争对手
        for company in listed_companies:
            company_name = company.get('name')
            company_code = company.get('code')
            market_str = company.get('market', '')
            
            if "A" in market_str:
                market = "A"
                if not (company_code.startswith('SH') or company_code.startswith('SZ')):
                    if company_code.startswith('6'):
                        company_code = f"SH{company_code}"
                    else:
                        company_code = f"SZ{company_code}"
            elif "港" in market_str:
                market = "HK"
            
            all_companies.append((company_name, company_code, market))
        
        # 收集所有公司的财务数据
        for company_name, company_code, market in all_companies:
            self.logger.info(f"📊 获取 {company_name}({market}:{company_code}) 的财务数据...")
            try:
                financials = get_all_financial_statements(
                    stock_code=company_code,
                    market=market,
                    period="年度",
                    verbose=False
                )
                
                # 保存到CSV文件
                save_financial_statements_to_csv(
                    financial_statements=financials,
                    stock_code=company_code,
                    market=market,
                    company_name=company_name,
                    period="年度",
                    save_dir=self.data_dir
                )
                
                # 将财务数据摘要存储到数据库
                if financials and isinstance(financials, dict) and len(financials) > 0:
                    financial_summary = f"{company_name}财务数据摘要:\n"
                    for statement_type, data in financials.items():
                        # 判断data为DataFrame时用data.empty，否则用len(data)
                        if hasattr(data, 'empty'):
                            if not data.empty:
                                financial_summary += f"{statement_type}: {len(data)}条记录\n"
                        elif data and len(data) > 0:
                            financial_summary += f"{statement_type}: {len(data)}条记录\n"
                    
                    self._add_search_results([{
                        'title': f'{company_name}财务数据',
                        'description': financial_summary,
                        'url': f'internal://financial/{company_code}'
                    }], f"{company_name}财务数据")
                
                self.logger.info(f"  ✅ {company_name} 财务数据收集完成")
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"  ❌ 获取 {company_name} 财务数据失败: {e}")
    
    def collect_company_info(self, listed_companies):
        """收集公司基础信息"""
        all_companies = [(self.target_company, self.target_company_code, self.target_company_market)]
        
        # 添加竞争对手
        for company in listed_companies:
            company_name = company.get('name')
            company_code = company.get('code')
            market_str = company.get('market', '')
            if "A" in market_str:
                market = "A"
                if not (company_code.startswith('SH') or company_code.startswith('SZ')):
                    if company_code.startswith('6'):
                        company_code = f"SH{company_code}"
                    else:
                        company_code = f"SZ{company_code}"
            elif "港" in market_str:
                market = "HK"
            all_companies.append((company_name, company_code, market))
        
        # 添加特定公司如百度
        all_companies.append(("百度", "09888", "HK"))
        
        for company_name, company_code, market in all_companies:
            self.logger.info(f"🏢 获取 {company_name}({market}:{company_code}) 的基础信息...")
            try:
                company_info = get_stock_intro(company_code, market=market)
                if company_info:
                    # 保存到文件
                    save_path = os.path.join(self.company_info_dir, f"{company_name}_{market}_{company_code}_info.txt")
                    save_stock_intro_to_txt(company_code, market, save_path)
                    
                    # 存储到数据库
                    self._add_search_results([{
                        'title': f'{company_name}公司介绍',
                        'description': company_info,
                        'url': f'internal://company/{company_code}'
                    }], f"{company_name}公司信息")
                    
                    self.logger.info(f"  ✅ {company_name} 基础信息收集完成")
                else:
                    self.logger.warning(f"  ⚠️ 未能获取到 {company_name} 的基础信息")
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"  ❌ 获取 {company_name} 基础信息失败: {e}")
    
    def collect_shareholder_info(self):
        """收集股东信息"""
        self.logger.info("👥 获取股东信息...")
        try:
            info = get_shareholder_info()
            shangtang_shareholder_info = info.get("tables")
            table_content = get_table_content(shangtang_shareholder_info)
            
            # 存储到数据库
            self._add_search_results([{
                'title': f'{self.target_company}股东结构',
                'description': table_content,
                'url': 'internal://shareholder'
            }], f"{self.target_company}股东信息")
            
            self.logger.info("✅ 股东信息收集完成")
            
        except Exception as e:
            self.logger.error(f"❌ 获取股东信息失败: {e}")
    
    def collect_industry_info(self, listed_companies):
        """收集行业信息"""
        self.logger.info("🔍 搜索行业信息...")
        all_companies = [self.target_company] + [company.get('name') for company in listed_companies]
        
        for company_name in all_companies:
            search_keywords = f"{company_name} 行业地位 市场份额 竞争分析 业务模式 发展战略"
            self.logger.info(f"  正在搜索: {search_keywords}")
            
            try:
                results = self.search_engine.search(search_keywords, 10)
                
                # 存储到数据库
                for result in results:
                    self._add_search_results([result], f"{company_name}行业信息")
                
                self.logger.info(f"  ✅ {company_name} 行业信息收集完成，共 {len(results)} 条结果")
                
                # 增加延迟避免请求过于频繁
                time.sleep(self.search_engine.delay * 2)
                
            except Exception as e:
                self.logger.error(f"  ❌ 搜索 {company_name} 行业信息失败: {e}")
        
        # 保存搜索结果到文件（备份）
        search_results_file = os.path.join(self.industry_info_dir, "all_search_results.json")
        with open(search_results_file, 'w', encoding='utf-8') as f:
            json.dump({company: [] for company in all_companies}, f, ensure_ascii=False, indent=2)
    
    def run_data_collection(self):
        """运行完整的数据收集流程"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 开始数据收集流程")
        self.logger.info("="*80)
        
        try:
            # 1. 收集竞争对手信息
            listed_companies = self.collect_competitors()
            
            # 2. 收集财务数据
            self.collect_financial_data(listed_companies)
            
            # 3. 收集公司基础信息
            self.collect_company_info(listed_companies)
            
            # 4. 收集股东信息
            self.collect_shareholder_info()
            
            # 5. 收集行业信息
            self.collect_industry_info(listed_companies)
            
            # 6. 显示数据库统计
            stats = self._get_statistics()
            self.logger.info(f"\n📊 数据库统计:")
            self.logger.info(f"  总文档数: {stats['total_documents']}")
            self.logger.info(f"  总文档块: {stats['total_chunks']}")
            self.logger.info(f"  最新更新: {stats.get('last_updated', '未知')}")
            
            self.logger.info("\n✅ 数据收集流程完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 数据收集流程失败: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据收集流程')
    parser.add_argument('--company', default='商汤科技', help='目标公司名称')
    parser.add_argument('--code', default='00020', help='股票代码')
    parser.add_argument('--market', default='HK', help='市场代码')
    parser.add_argument('--search-engine', choices=['ddg', 'sogou', 'all'], default='all',
                       help='搜索引擎选择')
    
    args = parser.parse_args()
    
    # 创建数据收集实例
    pipeline = DataCollectionPipeline(
        target_company=args.company,
        target_company_code=args.code,
        target_company_market=args.market,
        search_engine=args.search_engine
    )
    
    # 运行数据收集流程
    success = pipeline.run_data_collection()
    
    if success:
        print("\n🎉 数据收集流程执行完毕！")
        print("📊 所有数据已向量化存储到PostgreSQL数据库")
    else:
        print("\n❌ 数据收集流程执行失败！")


if __name__ == "__main__":
    main() 