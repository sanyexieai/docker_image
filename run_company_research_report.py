#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研报生成流程
基于PostgreSQL数据库中的数据，生成深度研报并输出为markdown格式
"""
import logging
import os
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv


from app.company.agent.agent_factory import OutlineAgentFactory, OutlineAgentType
from app.company.model.report_info import ReportInfo

from app.company.utils.content_convert import ContentConvert
from app.config.database_config import db_config
from app.data_collection_pipeline import DataCollectionPipeline
from app.document_conversion_pipeline import DocumentConversionPipeline
from app.llm.config import LLMConfig
from app.llm.llm_helper import LLMHelper
from app.marco.tools.document_processing.pure_python_converter import convert_md_to_docx_pure_python

from app.utils.rag_postgres import RAGPostgresHelper


class ReportGenerationConfig:
    """研报生成配置类"""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "8192"))
        self.data_dir = Path("./download_financial_statement_files")
        self.logs_dir = Path("logs")
        self.max_discuss_rounds = int(os.getenv("MAX_DISCUSS_ROUNDS", "1"))

    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")
        return True


class ReportGenerationPipeline:
    """研报生成流程类"""

    def __init__(self,
                 target_company: str = "4Paradigm",
                 target_company_code: str = "06682",
                 target_company_market: str = "HK"):
        """
        初始化研报生成流程

        Args:
            target_company: 目标公司名称
            target_company_code: 公司股票代码
            target_company_market: 市场代码
        """
        self.config = ReportGenerationConfig()
        self.config.validate()

        self.target_company = target_company
        self.target_company_code = target_company_code
        self.target_company_market = target_company_market

        # 延迟初始化
        self._logger: Optional[logging.Logger] = None
        self._llm: Optional[LLMHelper] = None
        self._rag_helper: Optional[RAGPostgresHelper] = None
        self._agents: Dict[str, Any] = {}
        self.report_info: Optional[ReportInfo] = None

        # 初始化基础组件
        self._setup_logging()

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if self._logger is None:
            self._setup_logging()
        return self._logger

    @property
    def llm(self) -> LLMHelper:
        """获取LLM助手"""
        if self._llm is None:
            self._setup_llm()
        return self._llm

    @property
    def rag_helper(self) -> 'RAGPostgresHelper':
        """获取RAG助手，若未初始化则返回None"""
        if self._rag_helper is None:
            self.logger.warning("RAG助手未初始化，相关功能将被跳过。")
        return self._rag_helper

    def _setup_logging(self) -> None:
        """配置日志记录"""
        self.config.logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = self.config.logs_dir / f"report_generation_{timestamp}.log"

        self._logger = logging.getLogger(f'ReportGeneration_{id(self)}')
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
        self._logger.info(f"📝 日志记录已启动，日志文件: {log_filename}")

    def _setup_llm(self) -> None:
        """初始化LLM配置"""
        try:
            llm_config = LLMConfig(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            self._llm = LLMHelper(llm_config)
            self.logger.info(f"🔧 LLM初始化成功，使用模型: {self.config.model}")
        except Exception as e:
            self.logger.error(f"❌ LLM初始化失败: {e}")
            raise

    def _setup_rag_helper(self) -> None:
        """初始化PostgreSQL RAG助手"""
        try:
            self.logger.info("🔗 初始化PostgreSQL RAG助手...")
            self._rag_helper = RAGPostgresHelper(
                db_config=db_config.get_postgres_config(),
                rag_config=db_config.get_rag_config()
            )
            self.logger.info("✅ PostgreSQL RAG助手初始化成功")
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL RAG助手初始化失败: {e}")
            self._rag_helper = None  # 失败时设为None，不抛出异常
            # 不再raise，主流程继续

    def _setup_agents(self) -> None:
        """初始化所有代理"""
        if self._agents:
            return

        try:
            factory = OutlineAgentFactory()
            self._agents = {
                'section_edit': factory.create_agent(
                    OutlineAgentType.PART_GENERATOR_PART, self.logger, self.llm
                ),
                'section_opinion': factory.create_agent(
                    OutlineAgentType.PART_OPINION_GENERATOR_PART, self.logger, self.llm
                ),
                'abstract': factory.create_agent(
                    OutlineAgentType.PART_ABSTRACT_GENERATOR_PART, self.logger, self.llm
                ),
                'outline': factory.create_agent(
                    OutlineAgentType.OUTLINE_GENERATOR_PART, self.logger, self.llm
                ),
                'outline_opinion': factory.create_agent(
                    OutlineAgentType.OUTLINE_OPINION_GENERATOR_PART, self.logger, self.llm
                )
            }
            self.logger.info("✅ 所有代理初始化成功")
        except Exception as e:
            self.logger.error(f"❌ 代理初始化失败: {e}")
            raise

    def _load_outline_template(self, template_name: str = "default") -> List[Dict[str, Any]]:
        """加载大纲模板"""
        # 这里可以从外部文件加载，现在先使用默认模板
        default_template = [
            {
                "part_num": "1",
                "part_title": "1. 宏观环境分析",
                "part_title_type": "章",
                "part_desc": "系统分析全球AI产业政策环境、技术演进趋势及市场需求变化，重点解读中国新型基础设施建设对计算机视觉行业的战略支持，阐明商汤科技发展的时代背景。",
                "part_content_type": "现状描述",
                "part_key_output": "明确各国AI战略差异及中国政策红利的具体表现，量化新基建投资中对计算机视觉的投入占比，建立宏观分析框架。",
                "part_data_source": "国务院AI发展规划、IDC技术支出报告、新基建投资白皮书",
                "part_importance": "核心",
                "part_length_ratio": "12%"
            },
            {
                "part_num": "1.1",
                "part_title": "1.1 全球AI政策图谱",
                "part_title_type": "节",
                "part_desc": "对比分析中美欧AI发展战略差异，特别关注芯片出口管制对AI算力供给的影响机制。",
                "part_content_type": "格局分析",
                "part_key_output": "绘制主要国家AI监管沙盒实施进度图，量化GPU供应风险指数。",
                "part_data_source": "美国商务部出口管制清单、欧盟AI法案文本",
                "part_importance": "核心",
                "part_length_ratio": "5%"
            },
            {
                "part_num": "1.2",
                "part_title": "1.2 中国新基建赋能分析",
                "part_title_type": "节",
                "part_desc": "拆解智慧城市、工业互联网等国家工程对计算机视觉技术的需求释放路径。",
                "part_content_type": "数据论证",
                "part_key_output": "测算政府项目在商汤收入结构的占比趋势，评估政策延续性风险。",
                "part_data_source": "财政部专项债投向数据、政府采购网中标记录",
                "part_importance": "次要",
                "part_length_ratio": "7%"
            },
            {
                "part_num": "2",
                "part_title": "2. 行业竞争格局",
                "part_title_type": "章",
                "part_desc": "深度解构AI视觉产业价值链分布，建立\"技术-场景-数据\"三维评估模型，分析四小龙差异化竞争策略。",
                "part_content_type": "格局分析",
                "part_key_output": "绘制产业价值分配热力图，识别商汤在算法层-平台层-应用层的卡位优势。",
                "part_data_source": "CB Insights产业链图谱、企业专利地图",
                "part_importance": "核心",
                "part_length_ratio": "18%"
            },
            {
                "part_num": "2.1",
                "part_title": "2.1 技术路线竞争",
                "part_title_type": "节",
                "part_desc": "对比CNN、Transformer等架构在视觉领域的应用效率，评估商汤SenseCore平台的迭代能力。",
                "part_content_type": "价值论证",
                "part_key_output": "建立算法性能基准测试对比表，量化商汤模型在ImageNet等数据集的表现优势。",
                "part_data_source": "arXiv论文库、Kaggle竞赛数据",
                "part_importance": "核心",
                "part_length_ratio": "8%"
            },
            {
                "part_num": "2.2",
                "part_title": "2.2 商业化路径对比",
                "part_title_type": "节",
                "part_desc": "分析各厂商在智慧城市、自动驾驶等场景的落地效率，测算项目ROI差异。",
                "part_content_type": "数据论证",
                "part_key_output": "构建客户获取成本(CAC)与生命周期价值(LTV)对比矩阵。",
                "part_data_source": "招投标数据库、行业专家访谈",
                "part_importance": "次要",
                "part_length_ratio": "10%"
            },
            {
                "part_num": "3",
                "part_title": "3. 公司核心竞争力",
                "part_title_type": "章",
                "part_desc": "从人才密度、算力储备、数据资产三个维度构建评估体系，解析商汤科技护城河构成。",
                "part_content_type": "价值论证",
                "part_key_output": "量化顶级AI科学家占比、GPU集群算力规模、标注数据量等核心指标行业排名。",
                "part_data_source": "LinkedIn人才图谱、超算中心运营数据",
                "part_importance": "核心",
                "part_length_ratio": "20%"
            },
            {
                "part_num": "3.1",
                "part_title": "3.1 人才战略分析",
                "part_title_type": "节",
                "part_desc": "研究商汤与中科院、香港高校的联合实验室运作机制，评估其人才造血能力。",
                "part_content_type": "现状描述",
                "part_key_output": "计算院士/IEEE Fellow级专家占比，分析关键技术人才流失风险。",
                "part_data_source": "学术合作论文统计、竞业限制诉讼记录",
                "part_importance": "次要",
                "part_length_ratio": "7%"
            },
            {
                "part_num": "3.2",
                "part_title": "3.2 算力基础设施",
                "part_title_type": "节",
                "part_desc": "审计商汤自建AI计算中心运营效率，分析AIDC的利用率与能耗比。",
                "part_content_type": "数据论证",
                "part_key_output": "建立算力成本对标模型，对比AWS等公有云服务的经济性差异。",
                "part_data_source": "数据中心PUE报告、GPU采购合同",
                "part_importance": "核心",
                "part_length_ratio": "13%"
            },
            {
                "part_num": "4",
                "part_title": "4. 财务健康度评估",
                "part_title_type": "章",
                "part_desc": "采用\"收入质量-研发效能-现金流韧性\"三维分析框架，穿透式解读财务报表关键指标。",
                "part_content_type": "数据论证",
                "part_key_output": "构建研发资本化率调整后的真实盈利模型，识别应收账款周转风险点。",
                "part_data_source": "年报附注、同行资本化政策对比",
                "part_importance": "核心",
                "part_length_ratio": "15%"
            },
            {
                "part_num": "4.1",
                "part_title": "4.1 收入结构透视",
                "part_title_type": "节",
                "part_desc": "分解软件授权、解决方案、平台服务等业务线的毛利率差异，追踪政府项目回款周期。",
                "part_content_type": "现状描述",
                "part_key_output": "绘制收入来源热力图，标注账期超过180天的重大合同明细。",
                "part_data_source": "客户集中度披露、账龄分析表",
                "part_importance": "次要",
                "part_length_ratio": "6%"
            },
            {
                "part_num": "4.2",
                "part_title": "4.2 研发效能审计",
                "part_title_type": "节",
                "part_desc": "分析研发支出资本化政策合理性，计算专利转化率与人均研发产出。",
                "part_content_type": "价值论证",
                "part_key_output": "建立研发投入-专利质量-商业产出关联模型，对比国际巨头效率差距。",
                "part_data_source": "专利引用指数、研发人员KPI数据",
                "part_importance": "核心",
                "part_length_ratio": "9%"
            },
            {
                "part_num": "5",
                "part_title": "5. 估值建模",
                "part_title_type": "章",
                "part_desc": "设计三阶段DCF模型，结合PSM法对平台型业务估值，设置技术突破/地缘政治等情景测试。",
                "part_content_type": "价值论证",
                "part_key_output": "生成乐观/中性/悲观三种情景下的估值区间，标注关键价值驱动因子敏感性。",
                "part_data_source": "WACC计算表、可比公司交易乘数",
                "part_importance": "核心",
                "part_length_ratio": "15%"
            },
            {
                "part_num": "5.1",
                "part_title": "5.1 现金流折现模型",
                "part_title_type": "节",
                "part_desc": "基于商汤四大业务板块拆分收入预测，动态调整研发费用资本化影响。",
                "part_content_type": "数据论证",
                "part_key_output": "输出分业务线自由现金流预测表，标注核心假设变动阈值。",
                "part_data_source": "细分业务增长率指引、资本开支计划",
                "part_importance": "次要",
                "part_length_ratio": "8%"
            },
            {
                "part_num": "5.2",
                "part_title": "5.2 场景压力测试",
                "part_title_type": "节",
                "part_desc": "模拟美国扩大芯片禁运、核心团队离职等极端情况对公司价值的冲击幅度。",
                "part_content_type": "价值论证",
                "part_key_output": "量化黑天鹅事件对估值的影响系数，制定风险对冲策略建议。",
                "part_data_source": "供应链替代方案评估、关键人保险数据",
                "part_importance": "核心",
                "part_length_ratio": "7%"
            },
            {
                "part_num": "6",
                "part_title": "6. 投资决策建议",
                "part_title_type": "章",
                "part_desc": "结合技术成熟度曲线与估值波段，制定差异化建仓策略，明确关键观测指标。",
                "part_content_type": "价值论证",
                "part_key_output": "生成投资决策矩阵，标注技术突破、政策催化等事件驱动交易时点。",
                "part_data_source": "机构持仓变动数据、期权隐含波动率",
                "part_importance": "核心",
                "part_length_ratio": "10%"
            },
            {
                "part_num": "6.1",
                "part_title": "6.1 配置策略",
                "part_title_type": "节",
                "part_desc": "根据资金属性设计VC/PE/二级市场不同参与方式的最优进入路径。",
                "part_content_type": "现状描述",
                "part_key_output": "绘制不同风险偏好下的仓位配置图谱，标注止损触发条件。",
                "part_data_source": "历史波动率分析、贝塔系数测算",
                "part_importance": "次要",
                "part_length_ratio": "5%"
            },
            {
                "part_num": "6.2",
                "part_title": "6.2 监测框架",
                "part_title_type": "节",
                "part_desc": "建立包含技术指标、财务指标、政策指标的三维预警体系。",
                "part_content_type": "价值论证",
                "part_key_output": "列出季度必须跟踪的10项关键绩效指标(KPI)及其阈值范围。",
                "part_data_source": "管理层指引、行业先行指标",
                "part_importance": "核心",
                "part_length_ratio": "5%"
            },
            {
                "part_num": "7",
                "part_title": "7. 风险全景图",
                "part_title_type": "章",
                "part_desc": "采用FMEA(失效模式与影响分析)方法，系统识别技术路线、商业变现、公司治理等维度的潜在风险。",
                "part_content_type": "现状描述",
                "part_key_output": "制作风险热力图，标注发生概率与影响程度的乘积排序。",
                "part_data_source": "历史风险事件库、专家概率评估",
                "part_importance": "次要",
                "part_length_ratio": "10%"
            }
        ]
        return default_template

    def _load_text_template(self, template_name: str = "default") -> List[str]:

        default_template = [
            "## 1. 宏观环境分析\n\n人工智能行业作为全球科技竞争的战略制高点，其发展受到政策支持、技术进步、经济环境和社会需求等多重宏观因素的深刻影响。本部分将从四个维度系统分析当前人工智能产业发展的宏观驱动因素。\n\n### 1.1 政策支持持续加码\n全球主要经济体已将人工智能上升为国家战略。中国\"十四五\"规划明确将人工智能列为前沿科技领域的七大重点产业之一，2023年中央财政科技支出同比增长7.1%[1]，其中AI相关领域获得重点倾斜。美国通过《芯片与科学法案》拨款520亿美元支持包括AI芯片在内的关键技术研发[2]。欧盟《人工智能法案》则构建了全球首个全面的AI监管框架。政策红利为行业提供了明确的制度保障和发展方向。\n\n### 1.2 技术突破加速商业化\n2023年全球AI领域专利申请量达13.8万件，同比增长26%[3]。Transformer架构的持续优化推动大模型参数量突破万亿级，GPT-4等模型的商业化落地使得AI技术渗透率快速提升。据IDC数据，全球AI算力投资2023年达1540亿美元，预计2026年将突破3000亿美元[4]。芯片制程进步与算法效率提升共同推动技术成本曲线下移，行业进入规模商用临界点。\n\n### 1.3 经济环境创造新需求\n后疫情时代企业数字化转型加速，全球企业软件支出2023年增长12.4%至8560亿美元[5]。制造业智能化改造催生工业视觉检测、预测性维护等AI应用场景，金融业智能风控渗透率已达43%[6]。经济结构调整过程中，AI技术成为企业降本增效的核心工具，预计到2025年将为全球经济额外贡献15.7万亿美元产出[7]。\n\n### 1.4 社会认知度显著提升\n消费者对AI产品的接受度持续提高，2023年全球智能语音助手用户突破20亿，中国城市家庭智能设备渗透率达62%[8]。医疗影像识别、智能教育等民生领域应用显著改善公共服务效率。同时，AI伦理和隐私保护意识增强，推动行业向负责任发展方向演进。\n\n综合来看，当前人工智能行业正处于政策、技术、经济和社会四重利好叠加的发展窗口期。商汤科技作为计算机视觉领域的领军企业，需要把握住基础研究突破与产业落地加速的战略机遇，在持续变化的宏观环境中保持技术领先优势。",
            "## 2. 行业分析",
            "### 2.1 计算机视觉细分市场\n\n计算机视觉作为人工智能的核心技术领域，已成为商汤科技（股票代码：0020.HK）最具竞争力的主营业务板块（数据来源：同花顺-主营介绍[2]）。本部分将从技术应用场景、市场规模及增长动力三个维度展开深度分析。\n\n#### 技术应用场景分析\n商汤科技的计算机视觉技术已实现四大商业化落地场景：\n1. **智慧城市**：覆盖全国30+省级行政区，部署超10万路智能摄像头，实现交通流量分析、异常行为识别等功能（数据来源：公司2022年报）；\n2. **智能手机**：为OPPO、vivo等头部厂商提供3D传感解决方案，2022年相关收入达12.4亿港元（数据来源：东方财富-港股-财务报表[1]）；\n3. **医疗影像**：在CT/MRI影像分析领域市占率达17%，合作医院超500家；\n4. **工业质检**：在3C制造领域实现缺陷识别准确率99.2%，年处理图像超50亿张。\n\n#### 市场规模量化分析\n根据自动化采集的行业数据显示：\n- 2022年全球计算机视觉市场规模达562亿美元，中国占比38%；\n- 商汤科技在中国计算机视觉软件市场的份额为22.1%，位列行业第一；\n- 细分领域增速差异显著：智慧城市（年复合增长率18.7%）<工业质检（31.2%）<医疗影像（41.5%）。\n\n#### 增长动力评估\n未来3-5年的核心驱动力包括：\n1. **政策支持**：国家\"十四五\"规划明确将计算机视觉列入新一代AI重点发展目录；\n2. **技术迭代**：Transformer架构的应用使图像识别准确率提升至98.3%；\n3. **成本下降**：GPU算力成本年均下降27%，推动商业化落地速度；\n4. **新兴场景**：元宇宙相关AR/VR应用预计带来200亿元增量市场。\n\n#### 商业化潜力预测\n基于财务模型测算（数据来源：东方财富-港股-财务报表[1]）：\n| 应用场景   | 2023E收入(亿港元) | 2025E收入(亿港元) | CAGR   |\n|------------|-------------------|-------------------|--------|\n| 智慧城市   | 28.5              | 39.2              | 17.2%  |\n| 智能手机   | 15.7              | 18.9              | 9.6%   |\n| 医疗影像   | 6.3               | 12.8              | 42.6%  |\n| 工业质检   | 4.1               | 9.5               | 52.3%  |\n\n（注：表中数据经蒙特卡洛模拟验证，置信区间95%）\n\n风险提示：需关注中美技术脱钩对GPU供应链的影响，以及医疗领域数据合规要求的持续升级。",
            "## 3. 公司基本面\n\n### 3.1 发展历程与战略定位\n商汤科技（股票代码：0020.HK）成立于2014年，是中国领先的人工智能软件公司，专注于计算机视觉和深度学习技术的研发与应用。公司以\"坚持原创，让AI引领人类进步\"为使命，通过SenseCore商汤AI大装置为核心基础设施，构建了覆盖智慧商业、智慧城市、智慧生活、智能汽车四大业务板块的完整AI生态体系[2]。\n\n### 3.2 股权结构与公司治理\n截至最新披露，公司主要股东包括：\n- 创始人团队持股占比23.5%\n- 阿里巴巴集团通过淘宝中国持股7.3%\n- 软银愿景基金持股14.2%\n- 其他机构投资者合计持股42.6%\n- 公众流通股占比12.4%[3]\n\n公司采用同股不同权架构，创始人团队通过特别投票权股份保持对公司的控制权。董事会由9名成员组成，其中独立非执行董事4名，符合港交所公司治理要求[3]。\n\n### 3.3 业务布局与收入结构\n公司主营业务分为四大板块（数据来源：同花顺-主营介绍[2]）：\n1. 智慧商业：为金融、零售、制造等行业提供AI解决方案，2022年收入占比32%\n2. 智慧城市：面向政府客户的城市管理AI平台，收入占比41% \n3. 智慧生活：涵盖手机、AR/VR等消费级AI应用，收入占比18%\n4. 智能汽车：自动驾驶和车舱交互解决方案，收入占比9%\n\n（注：此处应插入业务板块收入结构图，但财务研报汇总内容中未提供相关图片路径）\n\n### 3.4 财务表现与核心指标\n根据东方财富港股财务报表数据[1]：\n- 2022年营业收入38.1亿元人民币，同比增长12.3%\n- 毛利率达69.2%，较上年提升1.8个百分点\n- 研发投入28.6亿元，占收入比重75.1%\n- 净亏损收窄至60.5亿元，同比减少13.7%\n\n### 3.5 核心竞争优势\n1. 技术壁垒：拥有8,000多项AI相关专利，全球排名前列\n2. 基础设施优势：建成亚洲最大AI计算平台之一（算力达3.74 exaFLOPS）\n3. 商业化能力：已服务超过1,200家客户，包括120家《财富》500强企业\n4. 人才储备：研发人员占比67%，其中博士学历超过300人\n\n### 3.6 战略发展方向\n公司近期重点布局：\n- 加强AI大模型研发，推出\"日日新SenseNova\"大模型体系\n- 拓展海外市场，已在东南亚、中东等地建立业务\n- 深化行业应用，重点突破医疗、教育等垂直领域\n- 推进AI技术标准化，参与制定30余项国际/国家标准",
            "## 4. 公司深度分析\n\n### 4.1 商业模式解构\n商汤科技采用\"1（基础研究）+4（行业应用）+X（生态拓展）\"的商业模式架构[2]：\n1. **AI基础设施层**：通过SenseCore AI大装置提供算力支撑，2022年总算力规模突破4.91 exaFLOPS，年复合增长率达86%[1]\n2. **行业应用层**：覆盖智慧商业（营收占比32%）、智慧城市（28%）、智慧生活（25%）、智能汽车（15%）四大板块[1]\n3. **生态扩展层**：已建立包含800+合作伙伴的开发者生态，累计产生3.7万个商业模型[2]\n\n### 4.2 技术壁垒评估\n核心竞争优势体现在三个维度：\n- **专利储备**：截至2023H1全球累计专利资产12,700件，其中发明专利占比91%，在计算机视觉领域专利数量全球第一（数据来源：IPlytics统计）\n- **研发投入**：2022年研发支出41.2亿元，占营收比重达145%，研发人员占比67%（数据来源：东方财富-港股-财务报表[1]）\n- **算法效能**：在ImageNet等权威测试中保持95.3%的识别准确率，较行业平均水平高出7.2个百分点\n\n### 4.3 研发能力分析\n采用三级研发体系：\n1. **基础研究院**：9个实验室专注原创算法突破，论文被CVPR等顶会收录量连续5年居全球前3\n2. **工程创新中心**：实现算法-芯片协同优化，推理能耗较行业基准降低40%\n3. **产业研究院**：2022年新增行业解决方案47个，商业化转化周期缩短至8.3个月[1]\n\n### 4.4 商业化路径验证\n关键进展指标：\n- **标杆案例**：智慧城市项目已落地140+城市，智能汽车业务签约车企增至30家[2]\n- **收入结构**：2022年标准化产品收入占比提升至39%，较2020年提升21个百分点[1]\n- **客户质量**：年度合约价值超千万的客户达112家，复购率维持82%高位（数据来源：公司年报）\n\n### 4.5 长期竞争力研判\n差异化优势来源：\n1. **数据飞轮效应**：日均处理数据量达7.3PB，形成行业最大视觉数据库\n2. **政企协同网络**：参与制定16项国家AI标准，承担7个国家级重大科研项目\n3. **资本储备优势**：现金及等价物余额182亿元，可支撑5年以上高强度研发[1]\n\n风险提示：技术迭代风险（新一代多模态模型冲击）、行业政策风险（数据安全法规趋严）、商业化不及预期风险（2022年经调整净亏损59.3亿元[1]）",
            "## 5. 财务分析\n\n### 5.1 收入增长分析  \n根据东方财富港股财务报表数据[1]，商汤科技2022年实现营业收入38.08亿元，同比增长12.4%。分业务板块看：  \n- 智慧商业板块收入20.53亿元（占比53.9%），同比增长9.7%  \n- 智慧城市板块收入11.62亿元（占比30.5%），同比增长18.2%  \n- 智慧生活板块收入5.93亿元（占比15.6%），同比增长6.3%  \n\n收入增长主要受益于企业数字化转型加速（数据来源：同花顺主营介绍[2]），但增速较2021年的31.4%明显放缓，反映宏观经济压力对AI解决方案采购的影响。\n\n### 5.2 盈利能力分析  \n毛利率呈现结构性变化：  \n- 综合毛利率从2021年的69.7%下降至2022年的66.2%  \n- 硬件交付占比提升导致智慧城市业务毛利率下降5.2个百分点  \n- 研发人员薪酬上涨使智慧生活业务毛利率下降3.8个百分点  \n\n净利润持续为负，2022年经调整净亏损14.18亿元，亏损率37.2%，较2021年扩大2.1个百分点。主要由于：  \n1) 研发费用率达45.3%（17.25亿元）  \n2) 销售费用同比增长23.6%至8.72亿元  \n\n### 5.3 研发投入分析  \n研发支出呈现\"绝对额上升、占比下降\"特征：  \n- 2022年研发投入17.25亿元，同比增长11.2%  \n- 研发费用率从2021年的47.3%降至45.3%  \n- 研发人员数量达3,819人（占员工总数68%），人均研发支出45.2万元  \n\n重点投向生成式AI（如日日新大模型）、自动驾驶等前沿领域（数据来源：同花顺主营介绍[2]）。\n\n### 5.4 现金流状况  \n经营性现金流持续承压：  \n- 2022年经营性现金净流出16.24亿元，较2021年扩大28.6%  \n- 应收账款周转天数从2021年的187天增至213天  \n- 期末现金及等价物余额58.37亿元，按当前烧钱速度可维持3.6年运营  \n\n### 5.5 财务健康度评估  \n基于Z-score模型测算得分为1.2，处于警戒区间。关键风险点：  \n1) 流动比率1.3，短期偿债压力显著  \n2) 资产负债率攀升至48.7%  \n3) 无形资产占总资产比例达34.2%  \n\n### 5.6 未来三年预测  \n基于ARIMA模型预测：  \n| 指标   | 2023E | 2024E | 2025E |  \n|--------|-------|-------|-------|  \n| 营收(亿)| 42.7  | 48.3  | 54.6  |  \n| 毛利率 | 64.5% | 65.8% | 67.2% |  \n| 净亏损 | 12.4  | 9.8   | 6.3   |  \n\n关键假设：1) 年复合增长率12.7%；2) 2025年实现盈亏平衡；3) 政府补贴维持每年3-4亿元水平。",
            "## 6. 行业对比\n\n### 6.1 竞争格局概述\n商汤科技作为全球领先的人工智能平台公司，主要竞争对手包括国际巨头（如Google DeepMind、Meta AI）和国内头部企业（如旷视科技、云从科技、依图科技）。根据IDC最新数据显示，2023年中国计算机视觉应用市场份额中，商汤以22.3%的市占率保持首位，但较2021年的27.1%有所下滑[1]。\n\n### 6.2 核心技术对比\n#### 6.2.1 研发投入强度\n- 商汤科技：2023年研发支出达32.4亿元，占营收比重58.7%（数据来源：东方财富-港股-财务报表[1]）\n- 主要竞争对手：\n  - 旷视科技：研发占比49.2%\n  - 云从科技：研发占比52.1%\n  - Google DeepMind：研发投入绝对值超200亿元（按汇率折算）\n\n#### 6.2.2 专利储备\n截至2023年末：\n- 商汤累计专利授权4,200+项（含1,100+项发明专利）\n- 旷视科技专利总数3,700+项\n- 依图科技专利总数2,900+项\n（数据来源：国家知识产权局公开数据）\n\n### 6.3 产品矩阵对比\n| 产品维度       | 商汤科技       | 旷视科技       | 云从科技       |\n|----------------|----------------|----------------|----------------|\n| 基础算法平台   | SenseCore      | Brain++        | 轻舟平台       |\n| 商业化产品线   | 12大行业方案   | 9大行业方案    | 7大行业方案    |\n| 标杆案例       | 上海智慧城市   | 北京大兴机场   | 广州政务云     |\n（数据来源：各公司官网及年报[2]）\n\n### 6.4 财务指标对比\n#### 6.4.1 关键财务比率（2023年度）\n| 指标           | 商汤   | 行业平均 | 最优值（旷视） |\n|----------------|--------|----------|----------------|\n| 毛利率         | 68.2%  | 62.5%    | 71.3%          |\n| 营收增长率     | 12.7%  | 18.4%    | 23.6%          |\n| 经营性现金流   | -9.8亿 | -5.2亿   | +3.4亿         |\n（数据来源：东方财富-港股-财务报表[1]）\n\n#### 6.4.2 客户结构\n- 商汤科技：政府客户占比42%，企业客户58%（数据来源：同花顺-主营介绍[2]）\n- 旷视科技：政府客户占比38%\n- 云从科技：政府客户占比51%\n\n### 6.5 竞争矩阵分析\n通过构建四象限竞争矩阵（技术领先性vs商业化能力）显示：\n1. 技术领导者：商汤、DeepMind（高研发密度+专利质量）\n2. 商业转化强者：海康威视、华为（低技术投入但高营收规模）\n3. 均衡发展者：旷视科技\n4. 细分领域专家：云从科技（政务市场优势）\n\n### 6.6 优劣势总结\n**核心优势：**\n- 全栈式AI技术体系（芯片+算法+平台）\n- 智慧城市领域先发优势（占政府项目中标数35%）\n- 产学研协同效应（与港中文等机构深度合作）\n\n**主要劣势：**\n- 海外扩张受阻（美国实体清单影响）\n- 企业端渗透率低于旷视（相差7.2pct）\n- 现金流承压（连续5季度负经营性现金流）\n\n（注：本部分所有数据均通过自动化流程校验，原始数据接口参见附录A）",
            "## 7. 估值与预测\n\n### 7.1 估值方法选择与模型构建\n基于公司所处行业特性和财务特征，本报告采用三种主流估值方法进行交叉验证：\n1. **现金流折现法（DCF）**：适用于具备稳定现金流特征的成熟期企业，核心参数包括：\n   - 永续增长率设定为2.5%（参照港股同行业中位数[1]）\n   - WACC采用10.2%（根据CAPM模型计算得出[2]）\n2. **市盈率法（PE）**：选取可比公司2024年预测PE均值为15.8倍（数据来源：Bloomberg同业对比[3]）\n3. **市销率法（PS）**：适用于高增长阶段企业，行业平均PS为2.1倍（数据来源：Wind行业数据库[4]）\n\n### 7.2 关键参数假设\n| 参数项       | 基准值 | 乐观情景 | 悲观情景 | 数据来源               |\n|--------------|--------|----------|----------|------------------------|\n| 营收增长率   | 8.5%   | 12.0%    | 5.0%     | 东方财富-港股财务报表[1]|\n| 毛利率       | 32.4%  | 35.0%    | 28.0%    | 同花顺-主营介绍[2]     |\n| 资本开支占比 | 6.2%   | 5.0%     | 8.0%     | 公司2023年报           |\n\n### 7.3 估值结果\n通过三种方法加权计算（权重分配：DCF 50%，PE 30%，PS 20%），得出公司合理估值区间：\n- **目标价区间**：HK$25.6-32.4\n- **中枢价值**：HK$28.9（较当前股价存在18.7%上行空间）\n\n#### 分情景测算结果：\n1. **基准情景**：\n   - DCF估值：HK$27.5\n   - PE法估值：HK$29.1（按15.8倍*2024EPS）\n   - PS法估值：HK$24.8\n\n2. **乐观情景**（需求超预期）：\n   - 估值上限可达HK$34.2（+28%空间）\n\n3. **悲观情景**（成本压力）：\n   - 估值下限为HK$21.3（-20%风险）\n\n### 7.4 股价驱动因素\n1. **核心催化剂**：\n   - 新产品管线落地（预计2024Q2披露[5]）\n   - 东南亚市场扩张计划（公告编号：2023-086）\n2. **风险因素**：\n   - 原材料价格波动（沪铝期货同比+14%[6]）\n   - 行业竞争加剧（市场份额年降1.2pct[7]）\n\n### 7.5 敏感性分析\n通过蒙特卡洛模拟显示：\n- WACC变动±1%将导致估值波动∓7.2%\n- 永续增长率变动±0.5%对应估值变化∓4.8%\n- 毛利率每提升1pct可推升估值2.3%（数据来源：DCF模型测算）\n\n（注：所有估值结果均基于2024年3月31日汇率1HKD=0.91CNY换算）",
            "## 8. 治理结构\n\n### 8.1 公司治理架构\n公司建立了符合现代企业制度的治理架构，形成股东大会、董事会、监事会和经营管理层相互制衡的决策体系。董事会下设战略委员会、审计委员会、薪酬与考核委员会等专门委员会，各专业委员会运作规范[3]。根据最新披露信息，公司独立董事占比达到1/3，符合港交所上市规则要求。\n\n### 8.2 管理团队分析\n核心管理团队平均从业年限超过15年，主要高管均具备相关行业资深背景：\n- 董事长：拥有20年行业管理经验，曾主导多个重大战略项目\n- CEO：曾任国际知名企业高管，具备全球化运营视野\n- CFO：持有注册会计师资格，拥有10年以上上市公司财务管理经验\n\n管理团队近三年保持稳定，未出现核心高管异常变动情况[3]。\n\n### 8.3 股权结构与控制权\n截至最新披露：\n- 第一大股东持股比例：35.6%[3]\n- 前十大股东合计持股：68.2%[3]\n- 机构投资者持股比例：42.5%[3]\n\n股权结构相对集中，存在实际控制人，有利于重大决策效率，但需关注大股东控制风险。公司已建立关联交易审查机制，近年关联交易占比低于行业平均水平[1]。\n\n### 8.4 激励约束机制\n公司实施\"限制性股票+业绩奖金\"的复合激励方案：\n- 高管薪酬中绩效薪酬占比达60%\n- 近三年累计实施股权激励计划2次，覆盖核心技术人员30余人\n- 设置严格的业绩考核指标，包括ROE、营收增长率等关键财务指标[1]\n\n### 8.5 治理风险提示\n需重点关注：\n1. 独立董事实际履职效果待观察\n2. 中小股东权益保护机制有待完善\n3. 国际化业务扩张带来的跨境治理挑战\n\n整体而言，公司治理结构规范，管理层稳定性较好，激励约束机制较为完善，治理水平处于行业中上水平。建议持续关注控制权集中可能带来的治理风险[3]。",
            "## 9. 投资建议\n\n基于前述财务分析、行业对标及估值模型测算结果，我们给予该股\"买入\"评级，目标价位设定为28.5港元，建议投资期限为6-12个月。具体投资策略建议如下：\n\n1. **估值支撑**  \n   当前股价24.2港元较DCF估值结果（27.8-29.2港元区间）存在17.4%的潜在上涨空间[1]。EV/EBITDA倍数（8.3x）较行业均值（10.2x）低18.6%，PEG指标（0.89）显示估值具备吸引力[1]。\n\n2. **催化剂**  \n   - 预计2024年Q2新产线投产后产能提升30%（数据来源：同花顺-主营介绍[2]）  \n   - 大股东近6个月累计增持2.15%股份（数据来源：同花顺-股东信息网页爬虫[3]）\n\n3. **操作建议**  \n   - 现价24.2港元可建立基础仓位（30%）  \n   - 若回调至22港元支撑位可加仓至50%  \n   - 目标价位28.5港元分批减持  \n   - 止损位设置在20.8港元（较现价下浮14%）\n\n4. **风险提示**  \n   - 原材料价格波动可能影响毛利率1.5-2个百分点[1]  \n   - 行业政策变动风险（需关注2024年Q1新能源补贴细则）  \n\n本建议基于2023年12月31日收盘数据，后续将根据季度财报及重大事项进行动态调整。投资者应结合自身风险承受能力，建议单一个股配置比例不超过投资组合的15%。",
            "## 10. 风险提示\n\n### 10.1 行业风险\n1. **技术迭代风险**：人工智能行业技术更新迭代速度极快，若公司无法持续保持技术领先优势，可能面临市场份额被挤压的风险。根据行业数据显示，头部AI企业的年均研发投入增长率超过35%[1]，维持技术竞争力需要持续高额研发投入。\n2. **政策监管风险**：全球范围内对AI技术的监管日趋严格，特别是在数据隐私、算法伦理等领域。中国《生成式人工智能服务管理暂行办法》等法规的实施可能增加合规成本[2]。\n3. **行业竞争加剧**：据不完全统计，2023年中国AI领域新注册企业数量同比增长28%[3]，行业呈现红海竞争态势。\n\n### 10.2 公司风险\n1. **持续亏损风险**：财务数据显示公司近三年累计亏损达58.7亿元[1]，经营性现金流持续为负，存在资金链压力。\n2. **客户集中度风险**：前五大客户贡献营收占比超过45%[2]，存在大客户依赖风险。\n3. **人才流失风险**：AI行业核心技术人员流动率高达25%[3]，关键技术人才流失可能影响研发进度。\n\n### 10.3 估值风险\n1. **PS估值偏高**：当前市销率(PS)达18.7倍，显著高于行业平均11.2倍的水平[1]。\n2. **商业化不及预期**：智慧商业业务收入增速由2021年的89%降至2023年的32%[2]，商业化进程可能不及预期。\n3. **流动性风险**：港股通标的调整可能导致资金流出，近三个月日均成交量环比下降23%[3]。\n\n### 风险矩阵评估\n| 风险类别       | 影响程度(1-5) | 发生概率(1-5) | 风险值 |\n|----------------|---------------|---------------|--------|\n| 技术迭代风险   | 4             | 3             | 12     |\n| 政策监管风险   | 3             | 2             | 6      |\n| 持续亏损风险   | 5             | 4             | 20     |\n| 客户集中度风险 | 3             | 3             | 9      |\n| PS估值偏高     | 4             | 3             | 12     |\n\n注：风险值=影响程度×发生概率，分值越高风险越大\n\n## 引用文献\n[1] 东方财富-港股-财务报表: https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index  \n[2] 同花顺-主营介绍: https://basic.10jqka.com.cn/new/000066/operate.html  \n[3] 同花顺-股东信息: https://basic.10jqka.com.cn/HK0020/holder.html"
        ]
        return default_template

    def generate_outline(self, use_template: bool = True) -> Optional[List[Dict[str, Any]]]:
        """生成研报大纲"""
        try:
            if use_template:
                parts = self._load_outline_template()
                self.logger.info("📄 使用模板大纲")
            else:
                self._setup_agents()
                discuss_count = 0
                parts = []

                while discuss_count < 2:
                    self._agents['outline_opinion'].generate(self.report_info)
                    parts = self._agents['outline'].generate(self.report_info)
                    discuss_count += 1

                if not parts:
                    self.logger.error("❌ 大纲生成失败")
                    return None

            self.report_info.report_outline = parts
            self.logger.info(f"📄 大纲生成成功，共 {len(parts)} 个部分")
            return parts

        except Exception as e:
            self.logger.error(f"❌ 研报大纲生成失败: {e}")
            return None

    def generate_report_content_1(self, report_info: ReportInfo, use_template: bool = False) -> List[str]:

        if use_template:
            return self._load_text_template()

        outline = report_info.report_outline
        self.logger.info("\n✍️ 开始分段生成深度研报...")
        report_content = []
        part_all_num = len(outline)
        generated_names = set()
        nodes = self.report_info.has_sub_nodes()
        for idx, part in enumerate(outline):
            node = nodes[idx]

            self.report_info.cur_part_context.cur_part = part
            self.report_info.cur_part_context.is_report_last = (idx == len(outline) - 1)
            part_title = part.get('part_title', f'部分{idx + 1}')

            if node:
                part_title_name = self.report_info.cur_part_context.get_part_title_name()
                report_content.append(f"{part_title_name}")
            else:
                # 兼容rag_helper为None
                if self.rag_helper is not None:
                    try:
                        self.report_info.part_rag_context = self.rag_helper.get_context_for_llm(
                            f"{part_title} {self.report_info.target_company}",
                            max_tokens=4000, top_k=10
                        )
                    except Exception as e:
                        self.logger.error(f"RAG助手获取上下文失败: {e}")
                        self.report_info.part_rag_context = None
                else:
                    self.logger.warning("RAG助手不可用，跳过分段上下文获取。")
                    self.report_info.part_rag_context = None
                discuss_count = 0
                cur_content = ""
                while discuss_count < 2:
                    opinion = self._agents['section_opinion'].generate(self.report_info)
                    report_info.cur_part_context.cur_subsection_content_opinion = opinion
                    cur_content = self._agents['section_edit'].generate(report_info)
                    discuss_count += 1

                report_content.append(cur_content)
            self.logger.info(f"✅ 已完成：{report_content}")
            report_info.cur_part_context.cur_content = ""
            generated_names.add(part_title)
            self.logger.info(f"✅ 已完成：{part_title}")
        return report_content

    def _load_abstract_template(self, template_name: str = "default") -> str:

        return ""

    def get_abstract(self, report_info: ReportInfo, use_template: bool = False) -> str:
        if use_template:
            return self._load_abstract_template()
        abstract = self._agents['abstract'].generate(report_info)
        return abstract

    def get_full_report(self, report_info: ReportInfo) -> str:
        content_list = ContentConvert(self.report_info.report_outline).get_content_list_1()
        join = "\n\n".join(content_list)
        # get_content_list = report_info.create_report_content().get_content_list()
        self.logger.info(f"\n✍️ 目录生成成功...{join}")
        """获取完整报告"""
        try:

            report_text_list = self.generate_report_content_1(report_info, False)
            self.report_info.report_text_list = report_text_list

            abstract = self.get_abstract(report_info,False)

            # 构建完整报告
            full_report_parts = [
                f"# {self.report_info.report_title}\n",
            ]
            full_report_parts.append(abstract)
            full_report_parts.extend(content_list)
            full_report_parts.extend(report_text_list)

            return "\n\n".join(full_report_parts)

        except Exception as e:
            self.logger.error(f"❌ 生成完整报告失败: {e}")
            raise

    def generate_report(self) -> Optional[str]:
        """生成完整研报"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 开始研报生成流程")
        self.logger.info("=" * 80)

        try:
            # 初始化代理
            self._setup_agents()

            # 获取RAG上下文（兼容RAG助手为None的情况）
            rag_context = None
            rag_company = None
            if self.rag_helper is not None:
                try:
                    rag_context = self.rag_helper.get_context_for_llm(
                        f"{self.target_company} 公司分析 财务数据 行业地位 竞争分析",
                        max_tokens=4000, top_k=20
                    )
                    rag_company = self.rag_helper.get_context_for_llm(
                        f"{self.target_company}竞争对手分析",
                        max_tokens=4000, top_k=10
                    )
                except Exception as e:
                    self.logger.error(f"RAG助手获取上下文失败: {e}")
                    rag_context = None
                    rag_company = None
            else:
                self.logger.warning("RAG助手不可用，跳过上下文获取。")

            # 初始化报告信息
            self.report_info = ReportInfo(self.target_company, rag_context, rag_company)
            self.report_info.report_title = f"{self.report_info.target_company}研报"
            # 生成大纲
            outline = self.generate_outline(use_template=False)
            if not outline:
                raise ValueError("大纲生成失败")

            # 生成完整报告
            final_report = self.get_full_report(self.report_info)

            # 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # output_file = f"{self.target_company}深度研报_优化版_{timestamp}.md"
            output_file = f"Company_Research_Report.md"

            #copy to reports
            os.makedirs("reports", exist_ok=True)
            shutil.copy(output_file, f"reports/Company_Research_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            # 删除原始文件
            if os.path.exists(output_file):
                os.remove(output_file)
            # 删除word文件
            if os.path.exists(f"Company_Research_Report.docx"):
                os.remove(f"Company_Research_Report.docx")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_report)

            # 创建文档转换实例
            pipeline = DocumentConversionPipeline()
            
            # 运行文档转换流程
            result = pipeline.run_conversion(output_file)
            # convert_md_to_docx_pure_python(output_file)


            self.logger.info(f"\n✅ 研报生成完成！文件已保存到: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"❌ 研报生成失败: {e}")
            return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='研报生成流程')
    parser.add_argument('--company_name', default='4Paradigm', help='目标公司名称')
    parser.add_argument('--company_code', default='06682.HK', help='股票代码')
    parser.add_argument('--template', action='store_true', help='使用模板大纲')
    parser.epilog = '示例：python run_company_research_report.py --company_name 商汤科技 --company_code 00020.HK'
    args = parser.parse_args()

    try:
        # 创建研报生成实例
        company_code = args.company_code.split(".")

        pipeline = DataCollectionPipeline(
            target_company=args.company_name,
            target_company_code=company_code[0],
            target_company_market=company_code[1],
            search_engine="all"
        )

        # 运行数据收集流程
        success = pipeline.run_data_collection()

        if success:
            print("\n🎉 数据收集流程执行完毕！")
            print("📊 所有数据已向量化存储到PostgreSQL数据库")
        else:
            print("\n❌ 数据收集流程执行失败！")


        pipeline = ReportGenerationPipeline(
            target_company=args.company_name,
            target_company_code=company_code[0],
            target_company_market=company_code[1]
        )

        # 运行研报生成流程
        output_file = pipeline.generate_report()

        if output_file:
            print(f"\n🎉 研报生成流程执行完毕！")
            print(f"📋 研报文件: {output_file}")
        else:
            print("\n❌ 研报生成流程执行失败！")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()