from typing import Optional, Any, Mapping, Dict, List

from app.company.utils.content_convert import ContentConvert


class CurPart:

    is_part_last: bool = False

    is_subsection_last: bool = False

    is_report_last: bool = False

    all_parts_num: int = 0

    # 上一次章节
    prev_part_num: int = 0

    # 上一次章节
    prev_part: Dict[str, Any] = {}

    # 上一次章节的内容
    prev_part_content: List[str] = []

    prev_part_content_abstract: str = ""

    # 上一次章节的子部分
    prev_subsection: Dict[str, Any] = {}

    # 上一次章节的子部分内容
    prev_subsection_content: str = ""


    cur_part_num: int = 0
    # 当前处理的部分
    cur_part:Dict[str, Any] = {}

    # 当前处理的内容
    cur_part_content: List[str] = []

    cur_content: str = ""

    # 当前处理的子部分
    cur_subsection:Dict[str, Any] = {}

    cur_subsection_first: bool = True

    # 当前处理的子部分内容
    cur_subsection_content : str = ""

    # 当前处理的内容
    cur_subsection_content_opinion:  List[Dict[str, Any]] = []

    def get_part_title_name(self):
        part_title = self.get_cur_part_value("part_title")
        part_title_type = self.get_cur_part_value("part_title_type")
        # if 章/节/目
        if "章"==part_title_type:
            return f"## {part_title}"
        elif "节"==part_title_type:
            return f"### {part_title}"
        elif "目"==part_title_type:
            return f"### {part_title}"
        return


    def set_is_report_last(self, is_report_last: bool):
        self.is_report_last = is_report_last

    def clear_cur_subsection_content_opinion(self):
        self.cur_subsection_content_opinion = []

    def add_cur_part_num(self):
        self.cur_part_num += 1

    def get_cur_part_value(self, key:str):
        return self.cur_part.get(key, "")

    def get_cur_subsection_value(self, key: str):
        return self.cur_subsection.get(key, "")

    def get_prev_content_prompt(self)->str:
        if not self.prev_subsection_content:
            return "这是报告的第一个章节"


        if self.cur_subsection_first:

            prev_part_title = self.prev_part.get("part_title", "")
            return f"作为参考，这是上一章节 '{prev_part_title}'的内容摘要：\n{self.prev_part_content_abstract}...\n"

        # 子章节
        cur_part_title = self.cur_part.get("part_title", "")
        prev_subsection_title = self.prev_subsection.get("subsection_title", "")
        return f"作为参考，这是上一小章节 '{cur_part_title}'-‘{prev_subsection_title}’ 的内容摘要：\n{self.prev_subsection_content}...\n"

    def go_prev_part(self):
        self.prev_part_num = self.cur_part_num
        self.prev_part = self.cur_part
        self.prev_part_content = self.cur_part_content


    def go_prev_subsection(self):
        self.prev_subsection = self.cur_subsection
        self.prev_subsection_content = self.cur_subsection_content
        self.cur_subsection_content = ""



class ReportContent:

    report_title: str = ""
    report_outline: List[Dict[str, Any]] = []

    def get_content_list(self)->List[str]:
        return ContentConvert(self.report_outline).get_content_list()

    def init(self,report_title, report_outline):
        self.report_title = report_title
        self.report_outline = report_outline
        return self



class ReportInfo:
    outline_info_content_format = """
        - 以yaml格式输出，务必用```yaml和```包裹整个yaml内容
        - 每一项为一个主要部分，每部分需包含：
                  - part_num: 序号
                  - part_title: 章节标题（每个章节都要有序号因为要生成目录，但是摘要没有序号）
                  - part_target: 章节目标
                  - part_desc: 章节内容简介
                  - 章节需覆盖摘要（不可以修改成其他名称）、公司基本面、财务分析、行业对比、估值与预测、治理结构、投资建议、风险提示等顺序可以修改
        """


    outline_info_content_opinion_format="""
    - 以yaml格式输出，务必用```yaml和```包裹整个yaml内容
    - 每一项为一个主要部分，每部分需包含：
              - part_num: 序号
              - part_title: 章节标题
              - part_target: 章节目标
              - part_desc: 章节内容简介
              - part_opinion: 章节内容建议
              - 章节需覆盖摘要（不可以修改成其他名称）、公司基本面、财务分析、行业对比、估值与预测、治理结构、投资建议、风险提示等顺序可以修改
    """


    requirement = """
    1 在研究公司之前，我们应先研究该公司所处的行业。该行业处于怎样一个发展历程中。举个例子，普通钢铁水泥很显然处于产能过剩中，在这种可能衰退的行业中寻找投资标的很难，当然也不是不可以。市场普遍对这种衰退行业给的估值不高，且如果衰退趋势有加速，给的估值会进一步下降。
    再比如和资本市场关系密切的强周期行业券商$中信证券(hk06030)$ ，牛市时双击，熊市双杀，这是不以人为转移的，所以投资这类强周期行业的个股，要特别注意目前所处的节点，是周期顶端，周期中，还是周期末端。如果判断错周期节点，很可能蒙受巨大损失。
    再比如弱周期行业医药，医药有很多细分子行业，增速不一样，可能普通化药，原料药发展速度停滞了，但是医疗器械，生物药，生物检测增速还很快。当然资本市场给的估值也不同。
    一般来说弱周期行业增速很快的细分子行业易出黑马，强周期行业判断对周期，也可能获得超额收益（比如猪周期），衰退行业中的沙漠之花，市场份额不断上升，费用率因规模经济不断下降，业绩持续增长的公司也可能走出慢牛行情。
    2 研究完行业目前所处情况，我们还必须进行合理大胆的预测将来的行业趋势。想要预测正确，最好方法就是研究国外比我们走的早几步国家，和我国国情类似，行业所处情况类似阶段他们的资本市场相关个股的表现，之后几年十几年行业的趋势变化最需要研究，一般来说，就是日本韩国，美国，欧洲一些国家，有时候这部分资料可呢很难走到，这就是考验分析师能力的地方了，有的需要请教相关国家老一辈生活过的人，甚至需要亲自去现场去海外相关上市公司调研咨询，不是每个人都有这个条件的。
    3 行业研究完了，进入下一步公司研究。
    （1）首先我们要研究的是招股说明书，这里面学问大了。我们要研究这个公司的历史，它上市募集资金的用途，以及是否像书中所说的那样把钱花到相关项目中，还是仅仅做个样子，想上市圈钱，这个区别非常大。一般来说，真正的成长股黑马一定是继续募集资金发展主业，如果你看到募集自己用来高管发工资挥霍，用途不明，甚至买理财产品，这里面问题就大了。书中还有该公司所处行业竞争力分析，主要竞争对手的数据，大部分还是真实的，如果时间不长久（2年之内）有一定参考价值。然后我们可以比比上市之前公司主要竞争对手情况和上市以后，该公司在行业中的排名是否上升，如果你募集资金以后还没有很快增长，甚至比那些没上市的还发展速度慢，那毫无疑问不是啥好票。书中还有所投项目，市场变化很快，可能有的公司会转型成其他主业或者直接收购进入其他行业。但是如果一个公司一直作为壳存在，不停注入不停更换主业和管理层，我觉得应该不是能长线持有的公司。比如$海虹控股(sz000503)$   。
    （2）其次研究的是管理层，我个人认为人  才是企业不断发展的原动力。一个伟大的公司必定有伟大的管理层。如果管理层对公司事务不管不问，大肆挥霍，有造假前科，没有任何行业经验，这样的公司绝不能投。理想的高管（董事长）最好在35-45之间，有丰富的相关行业经验，对公司有决定权，股权集中，甚至待企业如自己孩子一样。脾气不能够太暴躁，也不能太温柔，最好还有一个性格互补的拍档，比如芒格和巴菲特，这样企业经营才能平稳安定。最忌讳那种好大喜功，急火攻心，哪个行业火投哪里的管理层。
    （3）第三才研究企业报表，估计很多人最看重的吧。我个人觉得企业报表一定要看，但绝对不是最重要的。中国公司报表内幕很多，可以信但不能全信。比如一个公司行业第五，居然净利润和净利率比第一还高。A股经常出现这类造假，必须提高警惕！还有些通过关连交易，通过不断上下游压货造成业绩虚高，里面的水分也需要我们仔细甄别。一般人看不出，没关系，多关注一些细分小行业龙头，很多年报表相差不太大（毕竟连续很多年造假也很难），市场关注度不高的这些公司反而机会更大些。很多人过分关注PEG，关注一季度半年的增速，这是不对的，其实30%和40%增速是差不多的，有时候只是修饰了一下报表。再看看同行业其他公司报表，仔细比对，发现不同。探寻该公司在该行业中增速情况，估值高低情况。再比较一下和其他行业的估值差距，找出理由。如果不是合理，要小心了。我们更应该关注的是公司是不是变得更加强大，在上下游的控制力和议价权是否更强，这才是以后我们长期投资最需要关注的。
    （4）企业战略.最最忌讳一种管理层朝令夕改，看到哪个行业火去哪里。比如上半年最火的互联网加，是个企业如果不和互联网沾上点关系，都不好意思和别人说我是上市公司。这其实是企业战略模糊的表现，真正的战略应该正视自己的优势和劣势，采取积极的心态主动调整，相当于田忌赛马，取长补短。把每一分钱都用在刀刃上，我们投资最害怕企业脑子一热，董事长一拍板说投就投还是大手笔，没有经过严密的审核和考虑。很多企业都是死在扩张而不是死于自己做减法。相反，砍去那些负担重，前景不好的子公司，轻装上阵反而能获得不错的企业生命力。
    （5）重视员工权利和股东权利。爱民如子，珍视股东权利，这样的公司才值得长期持有。企业发展关键还是靠人，我 宁可你多拿点钱出来给员工，也不希望你挥霍随意投资项目。重视分红，给投资者发表意见并听取合理采纳的公司值得密切关注。
    （6）重视主业，不随意乱扩张。重视和ZF关系，重视生态环境和周边居民关系。很多企业被曝光居民闹事，环境污染问题给企业蒙羞。
    （7）舍得花钱研发，技术进步企业才能获得更好的竞争力。我们也要把研发投入纳入考核指标，并重视研发产出比。
    （8）根据公司公告和各季度报告，不断更新自己的判断。每个公司都不是一成不变的，我们需要不断更新数据，甚至需要亲自去公司调研，寻找更好的投资标的，不能因循守旧，固执己见。    
        """
    opinion = """
        修改意见遵循
        1、整体结构方面
            （1）逻辑框架梳理
                现状描述部分：占比约 30%，涵盖项目大致情况，建议细分小模块，如按政策、市场背景分类阐述 。
                问题分析部分：占 40%，目前分析笼统，应采用 “现象 + 影响 + 措施” 递进短句组深入分析，举例说明成本超支问题的分析方式 。
                改进路径部分：占 30%，现有内容宽泛，需针对问题提具体可操作措施，举例说明技术落后问题的改进路径 。
                效果预判部分：目前不够详细，可参考过往经验数据或建模型，准确预测改进措施效果，举例说明预测营销策略效果的内容 。
            （2）层级递进关系
                逻辑衔接：各部分间要增加明确过渡句式，如从现状描述到问题分析，可用 “鉴于目前项目呈现出的上述现状，不难发现其中存在一些亟待解决的问题” 。
                问题分析：不同问题关系通过层级递进阐述，先提表面、直接问题，再深入分析连锁反应和深层次问题，举例说明资金周转问题引发的系列影响 。
        2、内容细节方面
            （1）数据处理与呈现
                数据准确性：报告数据锚点每千字约 3 - 5 个，需仔细核对数据，如市场规模数据要核实来源和统计口径 。
                数据转换：绝对值数据可转相对占比，像投入资金可转总预算占比，更直观反映问题 。
                数据可视化：增加柱状图、折线图等可视化元素，展示数据变化趋势，如用柱状图呈现项目销售额变化 。
            （2）专业术语与表达
                术语转化：把专业术语转化为领域基础概念，方便非专业人士理解，如 “边际效益” 转 “投入产出比”；通用文档要控制术语密度，每千字不超 25% 。
                句式简化：简化复合长句，采用 “现象 + 影响 + 措施” 递进短句组形式，还举例展示原句简化方式 。
            （3）案例引用
                案例选取：要增加案例佐证部分，选与本项目类似的成功或失败案例，需具代表性，能说明项目问题或借鉴经验，还以新产品推广项目举例 。
                案例分析：对选取案例深入分析，不仅描述过程和结果，还要分析背后原因与采取的措施，分析成功案例探讨营销、定位等因素，分析失败案例找出市场、质量等关键问题 。
        3、语言风格方面
            （1）语气适配
                情感饱和度：报告整体情感饱和度约 20 - 25%，为有限度情感表达，关键问题和重要结论可适当加强语气，如强调项目改进紧迫性时用更强烈措辞，但不过度夸张 。
                语气用词：依报告性质调整，建议性内容用 “建议”“应当” 等词，强调规定性内容用 “必须”“不得” 等规范用语，还举例说明 。
            （2）语言简洁性
                情感饱和度：报告整体情感饱和度约 20 - 25%，有限度情感表达。关键问题、重要结论可适当加强语气，如强调项目改进紧迫性时用更强烈措辞，不过度夸张 。
                语气用词：依报告性质调整，建议性内容用 “建议”“应当” 等；强调规定性内容用 “必须”“不得” 等规范用语，还举例说明 。
        4、格式规范方面
            （1）整体排版
                段落设置：移动端适配时，段落控制在 5 行内，句长 25 字内，长段落按观点或要点拆分 。
                标题层级：用 “一、”“（一）”“1” 等编号体系，不同层级标题在字体、字号、格式上有明显区别 。
            （2）特殊文档处理格式
                法律合同格式：报告涉及合同条款内容时，采用 “鉴于双方约定如违约则” 三段式结构，举例说明了该结构的具体表述 。
                演讲稿格式：按每 200 字插入互动提问或场景假设的要求调整，举例介绍在介绍项目进展到 200 字左右时插入互动提问的方式 。
                检讨书格式：遵循 “错误描述→根源剖析→改正承诺” 黄金比例 3:4:3 ，并以项目失误检讨书为例，说明各部分占比及内容撰写方式。
                规章制度格式：每章节按 “禁止性条款 + 正向引导 + 例外情形” 设置，以项目团队管理章节为例，阐释各部分内容示例 。
    """

    outline ="""
摘要： 简要概述报告的主要内容、分析结论和投资建议。
1. 公司概况
公司基本信息：成立时间、总部地点
公司历史：重要里程碑和发展阶段。
管理团队：主要成员的背景和经验。
公司治理结构：股东大会、董事会、监事会等。
主营业务：上下游产业链、具体产品情况、产量产能等。
2. 财务分析
收入和利润：历史数据和预测。
财务比率分析：盈利能力、偿债能力、运营效率等。
现金流量：经营、投资和筹资活动产生的现金流。
3. 行业分析
行业概况：行业规模、增长趋势、周期性等。
行业竞争格局：主要竞争对手、市场份额分布。
行业驱动因素：影响行业发展的关键因素。
行业未来发展趋势
4. 估值分析
盈利预测：收入预测、毛利率预测、费用率预测、净利润预测
相对估值：市盈率(P/E)、市净率(P/B)等。
绝对估值：如折现现金流(DCF)模型。
5. 投资建议
基于以上分析，提出投资评级（买入、持有、卖出）。
目标价格和投资时限。
风险因素
6. 结论
总结分析结果，重申投资建议。
7. 附录
相关数据表格、图表、参考文献等。
    """

    report_format ="""
在撰写研究报告时，遵循一定的格式和模板是非常重要的，这不仅可以提升报告的专业性，还能提高读者的阅读体验。本文将详细介绍研究报告的格式、范文以及常用模板和表格，帮助大家更好地完成研究报告。
一、研究报告的基本格式
研究报告的基本格式通常包括以下几个部分：标题页、摘要、目录、引言、文献综述、研究方法、结果与分析、结论与讨论、参考文献和附录。以下将逐一介绍各部分的内容及其撰写要点。
1. 标题页
标题页是研究报告的第一页，通常包含以下正文：
标题：简明扼要地概括报告的主题。
作者姓名：报告的主要撰写人。
机构名称：作者所属的单位或机构。
日期：报告完成的日期。
2. 摘要
摘要是对整个报告的简短总结，通常不超过300字。它应包括以下几个方面的内容：
研究目的：简要说明研究的背景和研究问题。
研究方法：概述所采用的研究方法和数据收集方式。
主要发现：简要列出研究的主要结果。
结论：简要描述研究的结论和建议。
3. 目录
目录列出了报告中各个章节的标题和对应的页码，便于读者快速查找感兴趣的内容。目录应包括从引言到结论与讨论的所有主要部分。
4. 引言
引言部分应包括以下几个方面的内容：
研究背景：介绍研究问题的背景和意义。
研究目的：明确研究的目标和研究问题。
研究范围：界定研究的具体内容和范围。
文献综述：简要回顾相关领域的已有研究，指出研究中存在的空白。
5. 文献综述
文献综述是对已有研究成果的系统梳理和评价，通常包括以下几个方面的内容：
相关理论和概念：介绍与研究主题相关的理论和概念。
前人研究成果：总结前人在相关领域的主要研究成果。
研究的不足：指出已有研究中的不足之处，阐明本研究的必要性。
6. 研究方法
研究方法是报告的核心部分之一，应详细说明所使用的研究方法和数据分析手段。通常包括以下几个方面：
研究设计：描述研究的整体设计和框架。
数据收集：说明数据的来源和收集方式。
数据分析：详细描述数据分析的方法和工具。
7. 结果与分析
结果与分析部分应呈现研究的主要发现，并对结果进行详细的解释和分析。通常包括以下几个方面：
数据展示：通过表格、图表等形式展示研究数据。
结果描述：对数据进行详细描述，并解释其含义。
结果分析：对结果进行深入分析，探讨其背后的逻辑和原因。
8. 结论与讨论
结论与讨论部分应对整个研究进行总结，并提出未来研究的建议。通常包括以下几个方面：
研究结论：总结研究的主要发现和结论。
研究意义：阐述研究的理论和实践意义。
研究建议：提出未来研究的方向和建议。
9. 参考文献
参考文献部分应列出报告中引用的所有文献，通常按照一定的格式排列，如APA、MLA等。参考文献应包括书籍、期刊文章、会议论文等各种形式的资料。
10. 附录
附录部分可以包含一些辅助性材料，如调查问卷、访谈记录、数据表等，供有兴趣的读者参考。
二、研究报告范文模板
以下是一篇完整的研究报告范文，展示了上述各个部分的具体内容和格式。

示例
研究报告格式范文模板表格
标题页
标题：关于大学生网络使用行为的研究报告
作者姓名：XXX
机构名称：某大学社会学系
日期：XXXX年XX月XX日
摘要
随着互联网的快速发展，大学生的网络使用行为越来越受到关注。本研究通过对某大学500名在校大学生的问卷调查，分析了大学生网络使用的现状及其影响因素。研究发现，大多数大学生每天上网时间超过3小时，主要用于学习、娱乐和社交。影响大学生网络使用的主要因素包括性别、年级和专业等。本研究为高校制定合理的网络管理政策提供了依据。
目录
引言
文献综述
研究方法
结果与分析
结论与讨论
参考文献
附录
引言
1.1 研究背景
互联网已成为现代社会不可或缺的一部分，特别是在大学生群体中，互联网的使用频率和时长不断增加。然而，过度的网络使用可能会对学生的学业和身心健康产生负面影响。因此，了解大学生的网络使用行为具有重要意义。
1.2 研究目的
本研究旨在探讨大学生网络使用的现状及其影响因素，为高校制定合理的网络管理政策提供依据。
1.3 研究范围
本研究以某大学在校大学生为研究对象，通过问卷调查的方式收集数据，分析大学生网络使用的时长、内容及其影响因素。
1.4 文献综述
已有研究表明，大学生的网络使用行为受多种因素影响，包括性别、年级、专业等。此外，网络使用对学业和身心健康的影响也得到了广泛关注。然而，目前关于大学生网络使用行为的研究仍存在一些空白，需要进一步探讨。
文献综述
2.1 相关理论和概念
网络使用行为是指个体在网络上的活动及其频率、时长等方面的表现。根据不同的标准，网络使用行为可以分为多种类型，如信息获取、娱乐、社交等。
2.2 前人研究成果
已有研究表明，大学生的网络使用行为具有多样性和复杂性。例如，一项对某高校学生的调查显示，大学生每天上网时间平均为3.5小时，主要用于学习、娱乐和社交（张三，2018）。另一项研究发现，性别是影响大学生网络使用的一个重要因素，男生的网络使用时间普遍高于女生（李四，2019）。
2.3 研究的不足
尽管已有研究为我们理解大学生的网络使用行为提供了重要参考，但仍存在一些不足之处。首先，现有研究多采用定量分析方法，缺乏对定性数据的深入挖掘。其次，大多数研究仅关注单一变量的影响，忽视了多因素的综合作用。因此，本研究将在前人基础上，进一步探讨大学生网络使用行为的影响因素。
研究方法
3.1 研究设计
本研究采用问卷调查的方式进行数据收集，问卷内容包括个人基本信息、网络使用情况及其影响因素等。调查对象为某大学在校大学生，共发放问卷500份，回收有效问卷480份。
3.2 数据收集
问卷调查采用匿名方式进行，被调查者需填写个人基本信息（如性别、年级、专业等）、每天上网时间及用途等内容。数据收集后，通过SPSS软件进行统计分析。
3.3 数据分析
数据分析主要包括描述性统计、相关性分析和回归分析等方法。首先，通过描述性统计了解大学生网络使用的基本特征；其次，通过相关性分析探讨各变量之间的关系；最后，通过回归分析确定影响大学生网络使用的主要因素。
结果与分析
4.1 数据展示
项目	平均值	标准差

每天上网时间	3.2小时	1.5小时
用于学习时间	1.5小时	1.0小时
用于娱乐时间	1.2小时	0.8小时
用于社交时间	0.8小时	0.6小时

4.2 结果描述
调查结果显示，大学生每天平均上网时间为3.2小时，其中用于学习的时间最多，平均为1.5小时；其次是娱乐时间，平均为1.2小时；用于社交的时间最少，平均为0.8小时。此外，不同性别、年级和专业的学生在网络使用时间和用途上存在一定的差异。
4.3 结果分析
进一步分析发现，性别是影响大学生网络使用的重要因素之一。男生每天上网时间显著高于女生（t=2.34, p<0.05），且男生更倾向于使用网络进行娱乐活动。年级方面，大一新生的网络使用时间最长，而大四毕业生的网络使用时间最短。专业方面，理工科学生的网络使用时间明显高于文科学生。这些结果表明，大学生的网络使用行为受多种因素的影响，需要综合考虑各方面的因素进行管理和引导。
结论与讨论
5.1 研究结论
本研究通过对某大学在校大学生的问卷调查，分析了大学生网络使用的现状及其影响因素。研究发现，大多数大学生每天上网时间超过3小时，主要用于学习、娱乐和社交。影响大学生网络使用的主要因素包括性别、年级和专业等。具体来说，男生每天上网时间显著高于女生，且更倾向于使用网络进行娱乐活动；大一新生的网络使用时间最长，而大四毕业生的网络使用时间最短；理工科学生的网络使用时间明显高于文科学生。这些结果表明，大学生的网络使用行为受多种因素的影响，需要综合考虑各方面的因素进行管理和引导。
5.
相关链接
如何撰写小学研究报告，格式和范文模板
历史研究报告格式范文模板
简讯格式及范文
小学研究报告格式范文模板及图片解析
大学研究报告格式范文模板
研究报告格式范文模板
"""
    background = """
    本报告基于自动化采集与分析流程，涵盖如下环节：
    - 公司基础信息等数据均通过akshare、公开年报、主流财经数据源自动采集。
    - 财务三大报表数据来源：东方财富-港股-财务报表-三大报表 (https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index)
    - 主营业务信息来源：同花顺-主营介绍 (https://basic.10jqka.com.cn/new/000066/operate.html)
    - 股东结构信息来源：同花顺-股东信息 (https://basic.10jqka.com.cn/HK0020/holder.html) 通过网页爬虫技术自动采集
    - 行业信息通过DuckDuckGo等公开搜索引擎自动抓取，引用了权威新闻、研报、公司公告等。
    - 财务分析、对比分析、估值与预测均由大模型（如GPT-4）自动生成，结合了行业对标、财务比率、治理结构等多维度内容。
    - 相关数据与分析均在脚本自动化流程下完成，确保数据来源可追溯、分析逻辑透明。
    - 详细引用与外部链接已在正文中标注。
    - 数据接口说明与免责声明见文末。
    """
    report_evaluation_criteria = """
【研报评判标准】
## 1. 内容深度（权重：25%）
评估标准：
- 覆盖范围 ：是否涵盖公司基本面、行业分析、技术评估、财务分析等核心要素
- 分析层次 ：从宏观行业到微观公司的分析深度
- 专业程度 ：技术细节、财务指标、行业数据的专业性
- 逻辑完整性 ：各章节间的逻辑关联和论证完整度
例如：
- 行业背景与发展趋势分析
- 公司核心竞争力深度剖析
- 竞争公司对比分析
- 技术壁垒和专利布局评估
- 财务健康度多维度分析
- 治理结构和管理层评估
- 风险因素识别和量化
## 2. 数据质量（权重：20%）
评估标准：
- 数据来源 ：权威性、时效性、可信度
- 数据完整性 ：关键指标的覆盖程度
- 数据准确性 ：数据间的一致性和逻辑性
- 引用规范 ：数据来源标注的完整性
例如：
- 财务数据的准确性和完整性
- 行业数据的权威性（如弗若斯特沙利文、头豹研究院等）
- 技术指标的专业性（专利数量、论文发表等）
- 市场数据的时效性和可比性
- 同业对比数据的全面性
## 3. 分析框架（权重：25%）
评估标准：
- 方法论科学性 ：采用的分析方法是否科学合理
- 框架完整性 ：分析框架的系统性和全面性
- 工具运用 ：估值模型、风险评估工具的专业性
- 创新性 ：分析方法的前瞻性和创新性
例如：
- 估值方法（DCF、SOTP、相对估值等）
- 风险评估框架（VaR、压力测试等）
- 竞争分析工具（波特五力、SWOT等）
- 情景分析和敏感性测试
- ESG评估框架
## 4. 实用性（权重：20%）
评估标准：
- 投资指导价值 ：对投资决策的实际指导意义
- 操作性 ：建议的可执行性和具体性
- 适用性 ：针对不同投资者类型的适用程度
- 时效性 ：建议的时效性和前瞻性
例如：
- 明确的投资评级和目标价
- 具体的建仓策略和仓位建议
- 风险控制和止损机制
- 关键催化剂和时间节点
- 跟踪指标和预警机制
- 不同风险偏好的投资策略
## 5. 创新性（权重：10%）
评估标准：
- 分析视角 ：独特的分析角度和洞察
- 方法创新 ：新颖的分析方法和工具
- 前瞻性 ：对未来趋势的预判能力
- 差异化 ：与市场主流观点的差异化见解
例如：
- 独特的行业洞察和技术趋势判断
- 创新的估值方法和风险评估模型
- 前瞻性的政策影响分析
- 差异化的投资逻辑和策略
- 新颖的数据分析和可视化方式
    """

    outline_opinion_1 = """
    ## 评分维度
    1. 技术分析深度 (2.5分)
       - 技术壁垒分析
       - 核心技术评估
       - 创新能力判断
       例如
        - 技术壁垒量化分析
        - 专利价值评估
        - 技术路线前瞻性
        - 研发投入效率分析
        - 技术商业化能力
    2. 投资实用性 (2.5分)
       - 投资建议明确性
       - 风险收益分析
       - 操作指导价值
       例如
        - 投资建议明确性
        - 风险收益量化分析
        - 投资时机判断
        - 组合配置建议
        - 动态调整策略
    3. 数据支撑质量 (2.5分)
       - 数据来源可靠性
       - 数据分析深度
       - 量化分析水平
       例如
        - 数据来源权威性
        - 数据完整性和时效性
        - 量化分析深度
        - 数据验证和交叉检验
        - 预测模型科学性
    4. 逻辑结构 (2.5分)
       - 报告结构合理性
       - 论证逻辑清晰度
       - 内容完整性
       例如
        - 报告结构合理性
        - 论证逻辑严密性
        - 内容完整性
        - 章节衔接自然度
        - 结论一致性
    5、加分项(最多0.5分)：
        - 独家数据或深度调研
        - 创新性分析方法
        - 前瞻性判断准确
        - 特殊价值发现    
    6、扣分项：
        - 数据错误或误导性信息 (-0.5分)
        - 逻辑矛盾或自相冲突 (-0.3分)
        - 格式不规范或表达不清 (-0.2分)
    参考建议示例
    一、整体结构调整建议
        1.1 逻辑顺序优化(是否符合投资者阅读习惯)
        1.2 章节权重重新分配（章节的权重分配是否合理）
    二、具体章节修改建议
        2.1 摘要（是否过于简单，是否缺乏核心亮点）
        2.2 行业分析框架（对行业的现状和未来是否有深入的分析）
        2.3 技术分析(技术分析是否全面，使用的方法是否足够说明公司状态）
        2.4 动态分析(是否有其他新入局者)
        2.5 财务分析（是否够深入，是否能体现公司现在和未来财务状况）
        2.6 文章篇幅（篇幅是否投资者关注，是否缺乏核心亮点）
        2.7 估值建模（文章的建模是否单一，是否能体现本次研报的重点）
        2.8 投资建议（是否过于简单，是否缺乏具体的投资策略）
        2.9 风险提示（是否过于简单，是否缺乏具体的风险提示，是否风险评估过于定性）
    """
    # 报告章节信息
    part_rag_context : Any
    # 报告题目
    report_title: str = ""
    # 报告大纲
    report_outline : List[Any] = []

    # 报告大纲修改意见
    report_outline_opinion : List[Dict[str, Any]] = []

    cur_part_context:CurPart = CurPart()

    report_content: ReportContent = ReportContent()

    generated_names = set()

    report_text_list=[]

    _report_is_sub_part = []

    def has_sub_nodes(self) -> List[bool]:
        """判断每个章节是否有子节点

        Args:
            parts: 包含章节信息的列表，每个元素包含part_num字段

        Returns:
            返回一个布尔列表，表示每个章节是否有子节点
        """
        result = []
        part_nums = [part["part_num"] for part in self.report_outline]

        for i, part in enumerate(self.report_outline):
            current_num = part["part_num"]
            # 检查是否是父节点（是否有以当前编号为前缀的其他编号）
            is_parent = any(
                other_num.startswith(current_num + ".")
                for other_num in part_nums
                if other_num != current_num
            )
            result.append(is_parent)

        return result

    def __init__(self,target_company,rag_context,rag_company):
        self.rag_context = rag_context
        self.rag_company = rag_company
        self.target_company = target_company

    def map_dict_to_cur_part(self) -> {}:
        """
        将字典映射到 CurPart 对象的属性上。

        :param cur_part: 包含章节信息的字典
        :return: CurPart 对象
        """
        cur_part_obj = {}

        # 映射基本属性
        for key, value in self.cur_part_context.cur_part.items():
            if hasattr(cur_part_obj, key):
                setattr(cur_part_obj, key, value)

        return cur_part_obj

    # def get_all_parts_num(self) -> int:
    #     """获取报告总章节数"""
    #     if self.report_outline!=0:
    #         return self.all_parts_num
    #     self.all_parts_num = len(self.report_outline)
    #     return self.all_parts_num

    # - part_title: 章节标题（格式："序号. 标题名称"）
    # - part_desc: 本部分内容简介（回答1 - 2个核心问题）
    # - part_content_type: 说明该部分内容的性质(例如：理论铺垫、数据论证、基础定义、价值论证、现状描述、格局分析等)
    # - part_key_output: 本部分需产出的关键结论或成果（例如：明确研究对象的核心定义、研究范围及研究必要性）
    # - part_data_source: 支撑该部分内容的典型数据 / 资料来源（例如：行业词典、政策文件、经典文献等）
    def get_user_prompt_part_input(self):

        part_title_name = self.cur_part_context.get_part_title_name()
        part_title = self.cur_part_context.get_cur_part_value("part_title")
        part_title_type = self.cur_part_context.get_cur_part_value("part_title_type")
        part_desc = self.cur_part_context.get_cur_part_value("part_desc")
        part_content_type = self.cur_part_context.get_cur_part_value("part_content_type")
        part_key_output = self.cur_part_context.get_cur_part_value("part_key_output")
        part_data_source = self.cur_part_context.get_cur_part_value("part_data_source")
        part_importance = self.cur_part_context.get_cur_part_value("part_importance")
        part_length_ratio = self.cur_part_context.get_cur_part_value("part_length_ratio")
        cur_content =self.cur_part_context.cur_content

        cur_part_title = self.cur_part_context.get_cur_part_value("part_title")
        cur_part_desc = self.cur_part_context.get_cur_part_value("part_desc")
        cur_part_central_idea  = self.cur_part_context.get_cur_part_value("part_central_idea")
        cur_part_content_type =  self.cur_part_context.get_cur_part_value("part_content_type")
        cur_part_key_output =  self.cur_part_context.get_cur_part_value("part_key_output")
        cur_part_data_source =  self.cur_part_context.get_cur_part_value("part_data_source")

        cur_subsection_title = self.cur_part_context.get_cur_subsection_value("subsection_title")
        cur_subsection_central_idea = self.cur_part_context.get_cur_part_value("subsection_central_idea")
        cur_subsection_desc = self.cur_part_context.get_cur_part_value("subsection_desc")
        prev_part_content_abstract = self.cur_part_context.get_prev_content_prompt()
        return {
            'part_title_name' :part_title_name,
            'part_title':  part_title,
            'part_title_type':  part_title_type,
            'part_desc':  part_desc,
            'part_content_type':part_content_type,
            'part_key_output':part_key_output,
            'part_data_source':part_data_source,
            'part_importance':part_importance,
            'part_length_ratio':part_length_ratio,
            'cur_content':cur_content,

            'prev_part_content':self.cur_part_context.prev_part_content,
            'report_data': self.rag_context,
            'rag_company': self.rag_company,
            'cur_part_content': self.cur_part_context.cur_part_content,

            'target_company':self.target_company,
            'report_outline': self.report_outline,

            'report_title': self.report_title,

            'cur_part_title': cur_part_title,
            'cur_part_central_idea': cur_part_central_idea,
            'cur_part_desc': cur_part_desc,
            'cur_part_content_type':cur_part_content_type,
            'cur_part_key_output':cur_part_key_output,
            'cur_part_data_source':cur_part_data_source,


            'cur_subsection_title': cur_subsection_title,
            'cur_subsection_central_idea': cur_subsection_central_idea,
            'cur_subsection_desc': cur_subsection_desc,
            'cur_subsection_content': self.cur_part_context.cur_subsection_content,
            'cur_subsection_content_opinion': self.cur_part_context.cur_subsection_content_opinion,
            'part_content_opinion': self.cur_part_context.cur_subsection_content_opinion,
            'prev_part_content_abstract': prev_part_content_abstract,

        }

    def create_report_content(self):
        init = self.report_content.init(self.rag_context, self.rag_company)

