from app.company.agent.base_agent import BaseOutlineAgent
from app.company.prompt.parts.generate_part_edit_part import USER_PROMPT_PART_EDIT_PART, \
    USER_PROMPT_PART_EDIT_MODIFY_PART, SYSTEM_PROMPT_PART_EDIT_PART, USER_PROMPT_PART_EDIT_END_PART
from app.company.model.report_info import ReportInfo
from app.llm.schema import Message


class PartGeneratorPart(BaseOutlineAgent):
    """大纲生成器"""
    
    def __init__(self, logger, llm):
        super().__init__(logger, llm, SYSTEM_PROMPT_PART_EDIT_PART)
    
    def generate(self, report_info: ReportInfo, **kwargs) -> str:
        prompt_input = report_info.get_user_prompt_part_input()
        # part = report_info.map_dict_to_cur_part()

        # 修复：添加空格
        # 根据是否有意见选择不同的提示词
        cur_subsection_content_opinion = report_info.cur_part_context.cur_subsection_content_opinion
        if cur_subsection_content_opinion:
            self.logger.info("📋 正在修改正文...")

            user_prompt = USER_PROMPT_PART_EDIT_MODIFY_PART.format(
                **prompt_input
            )
        else:
            if  report_info.cur_part_context.is_report_last:
                self.logger.info("📋 正在生成最后正文...")
                user_prompt = USER_PROMPT_PART_EDIT_END_PART.format(
                    **prompt_input
                )
                user_prompt += """
                   请在本节最后以"##  引用文献"格式，列出所有正文中用到的参考资料，格式如下：
                   [1] 东方财富-港股-财务报表: https://emweb.securities.eastmoney.com/PC_HKF10/FinancialAnalysis/index
                   [2] 同花顺-主营介绍: https://basic.10jqka.com.cn/new/000066/operate.html
                   [3] 同花顺-股东信息: https://basic.10jqka.com.cn/HK0020/holder.html
                   """
            else:
                self.logger.info("📋 正在生成初始正文...")
                user_prompt = USER_PROMPT_PART_EDIT_PART.format(
                    **prompt_input
                )


        self.logger.info(f"📋 生成正文提示词: {user_prompt}")

        generation = self._execute_generation(user_prompt)
        report_info.cur_part_context.cur_subsection_content = generation
        report_info.cur_part_context.cur_content = generation
        return generation
    
    def _execute_generation(self, user_prompt: str) -> str:
        """执行生成逻辑"""
        messages = [Message.user_message(user_prompt)]
        self.memory.add_messages(messages)
        
        # Token 计算和检查
        input_tokens = self._count_tokens(self.messages)
        self.logger.info(f"📋 输入tokens: {input_tokens}")
        
        # 调用 LLM
        response = self.llm.ask(self.memory.messages, temperature=0.3)
        self.logger.info(f"📋 已生成正文：{response}")
        self._reset_memory()
        return response