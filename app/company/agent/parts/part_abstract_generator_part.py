from app.company.agent.base_agent import BaseOutlineAgent
from app.company.prompt.parts.generate_part_abstract_part import SYSTEM_PROMPT_PART_ABSTRACT_PART, \
    USER_PROMPT_PART_ABSTRACT_PART
from app.company.model.report_info import ReportInfo
from app.llm.schema import Message


class PartAbstractGeneratorPart(BaseOutlineAgent):
    """大纲生成器"""
    
    def __init__(self, logger, llm):
        super().__init__(logger, llm, SYSTEM_PROMPT_PART_ABSTRACT_PART)
    
    def generate(self, report_info: ReportInfo, **kwargs) -> str:

        report_text_list = report_info.report_text_list
        if not report_text_list:
            self.logger.info("📋 没有文章，跳过")
            return ""

        self.logger.info("📋 正在生成正文章节摘要...")
        user_prompt = USER_PROMPT_PART_ABSTRACT_PART.format(
            report_text_list = report_text_list,
        )
        self.logger.info(f"📋 生成正文章节摘要提示词: {user_prompt}")

        generation = self._execute_generation(user_prompt)
        report_info.cur_part_context.prev_part_content_abstract = generation
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
        self.logger.info(f"📋 已正文章节摘要：{response}")
        self._reset_memory()
        return response