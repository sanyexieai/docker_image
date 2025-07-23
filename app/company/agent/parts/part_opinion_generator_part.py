
from typing import List, Dict, Any

from app.company.agent.base_agent import BaseOutlineAgent
from app.company.model.report_info import ReportInfo
from app.company.prompt.parts.generate_part_edit_opinion_part import USER_PROMPT_PART_EDIT_OPINION_PART, \
    SYSTEM_PROMPT_PART_EDIT_OPINION_PART
from app.llm.schema import Message


class PartOpinionGeneratorPart(BaseOutlineAgent):
    """大纲意见生成器"""
    
    def __init__(self, logger, llm):
        super().__init__(logger, llm, SYSTEM_PROMPT_PART_EDIT_OPINION_PART)
    
    def generate(self, report_info: ReportInfo, **kwargs) -> List[Dict[str, Any]]:
        """生成文章正文意见"""
        if not report_info.cur_part_context.cur_content:
            self.logger.info("无正文内容意见，跳过")
            return []
        
        self.logger.info("📋 正在生成正文内容意见...")
        user_input = report_info.get_user_prompt_part_input()
        user_prompt = USER_PROMPT_PART_EDIT_OPINION_PART.format(
            **user_input
        )
        self.logger.info(f"📋 正在生成正文内容意见: {user_prompt}")
        messages = [Message.user_message(user_prompt)]
        self.memory.add_messages(messages)
        
        input_tokens = self._count_tokens(self.messages)
        self.logger.info(f"📋 输入tokens: {input_tokens}")
        
        response = self.llm.ask(self.memory.messages, temperature=0.3)
        self.logger.info(f"📋 已生成正文内容意见：{response}")
        
        result = self._parse_yaml_response(response)
        report_info.cur_part_context.cur_subsection_content_opinion = result

        self._reset_memory()
        return result