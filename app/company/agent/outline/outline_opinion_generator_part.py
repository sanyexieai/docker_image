from typing import List, Dict, Any
from app.company.agent.base_agent import BaseOutlineAgent
from app.company.prompt.outline.generate_outline_edit_opinion_part import SYSTEM_PROMPT_OUTLINE_EDIT_OPINION_PART, \
    USER_PROMPT_OUTLINE_EDIT_OPINION_PART
from app.company.model.report_info import ReportInfo

from app.llm.schema import Message


class OutlineOpinionGeneratorPart(BaseOutlineAgent):
    """大纲意见生成器"""
    
    def __init__(self, logger, llm):
        super().__init__(logger, llm, SYSTEM_PROMPT_OUTLINE_EDIT_OPINION_PART)
    
    def generate(self, report_info: ReportInfo, **kwargs) -> List[Dict[str, Any]]:
        """生成大纲意见"""
        report_outline = report_info.report_outline
        if not report_outline:
            self.logger.info("未找到大纲，跳过")
            return []
        
        self.logger.info("📋 正在生成研报大纲意见...")

        # part_input = report_info.get_user_prompt_part_input()

        user_prompt = USER_PROMPT_OUTLINE_EDIT_OPINION_PART.format(
            target_company=report_info.target_company,
            part_title=report_info.report_title,
            report_outline=report_outline,
            report_data=report_info.rag_context,
        )
        self.logger.info(f"📋 正在生成研报大纲意见: {user_prompt}")
        messages = [Message.user_message(user_prompt)]
        self.memory.add_messages(messages)
        
        input_tokens = self._count_tokens(self.messages)
        self.logger.info(f"📋 输入tokens: {input_tokens}")
        
        response = self.llm.ask(self.memory.messages, temperature=0.3)
        self.logger.info(f"📋 已生成研报大纲意见：{response}")
        
        result = self._parse_yaml_response(response)
        report_info.report_outline_opinion = result

        self._reset_memory()
        return result