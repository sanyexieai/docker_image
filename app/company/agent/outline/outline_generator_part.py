from typing import List, Dict, Any

from app.company.agent.base_agent import BaseOutlineAgent
from app.company.model.report_info import ReportInfo

from app.company.prompt.outline.generate_outline_part import USER_PROMPT_OUTLINE_EDIT_PART, \
    USER_PROMPT_OUTLINE_EDIT_MODIFY_PART, SYSTEM_PROMPT_OUTLINE_PART_EDIT_PART

from app.llm.schema import Message


class OutlineGeneratorPart(BaseOutlineAgent):
    """大纲生成器"""
    
    def __init__(self, logger, llm):
        super().__init__(logger, llm, SYSTEM_PROMPT_OUTLINE_PART_EDIT_PART)
    
    def generate(self, report_info: ReportInfo, **kwargs) -> List[Dict[str, Any]]:
        """生成研报大纲"""
        # 根据是否有意见选择不同的提示词
        if report_info.report_outline_opinion:
            self.logger.info("📋 正在修改研报大纲...")
            user_prompt = USER_PROMPT_OUTLINE_EDIT_MODIFY_PART.format(
                target_company=report_info.target_company,
                part_title=report_info.report_title,
                report_data=report_info.rag_company,
                report_outline = report_info.report_outline,
                report_outline_opinion=report_info.report_outline_opinion
            )
        else:
            self.logger.info("📋 正在生成初始研报大纲...")
            user_prompt = USER_PROMPT_OUTLINE_EDIT_PART.format(
                target_company=report_info.target_company,
                part_title=report_info.report_title,
                report_data=report_info.rag_company,
            )
        self.logger.info(f"📋 生成大纲的提示词：{user_prompt}")
        report_outline = self._execute_generation(user_prompt)
        report_info.report_outline = report_outline
        return report_outline
    
    def _execute_generation(self, user_prompt: str) -> List[Dict[str, Any]]:
        """执行生成逻辑"""
        messages = [Message.user_message(user_prompt)]
        self.memory.add_messages(messages)
        
        # Token 计算和检查
        input_tokens = self._count_tokens(self.messages)
        self.logger.info(f"📋 输入tokens: {input_tokens}")
        
        # 调用 LLM
        response = self.llm.ask(self.memory.messages, temperature=0.3)
        self.logger.info(f"📋 已生成研报大纲：{response}")
        self._reset_memory()
        return self._parse_yaml_response(response)