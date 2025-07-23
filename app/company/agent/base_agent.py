from abc import ABC, abstractmethod
from typing import List, Dict, Any
import tiktoken
import yaml
from app.llm.schema import Message, Memory
from app.company.agent.token_counter import TokenCounter
from app.report_info import ReportInfo

class BaseOutlineAgent(ABC):
    """大纲生成 Agent 基类"""
    
    def __init__(self, logger, llm, system_prompt: str):
        self.logger = logger
        self.llm = llm
        self.system_prompt = system_prompt
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = llm.config.max_tokens
        
        # 统一使用 Memory 管理消息
        self.memory = Memory()
        self._initialize_memory()
        
        # 统一的 tokenizer 初始化
        self._initialize_tokenizer()
        
    def _initialize_memory(self):
        """初始化消息记忆"""
        messages = [Message.system_message(self.system_prompt)]
        self.memory.add_messages(messages)
        
    def _initialize_tokenizer(self):
        """初始化 tokenizer"""
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.llm.config.model)
        except KeyError:
            self.logger.warning(f"模型 {self.llm.config.model} 不被 tiktoken 支持，使用默认编码 cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.token_counter = TokenCounter(self.tokenizer)
    
    def _parse_yaml_response(self, response: str) -> List[Dict[str, Any]]:
        """统一的 YAML 解析逻辑"""
        try:
            if '```yaml' in response:
                yaml_block = response.split('```yaml')[1].split('```')[0]
            else:
                yaml_block = response
            
            result = yaml.safe_load(yaml_block)
            if isinstance(result, dict):
                result = list(result.values())
            
            self.logger.info(f"📄 解析结果: {result}")
            return result
        except Exception as e:
            self.logger.error(f"[YAML解析失败] {e}")
            return []
    
    def _count_tokens(self, messages: List[Message]) -> int:
        """计算 token 数量"""
        messages_dict = [msg.to_dict() for msg in messages]
        return self.token_counter.count_message_tokens(messages_dict)
    
    def _check_token_limit(self, input_tokens: int) -> bool:
        """检查 token 限制"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        return True
    def _reset_memory(self) -> None:
        """重置内存并添加系统消息"""
        self.memory.clear()
        messages = [Message.system_message(self.system_prompt)]
        self.memory.add_messages(messages)

    def _user_prompt(self,user_prompt):
        self.logger.info(f"📄 用户提示词: {user_prompt}")

    @property
    def messages(self) -> List[Message]:
        """获取消息列表"""
        return self.memory.messages
    
    @messages.setter
    def messages(self, value: List[Message]):
        """设置消息列表"""
        self.memory.messages = value
    
    @abstractmethod
    def generate(self, report_info: ReportInfo, **kwargs) -> List[Dict[str, Any]]:
        """生成大纲的抽象方法"""
        pass