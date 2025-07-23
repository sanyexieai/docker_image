



import re
from typing import List, Dict, Any, Optional


class ContentConvert:
    """内容转换器，用于生成目录结构"""
    
    def __init__(self, parts: List[Dict[str, Any]]) -> None:
        """
        初始化内容转换器
        
        Args:
            parts: 包含部分信息的字典列表
            
        Raises:
            TypeError: 当parts不是列表时
            ValueError: 当parts为空时
        """
        if not isinstance(parts, list):
            raise TypeError("parts 必须是一个列表")
        if not parts:
            raise ValueError("parts 不能为空")
        self.parts = parts

    def get_content_list_1(self) -> List[str]:
        content_lines = ["## 目录\n"]
        for idx, part in enumerate(self.parts):
            part_title = part.get('part_title', f'部分{idx + 1}')
                
            part_title_type = part.get('part_title_type', f'部分{idx + 1}')
            if part_title_type == '章':
                part_title = self._clean_trailing_dot(part_title)
                # 使用 _create_anchor 方法生成正确的锚点
                anchor = self._create_anchor(part_title)
                content_lines.append(f"- [{part_title}](#{anchor})")
            elif part_title_type == '节':
                anchor = self._create_anchor(part_title)
                content_lines.append(f"  - [{part_title}](#{anchor})")
            elif part_title_type == '小节':
                anchor = self._create_anchor(part_title)
                content_lines.append(f"   - [{part_title}](#{anchor})")
        
        # 引用文献的锚点也需要正确格式化
        ref_anchor = self._create_anchor("引用文献")
        content_lines.append(f"- [引用文献](#{ref_anchor})")
    
        return content_lines
    
    def _clean_trailing_dot(self, title: str) -> str:
        """
        清理标题中数字尾部的点号
        
        Args:
            title: 原始标题
            
        Returns:
            str: 清理后的标题
        """
        # 使用正则表达式匹配数字后面的点号并移除
        # 匹配模式：数字 + 点号 + 可能的空格 + 其他内容
        cleaned_title = re.sub(r'(\d+)\.\s*', r'\1 ', title)
        return cleaned_title.strip()

    def get_content_list(self) -> List[str]:
        """
        生成目录列表的 Markdown 格式字符串列表
        
        Returns:
            List[str]: 格式化的目录字符串列表
        """
        content_lines = ["## 目录\n"]

        for idx, part in enumerate(self.parts):
            if not isinstance(part, dict):
                continue

            part_title = part.get('part_title', f'部分{idx + 1}')
            part_num = part.get('part_num', f'部分{idx + 1}')

            # 生成锚点链接
            anchor = self._create_anchor(f"{part_num}-{part_title}")
            content_lines.append(f"- [{part_title}](#{anchor})")

            # 处理子章节
            subsections = part.get('subsections', [])
            if isinstance(subsections, list) and subsections:
                subsection_lines = self._process_subsections(subsections)
                content_lines.extend(subsection_lines)

        return content_lines
    
    def _process_subsections(self, subsections: List[Dict[str, Any]]) -> List[str]:
        """
        处理子章节
        
        Args:
            subsections: 子章节列表
            
        Returns:
            List[str]: 格式化的子章节字符串列表
        """
        subsection_lines = []
        
        for idx, subsection in enumerate(subsections):
            if not isinstance(subsection, dict):
                continue
                
            subsection_title = subsection.get('subsection_title', f'子章节{idx + 1}')
            subsection_num = subsection.get('subsection_num', f'子章节{idx + 1}')
            
            # 生成锚点链接
            anchor = self._create_anchor(f"{subsection_num}-{subsection_title}")
            subsection_lines.append(f"  - [{subsection_title}](#{anchor})")
        
        return subsection_lines
    
    def _create_anchor(self, text: str) -> str:
        """
        创建 Markdown 锚点链接
        
        Args:
            text: 原始文本
            
        Returns:
            str: 处理后的锚点文本
        """
        # 转换为小写
        anchor = text.lower()
        # 移除点号，保留空格暂时作为分隔符
        anchor = re.sub(r'\.', '', anchor)  # 移除点号
        anchor = re.sub(r'\s+', '-', anchor)  # 空格转为连字符
        anchor = re.sub(r'[^a-z0-9\u4e00-\u9fff\-]', '-', anchor)  # 其他特殊字符转连字符
        anchor = re.sub(r'-+', '-', anchor)  # 合并多个连字符
        return anchor.strip('-')
    
    def validate_structure(self) -> bool:
        """
        验证数据结构的有效性
        
        Returns:
            bool: 结构是否有效
        """
        if not self.parts:
            return False
            
        for part in self.parts:
            if not isinstance(part, dict):
                return False
            if not any(key in part for key in ['part_title', 'part_num']):
                return False
                
        return True
    
    def get_parts_count(self) -> int:
        """
        获取部分数量
        
        Returns:
            int: 部分数量
        """
        return len(self.parts)
    
    def get_subsections_count(self) -> int:
        """
        获取所有子章节的总数量
        
        Returns:
            int: 子章节总数量
        """
        return sum(
            len(part.get('subsections', []))
            for part in self.parts
            if isinstance(part, dict) and isinstance(part.get('subsections'), list)
        )
    
    def get_part_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        根据索引获取部分信息
        
        Args:
            index: 部分索引
            
        Returns:
            Optional[Dict[str, Any]]: 部分信息，如果索引无效则返回None
        """
        if 0 <= index < len(self.parts):
            return self.parts[index]
        return None
    
    def add_part(self, part: Dict[str, Any]) -> None:
        """
        添加新的部分
        
        Args:
            part: 部分信息字典
            
        Raises:
            TypeError: 当part不是字典时
        """
        if not isinstance(part, dict):
            raise TypeError("part 必须是一个字典")
        self.parts.append(part)
    
    def remove_part(self, index: int) -> bool:
        """
        根据索引移除部分
        
        Args:
            index: 要移除的部分索引
            
        Returns:
            bool: 是否成功移除
        """
        if 0 <= index < len(self.parts):
            self.parts.pop(index)
            return True
        return False




