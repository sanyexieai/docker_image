#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档格式转换流程
处理markdown文件中的图片，并转换为Word格式文档
"""

import os
import glob
import time
import json
import re
import shutil
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlparse

from app.utils.markdown_tools import convert_to_docx, format_markdown

class DocumentConversionPipeline:
    """文档格式转换流程类"""
    
    def __init__(self):
        # 配置日志记录
        self.setup_logging()
        
        # 环境变量与全局配置
        load_dotenv()
        
        self.logger.info("🔧 文档格式转换流程初始化完成")
    
    def setup_logging(self):
        """配置日志记录"""
        # 创建logs目录
        os.makedirs("logs", exist_ok=True)
        
        # 生成日志文件名（包含时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"logs/document_conversion_{timestamp}.log"
        
        # 配置日志记录器
        self.logger = logging.getLogger('DocumentConversion')
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
    
    def ensure_dir(self, path):
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def is_url(self, path):
        """判断是否为URL"""
        return path.startswith('http://') or path.startswith('https://')
    
    def download_image(self, url, save_path):
        """下载图片"""
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            return True
        except Exception as e:
            self.logger.error(f"[下载失败] {url}: {e}")
            return False
    
    def copy_image(self, src, dst):
        """复制图片"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"[复制失败] {src}: {e}")
            return False
    
    def extract_images_from_markdown(self, md_path, images_dir, new_md_path):
        """从markdown中提取图片"""
        self.logger.info("🖼️ 处理markdown中的图片...")
        self.ensure_dir(images_dir)
        
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 匹配 ![alt](path) 形式的图片
        pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        matches = pattern.findall(content)
        used_names = set()
        replace_map = {}
        not_exist_set = set()

        for img_path in matches:
            img_path = img_path.strip()
            # 取文件名
            if self.is_url(img_path):
                filename = os.path.basename(urlparse(img_path).path)
            else:
                filename = os.path.basename(img_path)
            # 防止重名
            base, ext = os.path.splitext(filename)
            i = 1
            new_filename = filename
            while new_filename in used_names:
                new_filename = f"{base}_{i}{ext}"
                i += 1
            used_names.add(new_filename)
            new_img_path = os.path.join(images_dir, new_filename)
            # 下载或复制
            img_exists = True
            if self.is_url(img_path):
                success = self.download_image(img_path, new_img_path)
                if not success:
                    img_exists = False
            else:
                # 支持绝对和相对路径
                abs_img_path = img_path
                if not os.path.isabs(img_path):
                    abs_img_path = os.path.join(os.path.dirname(md_path), img_path)
                if not os.path.exists(abs_img_path):
                    self.logger.warning(f"[警告] 本地图片不存在: {abs_img_path}")
                    img_exists = False
                else:
                    self.copy_image(abs_img_path, new_img_path)
            # 记录替换
            if img_exists:
                replace_map[img_path] = f'./images/{new_filename}'
            else:
                not_exist_set.add(img_path)

        # 替换 markdown 内容，不存在的图片直接删除整个图片语法
        def replace_func(match):
            orig = match.group(1).strip()
            if orig in not_exist_set:
                return ''  # 删除不存在的图片语法
            return match.group(0).replace(orig, replace_map.get(orig, orig))

        new_content = pattern.sub(replace_func, content)
        with open(new_md_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        self.logger.info(f"✅ 图片处理完成！新文件: {new_md_path}")

        # 记录未能插入markdown的图片信息
        if not_exist_set:
            for img_path in not_exist_set:
                self.logger.error(f"图片未能插入markdown，原因：下载/复制失败或文件不存在。原始路径: {img_path}")
        
        return new_md_path
    
    def format_markdown(self, md_path):
        """格式化markdown文件"""
        self.logger.info("🎨 格式化markdown文件...")
        try:
            format_markdown(md_path)
            self.logger.info("✅ markdown格式化完成")
            return True
        except Exception as e:
            self.logger.error(f"❌ markdown格式化失败: {e}")
            return False
    
    def convert_to_word(self, md_path, docx_output=None):
        """转换为Word文档"""
        self.logger.info("📄 转换为Word文档...")
        
        if docx_output is None:
            docx_output = md_path.replace('.md', '.docx')
        
        try:
            convert_to_docx(md_path, docx_output=docx_output)
            self.logger.info(f"✅ Word文档转换完成: {docx_output}")
            return docx_output
        except Exception as e:
            self.logger.error(f"❌ Word文档转换失败: {e}")
            return None
    
    def process_markdown_file(self, md_path,word_output_path=None):
        """处理单个markdown文件"""
        self.logger.info(f"📁 处理markdown文件: {md_path}")
        
        if not os.path.exists(md_path):
            self.logger.error(f"❌ 文件不存在: {md_path}")
            return None
        
        try:
            # 1. 处理图片
            images_dir = os.path.join(os.path.dirname(md_path), 'images')
            new_md_path = md_path.replace('.md', '_images.md')
            processed_md_path = self.extract_images_from_markdown(md_path, images_dir, new_md_path)
            
            # 2. 格式化markdown
            self.format_markdown(processed_md_path)
            
            # 3. 转换为Word文档
            docx_path = self.convert_to_word(processed_md_path,word_output_path)
            
            return {
                'original_md': md_path,
                'processed_md': processed_md_path,
                'docx': docx_path,
                'images_dir': images_dir
            }
            
        except Exception as e:
            self.logger.error(f"❌ 处理markdown文件失败: {e}")
            return None
    
    def find_latest_markdown(self, pattern="*深度研报_*.md"):
        """查找最新的markdown文件"""
        files = glob.glob(pattern)
        if not files:
            return None
        
        # 按修改时间排序，取最新的
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]
    
    def run_conversion(self, md_path=None,word_output_path=None):
        """运行文档转换流程"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 开始文档格式转换流程")
        self.logger.info("="*80)
        
        try:
            # 如果没有指定文件，查找最新的markdown文件
            if md_path is None:
                md_path = self.find_latest_markdown()
                if md_path is None:
                    self.logger.error("❌ 未找到markdown文件")
                    return None
                self.logger.info(f"📁 自动选择文件: {md_path}")
            
            # 处理markdown文件
            result = self.process_markdown_file(md_path,word_output_path)
            
            if result:
                self.logger.info("\n✅ 文档转换流程完成！")
                self.logger.info(f"📄 原始文件: {result['original_md']}")
                self.logger.info(f"📝 处理后文件: {result['processed_md']}")
                self.logger.info(f"📋 Word文档: {result['docx']}")
                self.logger.info(f"🖼️ 图片目录: {result['images_dir']}")
                return result
            else:
                self.logger.error("❌ 文档转换流程失败")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 文档转换流程失败: {e}")
            return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文档格式转换流程')
    parser.add_argument('--input', help='输入的markdown文件路径（可选，不指定则自动查找最新文件）')
    parser.add_argument('--output', help='输出的Word文件路径（可选）')
    
    args = parser.parse_args()
    
    # 创建文档转换实例
    pipeline = DocumentConversionPipeline()
    
    # 运行文档转换流程
    result = pipeline.run_conversion(args.input)
    
    if result:
        print(f"\n🎉 文档转换流程执行完毕！")
        print(f"📄 原始文件: {result['original_md']}")
        print(f"📝 处理后文件: {result['processed_md']}")
        print(f"📋 Word文档: {result['docx']}")
        print(f"🖼️ 图片目录: {result['images_dir']}")
    else:
        print("\n❌ 文档转换流程执行失败！")


if __name__ == "__main__":
    main() 