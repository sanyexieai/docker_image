"""
文档转换工具模块
提供Markdown转Word等功能（使用pypandoc库）
"""

import os
import re
from typing import Optional

try:
    import pypandoc
except ImportError:
    pypandoc = None
    print("[警告] pypandoc未安装。请运行: pip install pypandoc")

def convert_to_docx_basic(input_file: str, docx_output: Optional[str] = None) -> str:
    """
    将Markdown文件转换为Word文档（基本版本）
    
    Args:
        input_file (str): 输入的Markdown文件路径
        docx_output (str, optional): 输出的Word文件路径，默认使用与输入文件相同的名称但扩展名为.docx
        
    Returns:
        str: 输出文件的路径，如果转换失败则返回None
    """
    if not os.path.exists(input_file):
        print(f"[错误] 输入文件不存在: {input_file}")
        return None

    if docx_output is None:
        docx_output = os.path.splitext(input_file)[0] + '.docx'
    
    if pypandoc is None:
        print("[错误] pypandoc库未安装，无法进行转换")
        print("请运行以下命令安装：")
        print("pip install pypandoc")
        return None
    
    try:
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 预处理：将表格分隔线中的全角破折号'—'替换为半角'-'，确保pandoc能识别。
        processed_content = content.replace('—', '-')

        print(f"正在使用pypandoc进行转换，输出至: {docx_output}")
        
        # 使用pypandoc直接转换
        pypandoc.convert_text(
            processed_content,
            'docx',
            format='md',
            outputfile=docx_output,
            extra_args=[
                '--standalone',
                '--resource-path=.'
            ]
        )
        
        print(f"\n📄 Word版报告已生成: {docx_output}")
        return docx_output

    except Exception as e:
        print(f"[提示] pypandoc转换失败。错误信息: {e}")
        print("请确保：")
        print("1. pypandoc已正确安装: pip install pypandoc")
        print("2. pandoc引擎已安装: pypandoc.download_pandoc() 或访问 https://pandoc.org/installing.html")
    
    return None

def convert_to_docx_with_indent(input_file: str, docx_output: Optional[str] = None) -> str:
    """
    将Markdown文件转换为Word文档（带自定义样式的高级版本）。
    使用一个预先配置好的 `reference.docx` 文件来应用所有样式，包括字体、缩进、表格边框等。

    Args:
        input_file (str): 输入的Markdown文件路径
        docx_output (str, optional): 输出的Word文件路径，默认与输入文件同名

    Returns:
        str: 输出文件的路径，如果转换失败则返回None
    """
    if not os.path.exists(input_file):
        print(f"[错误] 输入文件不存在: {input_file}")
        return None

    if docx_output is None:
        docx_output = os.path.splitext(input_file)[0] + '.docx'

    if pypandoc is None:
        print("[错误] pypandoc库未安装，无法进行转换")
        print("请运行以下命令安装：")
        print("pip install pypandoc")
        return None

    # 定义参考文档的路径。这个文件现在应该是预先配置好的。
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    reference_docx = os.path.join(utils_dir, "reference.docx")

    if not os.path.exists(reference_docx):
        print(f"❌ [错误] 参考文档 'reference.docx' 不存在于: {utils_dir}")
        print("➡️ [操作建议] 请先在终端运行 `python create_reference.py` 脚本来生成默认的参考文档。")
        print("             然后您可以在Word中打开并编辑它，以定义您自己的样式。")
        print("---------------------------------------------------------------------")
        print("⚠️ [自动回退] 由于缺少样式文件，将使用无格式的基础模式进行转换。")
        return convert_to_docx_basic(input_file, docx_output)

    # --- 使用配置好的参考文档进行转换 ---
    try:
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 预处理：将表格分隔线中的全角破折号'—'替换为半角'-'，确保pandoc能识别。
        processed_content = content.replace('—', '-')

        print(f'正在使用自定义参考文档进行转换: {reference_docx}')
        print(f"正在调用pypandoc进行转换，输出至: {docx_output}")
        
        # 使用pypandoc和参考文档进行转换
        pypandoc.convert_text(
            processed_content,
            'docx',
            format='md',
            outputfile=docx_output,
            extra_args=[
                '--standalone',
                '--resource-path=.',
                f'--reference-doc={reference_docx}'
            ]
        )

        print(f"\n📄 Word版报告已生成: {docx_output}")
        return docx_output

    except Exception as e:
        print(f"[提示] pypandoc转换失败。错误信息: {e}")
        print("请确保：")
        print("1. pypandoc已正确安装: pip install pypandoc")
        print("2. pandoc引擎已安装: pypandoc.download_pandoc() 或访问 https://pandoc.org/installing.html")
        print("3. 参考文档格式正确且可访问")

    return None

def install_pandoc_if_needed():
    """
    检查并自动安装pandoc引擎（如果需要）
    """
    if pypandoc is None:
        print("[错误] pypandoc库未安装")
        return False
    
    try:
        # 测试pypandoc是否能正常工作
        pypandoc.convert_text('# Test', 'html', format='md')
        print("✅ pypandoc工作正常")
        return True
    except Exception as e:
        print(f"[警告] pypandoc无法正常工作: {e}")
        print("正在尝试自动下载pandoc引擎...")
        try:
            pypandoc.download_pandoc()
            print("✅ pandoc引擎下载成功")
            return True
        except Exception as download_error:
            print(f"[错误] 自动下载失败: {download_error}")
            print("请手动安装pandoc:")
            print("- macOS: brew install pandoc")
            print("- Windows: 访问 https://pandoc.org/installing.html")
            print("- Linux: sudo apt-get install pandoc")
            return False

if __name__ == "__main__":
    import os
    import sys

    # 检查pypandoc是否可用
    if not install_pandoc_if_needed():
        print("无法继续，请先安装必要的依赖")
        sys.exit(1)

    # 构建到Markdown文件的绝对路径，确保无论从哪里运行脚本都能找到文件
    # 1. 获取当前脚本(doc_converter.py)所在的目录
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # 2. 从脚本目录向上回溯三级，到达项目根目录
    project_root = os.path.abspath(os.path.join(script_directory, '..', '..', '..'))
    # 3. 拼接得到目标Markdown文件的完整路径 (现在位于reports目录下)
    md_file = os.path.join(project_root, 'reports', 'Industry_Research_Report.md')

    print(f"目标文件路径: {md_file}")
    
    if os.path.exists(md_file):
        convert_to_docx_with_indent(md_file)
    else:
        print(f"文件不存在: {md_file}")
        # 尝试使用商汤科技研报作为测试
        test_file = os.path.join(project_root, '商汤科技深度研报_20250711_103844.md')
        if os.path.exists(test_file):
            print(f"使用测试文件: {test_file}")
            convert_to_docx_with_indent(test_file)
        else:
            print("未找到可用的测试文件")