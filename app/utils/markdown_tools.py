import logging
import subprocess
import os



def format_markdown(output_file):
    try:
        import subprocess
        format_cmd = ["mdformat", output_file]
        subprocess.run(format_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logger = logging.getLogger('InDepthResearch')
        logger.info(f"✅ 已用 mdformat 格式化 Markdown 文件: {output_file}")
    except Exception as e:
        logger = logging.getLogger('InDepthResearch')
        logger.error(f"[提示] mdformat 格式化失败: {e}\n请确保已安装 mdformat (pip install mdformat)")

def convert_to_docx(output_file, docx_output):
    try:
        import subprocess
        import os
        pandoc_cmd = [
            "pandoc",
            output_file,
            "-o",
            docx_output,
            "--standalone",
            "--resource-path=.",
            "--extract-media=."
        ]
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run(pandoc_cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        logger = logging.getLogger('InDepthResearch')
        logger.info(f"\n📄 Word版报告已生成: {docx_output}")
    except subprocess.CalledProcessError as e:
        logger = logging.getLogger('InDepthResearch')
        logger.error(f"[提示] pandoc转换失败。错误信息: {e.stderr}")
        logger.warning("[建议] 检查图片路径是否正确，或使用 --extract-media 选项")
    except Exception as e:
        logger = logging.getLogger('InDepthResearch')
        logger.error(f"[提示] 若需生成Word文档，请确保已安装pandoc。当前转换失败: {e}")
