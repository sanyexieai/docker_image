import subprocess
import sys
import os

# 定义要依次运行的脚本及其参数
scripts = [
    ("run_company_research_report.py", ["--company_name", "商汤科技", "--company_code", "00020.HK"]),
    ("run_industry_research_report.py", ["--industry_name", "中国智能服务机器人产业"]),
    ("run_marco_research_report.py", ["--marco_name", "国家级'人工智能+'政策效果评估", "--time", "2023-2025"]),
]

# 获取项目根目录
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for script, args in scripts:
        print(f"\n===== 正在运行: {script} =====\n")
        script_path = os.path.join(project_root, script)
        if not os.path.exists(script_path):
            print(f"未找到脚本: {script_path}")
            continue
        try:
            result = subprocess.run(
                [sys.executable, script_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=project_root
            )
            print(f"[stdout]\n{result.stdout}")
            if result.stderr:
                print(f"[stderr]\n{result.stderr}")
            if result.returncode != 0:
                print(f"脚本 {script} 执行失败，退出码: {result.returncode}")
        except Exception as e:
            print(f"运行 {script} 时发生异常: {e}")

if __name__ == "__main__":
    main()
