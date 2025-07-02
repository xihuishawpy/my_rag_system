#!/usr/bin/env python3
"""
RAG系统Web界面启动脚本
用于启动Streamlit Web界面
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🚀 启动RAG系统Web界面...")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查Web界面文件
    web_interface_path = Path("src/web_interface.py")
    if not web_interface_path.exists():
        print(f"❌ 找不到Web界面文件: {web_interface_path}")
        return
    
    # 启动Streamlit
    try:
        print("🌐 正在启动Web服务器...")
        print("📱 界面将在浏览器中自动打开")
        print("🔗 访问地址: http://localhost:8501")
        print("⏹️  按 Ctrl+C 停止服务")
        print("-" * 50)
        
        # 启动streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(web_interface_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Web界面已停止")
    except Exception as e:
        print(f"❌ 启动Web界面失败: {str(e)}")


if __name__ == "__main__":
    main() 