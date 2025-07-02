@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 商城RAG系统启动脚本 (Windows版本)
:: 自动检查环境并启动Web界面

echo ==========================================
echo       商城RAG知识库问答系统启动
echo ==========================================
echo.

:: 检查Python环境
echo [INFO] 检查Python环境...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python未安装，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo [SUCCESS] Python版本: !python_version!

:: 检查pip
echo [INFO] 检查pip包管理器...
pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] pip未安装或不可用
    pause
    exit /b 1
)

:: 检查requirements.txt
echo [INFO] 检查依赖文件...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt文件不存在
    pause
    exit /b 1
)

:: 检查关键依赖包
echo [INFO] 检查Python依赖包...
set missing_packages=

python -c "import streamlit" >nul 2>&1
if !errorlevel! neq 0 set missing_packages=!missing_packages! streamlit

python -c "import chromadb" >nul 2>&1
if !errorlevel! neq 0 set missing_packages=!missing_packages! chromadb

python -c "import faiss" >nul 2>&1
if !errorlevel! neq 0 set missing_packages=!missing_packages! faiss-cpu

if not "!missing_packages!"=="" (
    echo [WARNING] 缺少依赖包:!missing_packages!
    set /p install_deps="是否自动安装? (Y/n): "
    if /i "!install_deps!"=="n" (
        echo [ERROR] 请手动安装依赖包: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [INFO] 正在安装依赖包...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] 依赖包安装失败
        pause
        exit /b 1
    )
    echo [SUCCESS] 依赖包安装完成
) else (
    echo [SUCCESS] 所有依赖包已安装
)

:: 检查API密钥
echo [INFO] 检查API密钥配置...
if "%DASHSCOPE_API_KEY%"=="" (
    echo [WARNING] DASHSCOPE_API_KEY环境变量未设置
    
    :: 检查.env文件
    if exist ".env" (
        echo [INFO] 发现.env文件，正在加载...
        for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
            if "%%a"=="DASHSCOPE_API_KEY" set DASHSCOPE_API_KEY=%%b
        )
        if not "!DASHSCOPE_API_KEY!"=="" (
            echo [SUCCESS] 从.env文件加载API密钥
        )
    )
    
    if "!DASHSCOPE_API_KEY!"=="" (
        echo [ERROR] 请设置DASHSCOPE_API_KEY环境变量
        echo 方法1: set DASHSCOPE_API_KEY=your_api_key
        echo 方法2: 创建.env文件并添加 DASHSCOPE_API_KEY=your_api_key
        echo 获取密钥: https://bailian.console.aliyun.com/
        pause
        exit /b 1
    )
) else (
    echo [SUCCESS] API密钥已配置
)

:: 创建必要的目录
echo [INFO] 检查数据目录...
if not exist "vector_store" mkdir vector_store
if not exist "cache" mkdir cache
if not exist "qa_cache" mkdir qa_cache
if not exist "embeddings_cache" mkdir embeddings_cache
if not exist "logs" mkdir logs

:: 检查是否有数据
set has_data=false

if exist "vector_store\chroma" (
    dir /b "vector_store\chroma" 2>nul | findstr "." >nul
    if !errorlevel! equ 0 (
        echo [SUCCESS] 发现ChromaDB数据
        set has_data=true
    )
)

if exist "vector_store\faiss_index.bin" (
    echo [SUCCESS] 发现FAISS索引数据
    set has_data=true
)

if exist "processed_documents.json" (
    echo [SUCCESS] 发现处理后的文档数据
    set has_data=true
)

if "!has_data!"=="false" (
    echo [WARNING] 未发现现有数据
    echo [INFO] 首次使用请先运行数据处理或快速设置:
    echo   python src\run_data_processing.py  # 处理Excel数据
    echo   python src\quick_setup.py         # 快速设置测试数据
    echo.
    set /p setup_data="是否运行快速设置? (Y/n): "
    if /i not "!setup_data!"=="n" (
        echo [INFO] 正在运行快速设置...
        python src\quick_setup.py
        if !errorlevel! equ 0 (
            echo [SUCCESS] 快速设置完成
        )
    )
)

:: 检查Web界面文件
echo [INFO] 检查Web界面文件...
if not exist "src\web_interface.py" (
    echo [ERROR] Web界面文件不存在: src\web_interface.py
    pause
    exit /b 1
)

:: 启动Web界面
echo.
echo ==========================================
echo [INFO] 启动RAG系统Web界面...
echo [INFO] 浏览器将自动打开 http://localhost:8501
echo.
echo [INFO] 使用Ctrl+C停止服务
echo ==========================================
echo.

streamlit run src\web_interface.py --server.headless true

:: 脚本结束
echo.
echo [INFO] 系统已停止
pause 