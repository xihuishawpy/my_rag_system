#!/bin/bash

# RAG系统启动脚本
# 自动检查环境并启动Web界面

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查Python版本
check_python() {
    print_info "检查Python环境..."
    
    if ! command_exists python3; then
        print_error "Python3未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python版本: $python_version"
    
    # 检查版本是否满足要求 (3.8+)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_error "Python版本过低，需要3.8+版本"
        exit 1
    fi
}

# 检查并安装依赖
check_dependencies() {
    print_info "检查Python依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt文件不存在"
        exit 1
    fi
    
    # 检查是否在虚拟环境中
    if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "建议在虚拟环境中运行"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "退出。建议先创建虚拟环境:"
            echo "  python3 -m venv venv"
            echo "  source venv/bin/activate"
            exit 1
        fi
    fi
    
    # 检查关键依赖包
    missing_packages=()
    
    if ! python3 -c "import streamlit" 2>/dev/null; then
        missing_packages+=("streamlit")
    fi
    
    if ! python3 -c "import chromadb" 2>/dev/null; then
        missing_packages+=("chromadb")
    fi
    
    if ! python3 -c "import faiss" 2>/dev/null; then
        missing_packages+=("faiss-cpu")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "缺少依赖包: ${missing_packages[*]}"
        read -p "是否自动安装? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_info "正在安装依赖包..."
            pip install -r requirements.txt
            print_success "依赖包安装完成"
        else
            print_error "请手动安装依赖包: pip install -r requirements.txt"
            exit 1
        fi
    else
        print_success "所有依赖包已安装"
    fi
}

# 检查API密钥
check_api_key() {
    print_info "检查API密钥配置..."
    
    if [ -z "$DASHSCOPE_API_KEY" ]; then
        print_warning "DASHSCOPE_API_KEY环境变量未设置"
        
        # 检查.env文件
        if [ -f ".env" ]; then
            print_info "发现.env文件，正在加载..."
            source .env
            if [ -n "$DASHSCOPE_API_KEY" ]; then
                print_success "从.env文件加载API密钥"
                export DASHSCOPE_API_KEY
            fi
        fi
        
        if [ -z "$DASHSCOPE_API_KEY" ]; then
            print_error "请设置DASHSCOPE_API_KEY环境变量"
            echo "方法1: export DASHSCOPE_API_KEY=\"your_api_key\""
            echo "方法2: 创建.env文件并添加 DASHSCOPE_API_KEY=your_api_key"
            echo "获取密钥: https://bailian.console.aliyun.com/"
            exit 1
        fi
    else
        print_success "API密钥已配置"
    fi
}

# 检查数据目录
check_data() {
    print_info "检查数据目录..."
    
    # 创建必要的目录
    mkdir -p vector_store
    mkdir -p cache
    mkdir -p qa_cache
    mkdir -p embeddings_cache
    mkdir -p logs
    
    # 检查是否有数据
    has_data=false
    
    if [ -d "vector_store/chroma" ] && [ "$(ls -A vector_store/chroma 2>/dev/null)" ]; then
        print_success "发现ChromaDB数据"
        has_data=true
    fi
    
    if [ -f "vector_store/faiss_index.bin" ]; then
        print_success "发现FAISS索引数据"
        has_data=true
    fi
    
    if [ -f "processed_documents.json" ]; then
        print_success "发现处理后的文档数据"
        has_data=true
    fi
    
    if [ "$has_data" = false ]; then
        print_warning "未发现现有数据"
        print_info "首次使用请先运行数据处理或快速设置:"
        echo "  python src/run_data_processing.py  # 处理Excel数据"
        echo "  python src/quick_setup.py         # 快速设置测试数据"
        
        read -p "是否运行快速设置? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_info "正在运行快速设置..."
            python3 src/quick_setup.py
            print_success "快速设置完成"
        fi
    fi
}

# 检查端口占用
check_port() {
    local port=${1:-8501}
    print_info "检查端口 $port..."
    
    if command_exists netstat; then
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            print_warning "端口 $port 已被占用"
            return 1
        fi
    elif command_exists lsof; then
        if lsof -ti:$port >/dev/null 2>&1; then
            print_warning "端口 $port 已被占用"
            return 1
        fi
    fi
    
    return 0
}

# 启动Web界面
start_web_interface() {
    print_info "启动RAG系统Web界面..."
    
    # 检查Streamlit是否可用
    if ! command_exists streamlit; then
        print_error "Streamlit未安装"
        exit 1
    fi
    
    # 检查源文件
    if [ ! -f "src/web_interface.py" ]; then
        print_error "Web界面文件不存在: src/web_interface.py"
        exit 1
    fi
    
    # 选择可用端口
    port=8501
    while ! check_port $port; do
        ((port++))
        if [ $port -gt 8510 ]; then
            print_error "无法找到可用端口"
            exit 1
        fi
    done
    
    print_success "使用端口: $port"
    print_info "正在启动Web界面..."
    print_info "浏览器将自动打开 http://localhost:$port"
    echo
    print_info "使用Ctrl+C停止服务"
    echo
    
    # 启动Streamlit
    streamlit run src/web_interface.py --server.port $port --server.headless true
}

# 显示帮助信息
show_help() {
    echo "RAG系统启动脚本"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --check    仅检查环境，不启动服务"
    echo "  -p, --port     指定端口号 (默认: 8501)"
    echo "  -q, --quick    跳过某些检查，快速启动"
    echo
    echo "示例:"
    echo "  $0              # 完整检查并启动"
    echo "  $0 --check     # 仅检查环境"
    echo "  $0 --port 8502 # 使用指定端口"
    echo "  $0 --quick     # 快速启动"
}

# 主函数
main() {
    local check_only=false
    local quick_mode=false
    local custom_port=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -p|--port)
                custom_port="$2"
                shift 2
                ;;
            -q|--quick)
                quick_mode=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo "=========================================="
    echo "      RAG知识库问答系统启动"
    echo "=========================================="
    echo
    
    # 执行检查
    check_python
    check_dependencies
    check_api_key
    
    if [ "$quick_mode" = false ]; then
        check_data
    fi
    
    if [ "$check_only" = true ]; then
        print_success "环境检查完成！"
        exit 0
    fi
    
    # 启动服务
    echo
    echo "=========================================="
    start_web_interface
}

# 捕获Ctrl+C信号
trap 'echo; print_info "正在停止服务..."; exit 0' INT

# 运行主函数
main "$@" 