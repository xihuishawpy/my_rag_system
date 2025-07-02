# 🚀 GitHub用户快速设置指南

## 📋 项目简介

这是一个基于RAG（检索增强生成）技术的智能知识库问答系统，支持多种向量数据库和智能检索策略。

**核心特性：**
- 🧠 基于通义千问大模型的智能问答
- 🔍 支持FAISS和ChromaDB两种向量数据库
- 🎯 使用gte-rerank-v2模型进行精确重排序
- 🎨 美观的Streamlit Web界面
- 🔧 跨平台一键启动脚本

## 🔧 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/xihuishawpy/my_rag_system.git
cd my_rag_system
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置API密钥
```bash
# 复制配置模板
cp env_template.txt .env

# 编辑.env文件，填入您的API密钥
# DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 4. 启动系统

#### Linux/Mac用户：
```bash
./start.sh
```

#### Windows用户：
```bash
start.bat
```

#### 手动启动：
```bash
streamlit run src/web_interface.py
```

### 5. 访问系统
浏览器打开：`http://localhost:8501`

## 🔑 API密钥获取

1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 注册/登录账号
3. 开通百炼服务
4. 创建API密钥
5. 将密钥填入`.env`文件

## 📁 项目结构

```
my_rag_system/
├── README.md                    # 详细说明文档
├── GITHUB_SETUP.md             # 快速设置指南
├── 使用注意事项.md               # 详细注意事项
├── requirements.txt             # Python依赖
├── start.sh                     # Linux/Mac启动脚本
├── start.bat                    # Windows启动脚本
├── env_template.txt             # 环境配置模板
└── src/                         # 源代码目录
    ├── web_interface.py         # Web界面
    ├── qa_engine.py             # 问答引擎
    ├── vector_store.py          # 向量存储
    ├── retrieval.py             # 检索模块
    ├── reranking.py             # 重排序模块
    ├── embedding.py             # 嵌入模块
    ├── config.py                # 系统配置
    └── quick_setup.py           # 快速设置
```

## ⚠️ 重要说明

### 被忽略的文件
为了保护隐私和减少仓库大小，以下文件/目录被`.gitignore`忽略：
- `.env` - API密钥配置文件
- `data/` - 原始数据目录
- `vector_store/` - 向量数据库文件
- `*_cache/` - 各种缓存目录
- `processed_documents.json` - 处理后的文档
- `*.log` - 日志文件

### 首次运行
1. 系统会提示您初始化数据
2. 可以选择运行快速设置生成测试数据
3. 或者上传您自己的数据文件

## 🆘 常见问题

**Q: 提示API密钥未配置**
A: 确保`.env`文件存在且包含正确的`DASHSCOPE_API_KEY`

**Q: 启动脚本没有执行权限**
A: 运行 `chmod +x start.sh`

**Q: 依赖包安装失败**
A: 建议使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Q: 没有数据文件**
A: 首次运行时选择"快速设置"生成测试数据

## 📚 详细文档

- [完整使用说明](README.md)
- [注意事项和最佳实践](使用注意事项.md)
- [系统详细方案](RAG系统详细方案.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。 