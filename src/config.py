import os
from pathlib import Path

######################## 项目配置 ########################
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 模型配置
EMBEDDING_MODEL = "text-embedding-v3"
RERANK_MODEL = "gte-rerank-v2" 
QA_MODEL = "qwen-max"

# 数据处理配置
EXCEL_FILE_PATH = DATA_DIR / "中心材料实验室-能力拆解_清理后.xlsx"
CHUNK_SIZE = 512
OVERLAP_SIZE = 50

######################## 向量存储 ########################
VECTOR_STORE_TYPE = "chroma"  # chroma, faiss
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

# 添加FAISS专用配置
FAISS_INDEX_TYPE = "IndexFlatIP"  # 可选：IndexFlatIP, IndexFlatL2, IVFFlat

# 不同的FAISS索引类型，可根据需求选择：
# 1. 精确搜索（小数据量）
# index_type="IndexFlatIP"      # 内积（推荐用于归一化向量）
# index_type="IndexFlatL2"      # L2距离

# 2. 近似搜索（大数据量）  
# index_type="IVFFlat"          # 倒排索引，速度与精度平衡

######################## 检索配置 ########################
TOP_K = 10
RERANK_TOP_K = 3

# 检索配置
DEFAULT_SEARCH_METHOD = "hybrid"  # 默认
DEFAULT_TOP_K = 10
DEFAULT_HYBRID_ALPHA = 0.7

# 检索方式映射
SEARCH_METHODS = {
    "语义搜索": "semantic",
    "混合搜索": "hybrid", 
    "过滤搜索": "filtered",
    "多查询搜索": "multi",
    "智能搜索": "smart"
}

######################## 日志配置 ########################
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "rag_system.log"

######################## API配置 ########################
# Qwen API配置 - 通义千问API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# Qwen API相关配置
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 