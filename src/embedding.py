"""
向量化模块
负责将文档和查询转换为向量表示，集成text-embedding-v3模型
"""
import numpy as np
import logging
import json
import pickle
import hashlib
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()

from data_processing import Document
from config import EMBEDDING_MODEL, PROJECT_ROOT

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """嵌入结果对象"""
    embedding: np.ndarray
    text: str
    model_name: str
    timestamp: float
    metadata: Dict[str, Any] = None

class EmbeddingCache:
    """嵌入向量缓存管理器"""
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, str]:
        """加载缓存索引"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """获取缓存的嵌入向量"""
        cache_key = self._get_cache_key(text, model_name)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding_result = pickle.load(f)
                    return embedding_result.embedding
                except Exception as e:
                    logger.warning(f"读取缓存失败: {e}")
                    # 清理损坏的缓存
                    self._remove_cache(cache_key)
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """设置缓存的嵌入向量"""
        cache_key = self._get_cache_key(text, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            embedding_result = EmbeddingResult(
                embedding=embedding,
                text=text,
                model_name=model_name,
                timestamp=time.time()
            )
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding_result, f)
            
            self.cache_index[cache_key] = {
                "file": f"{cache_key}.pkl",
                "timestamp": embedding_result.timestamp,
                "text_length": len(text)
            }
            self._save_cache_index()
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _remove_cache(self, cache_key: str):
        """移除损坏的缓存"""
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            del self.cache_index[cache_key]
            self._save_cache_index()
    
    def clear_cache(self):
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.cache_index.clear()
        self._save_cache_index()
        logger.info("已清空所有缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_files = len(list(self.cache_dir.glob("*.pkl")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
        
        return {
            "total_cached_items": len(self.cache_index),
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_dir)
        }

class EmbeddingEngine:
    """嵌入向量引擎"""
    
    def __init__(self, 
                 model_name: str = EMBEDDING_MODEL,
                 api_key: Optional[str] = None,
                 cache_enabled: bool = True,
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        初始化嵌入引擎
        
        Args:
            model_name: 嵌入模型名称
            api_key: API密钥
            cache_enabled: 是否启用缓存
            batch_size: 批处理大小
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 初始化OpenAI客户端
        try:
            # 获取API密钥
            if not api_key:
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if not api_key:
                    raise ValueError("未找到API密钥，请设置DASHSCOPE_API_KEY环境变量")
            
            # 初始化Qwen API客户端
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        except Exception as e:
            logger.error(f"初始化Qwen API客户端失败: {e}")
            raise
        
        # 初始化缓存
        self.cache_enabled = cache_enabled
        if cache_enabled:
            cache_dir = PROJECT_ROOT / "embeddings_cache"
            self.cache = EmbeddingCache(str(cache_dir))
        else:
            self.cache = None
        
        logger.info(f"向量化引擎初始化完成，模型: {model_name}")
    
    def _embed_single_text(self, text: str) -> np.ndarray:
        """
        对单个文本进行嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 嵌入向量
        """
        # 检查缓存
        if self.cache_enabled and self.cache:
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                logger.debug(f"从缓存获取嵌入向量，文本长度: {len(text)}")
                return cached_embedding
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                # 保存到缓存
                if self.cache_enabled and self.cache:
                    self.cache.set(text, self.model_name, embedding)
                
                logger.debug(f"成功生成嵌入向量，维度: {embedding.shape}")
                return embedding
                
            except Exception as e:
                logger.warning(f"嵌入生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    raise RuntimeError(f"嵌入生成失败，已重试 {self.max_retries} 次: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        对单个文本进行嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        return self._embed_single_text(text.strip())
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        对查询进行嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 查询嵌入向量
        """
        return self.embed_text(query)
    
    def batch_embed(self, texts: List[str], batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，如果为None则使用默认值
            
        Returns:
            List[np.ndarray]: 嵌入向量列表
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        logger.info(f"开始批量嵌入，共 {len(texts)} 个文本，批大小: {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                try:
                    embedding = self.embed_text(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"文本嵌入失败: {text[:100]}... 错误: {e}")
                    # 使用零向量作为fallback
                    if batch_embeddings:
                        zero_embedding = np.zeros_like(batch_embeddings[0])
                    else:
                        # 如果这是第一个文本且失败了，创建一个默认维度的零向量
                        zero_embedding = np.zeros(1536, dtype=np.float32)  # text-embedding-v3的默认维度
                    batch_embeddings.append(zero_embedding)
            
            embeddings.extend(batch_embeddings)
            
            # 添加延迟避免API限制
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        logger.info(f"批量嵌入完成，生成 {len(embeddings)} 个向量")
        return embeddings
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        对文档列表进行嵌入
        
        Args:
            documents: 文档列表
            
        Returns:
            List[np.ndarray]: 文档嵌入向量列表
        """
        if not documents:
            return []
        
        texts = [doc.content for doc in documents]
        logger.info(f"开始嵌入 {len(documents)} 个文档")
        
        embeddings = self.batch_embed(texts)
        
        # 验证结果数量
        if len(embeddings) != len(documents):
            logger.warning(f"嵌入数量不匹配: 文档数{len(documents)}, 嵌入数{len(embeddings)}")
        
        return embeddings
    
    def save_embeddings(self, embeddings: List[np.ndarray], documents: List[Document], output_path: str):
        """
        保存嵌入向量到文件
        
        Args:
            embeddings: 嵌入向量列表
            documents: 对应的文档列表
            output_path: 输出文件路径
        """
        if len(embeddings) != len(documents):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与文档数量({len(documents)})不匹配")
        
        save_data = {
            "model_name": self.model_name,
            "timestamp": time.time(),
            "total_documents": len(documents),
            "embedding_dimension": embeddings[0].shape[0] if embeddings else 0,
            "embeddings": [],
            "documents": []
        }
        
        for embedding, doc in zip(embeddings, documents):
            save_data["embeddings"].append({
                "doc_id": doc.doc_id,
                "embedding": embedding.tolist(),
                "shape": embedding.shape
            })
            save_data["documents"].append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(embeddings)} 个嵌入向量到: {output_path}")
    
    def load_embeddings(self, input_path: str) -> tuple[List[np.ndarray], List[Document]]:
        """
        从文件加载嵌入向量
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            tuple: (嵌入向量列表, 文档列表)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embeddings = []
        documents = []
        
        for emb_data, doc_data in zip(data["embeddings"], data["documents"]):
            # 重建嵌入向量
            embedding = np.array(emb_data["embedding"], dtype=np.float32)
            embeddings.append(embedding)
            
            # 重建文档对象
            document = Document(
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                doc_id=doc_data["doc_id"]
            )
            documents.append(document)
        
        logger.info(f"已加载 {len(embeddings)} 个嵌入向量，模型: {data['model_name']}")
        return embeddings, documents
    
    def get_embedding_stats(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        获取嵌入向量统计信息
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            Dict: 统计信息
        """
        if not embeddings:
            return {"total_embeddings": 0}
        
        dimensions = [emb.shape[0] for emb in embeddings]
        
        stats = {
            "total_embeddings": len(embeddings),
            "embedding_dimension": embeddings[0].shape[0],
            "dimensions_consistent": len(set(dimensions)) == 1,
            "memory_usage_mb": sum(emb.nbytes for emb in embeddings) / (1024 * 1024),
            "model_name": self.model_name
        }
        
        if self.cache_enabled and self.cache:
            cache_stats = self.cache.get_cache_stats()
            stats["cache_stats"] = cache_stats
        
        return stats

def main():
    """主函数，用于测试向量化模块"""
    import sys
    import os
    from config import EXCEL_FILE_PATH
    
    # 添加src目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_processing import DataProcessor
    
    try:
        # 1. 加载处理好的文档
        print("=== 向量化模块测试 ===")
        print("1. 加载文档数据...")
        
        processor = DataProcessor(EXCEL_FILE_PATH)
        documents = processor.process_excel_to_documents()
        print(f"   加载了 {len(documents)} 个文档")
        
        # 2. 初始化嵌入引擎
        print("2. 初始化嵌入引擎...")
        embedding_engine = EmbeddingEngine(cache_enabled=True)
        
        # 3. 测试单个文本嵌入
        print("3. 测试单个文本嵌入...")
        test_text = documents[0].content
        embedding = embedding_engine.embed_text(test_text)
        print(f"   文本: {test_text[:100]}...")
        print(f"   嵌入向量维度: {embedding.shape}")
        print(f"   向量范数: {np.linalg.norm(embedding):.4f}")
        
        # 4. 测试批量嵌入（只处理前5个文档以节省时间）
        print("4. 测试批量文档嵌入...")
        test_documents = documents[:5]
        embeddings = embedding_engine.embed_documents(test_documents)
        
        # 5. 显示统计信息
        print("5. 嵌入统计信息:")
        stats = embedding_engine.get_embedding_stats(embeddings)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # 6. 保存嵌入结果
        print("6. 保存嵌入结果...")
        output_path = "test_embeddings.json"
        embedding_engine.save_embeddings(embeddings, test_documents, output_path)
        
        # 7. 测试加载功能
        print("7. 测试加载功能...")
        loaded_embeddings, loaded_documents = embedding_engine.load_embeddings(output_path)
        print(f"   加载了 {len(loaded_embeddings)} 个嵌入向量")
        
        print("\n✅ 向量化模块测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 