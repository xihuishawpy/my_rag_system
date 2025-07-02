"""
向量存储模块
负责向量数据的存储、检索和管理，支持多种向量数据库后端
"""
import numpy as np
import logging
import json
import pickle
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from data_processing import Document
from config import VECTOR_STORE_TYPE, VECTOR_STORE_PATH

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果对象"""
    document: Document
    score: float
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class VectorStoreStats:
    """向量存储统计信息"""
    total_documents: int
    embedding_dimension: int
    storage_type: str
    storage_size_mb: float
    index_type: str
    last_updated: float
    metadata: Dict[str, Any] = None

class VectorStoreBase(ABC):
    """向量存储基类"""
    
    @abstractmethod
    def create_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """创建索引"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int, **kwargs) -> List[SearchResult]:
        """搜索相似文档"""
        pass
    
    @abstractmethod
    def update_document(self, doc_id: str, document: Document, embedding: np.ndarray) -> bool:
        """更新文档"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> VectorStoreStats:
        """获取统计信息"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """保存索引"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """加载索引"""
        pass

class ChromaVectorStore(VectorStoreBase):
    """ChromaDB向量存储实现"""
    
    def __init__(self, store_path: str, collection_name: str = "documents"):
        """
        初始化ChromaDB存储
        
        Args:
            store_path: 存储路径
            collection_name: 集合名称
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装chromadb: pip install chromadb")
        
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=str(self.store_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"已连接到现有集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            logger.info(f"已创建新集合: {collection_name}")
        
        self._embedding_dimension = None
        self._total_documents = 0
    
    def create_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """创建索引"""
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与嵌入数量({len(embeddings)})不匹配")
        
        logger.info(f"开始创建索引，共 {len(documents)} 个文档")
        
        # 准备数据
        ids = [doc.doc_id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        documents_content = [doc.content for doc in documents]
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # 清空现有数据
        try:
            self.collection.delete()
            logger.info("已清空现有集合数据")
        except:
            pass
        
        # 批量添加文档
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings_list[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=documents_content[i:end_idx]
            )
            
            logger.info(f"已添加文档 {i+1}-{end_idx}/{len(documents)}")
        
        self._embedding_dimension = len(embeddings[0]) if embeddings else 0
        self._total_documents = len(documents)
        
        logger.info(f"索引创建完成，维度: {self._embedding_dimension}")
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """添加文档"""
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与嵌入数量({len(embeddings)})不匹配")
        
        ids = [doc.doc_id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        documents_content = [doc.content for doc in documents]
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_content
        )
        
        self._total_documents += len(documents)
        logger.info(f"已添加 {len(documents)} 个文档")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, **kwargs) -> List[SearchResult]:
        """搜索相似文档"""
        # 提取过滤条件
        where_filters = kwargs.get('where', None)
        where_document_filters = kwargs.get('where_document', None)
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._total_documents) if self._total_documents > 0 else top_k,
            where=where_filters,
            where_document=where_document_filters,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )
        
        # 转换结果格式
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                document = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    doc_id=doc_id
                )
                
                # ChromaDB返回的是距离，需要转换为相似度分数
                distance = results['distances'][0][i]
                score = 1.0 / (1.0 + distance)  # 转换为相似度分数
                
                embedding = np.array(results['embeddings'][0][i]) if results['embeddings'] else None
                
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    embedding=embedding
                ))
        
        logger.debug(f"搜索完成，返回 {len(search_results)} 个结果")
        return search_results
    
    def update_document(self, doc_id: str, document: Document, embedding: np.ndarray) -> bool:
        """更新文档"""
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[document.metadata],
                documents=[document.content]
            )
            logger.info(f"已更新文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            self.collection.delete(ids=[doc_id])
            self._total_documents = max(0, self._total_documents - 1)
            logger.info(f"已删除文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_statistics(self) -> VectorStoreStats:
        """获取统计信息"""
        # 获取集合信息
        count_result = self.collection.count()
        
        # 计算存储大小
        storage_size = 0
        if self.store_path.exists():
            for file_path in self.store_path.rglob("*"):
                if file_path.is_file():
                    storage_size += file_path.stat().st_size
        
        return VectorStoreStats(
            total_documents=count_result,
            embedding_dimension=self._embedding_dimension or 0,
            storage_type="ChromaDB",
            storage_size_mb=storage_size / (1024 * 1024),
            index_type="HNSW",
            last_updated=time.time(),
            metadata={
                "collection_name": self.collection_name,
                "store_path": str(self.store_path)
            }
        )
    
    def save_index(self, path: str) -> None:
        """保存索引（ChromaDB自动持久化）"""
        # ChromaDB使用PersistentClient自动保存
        logger.info(f"ChromaDB索引已自动保存到: {self.store_path}")
    
    def load_index(self, path: str) -> None:
        """加载索引（ChromaDB自动加载）"""
        # ChromaDB使用PersistentClient自动加载
        stats = self.get_statistics()
        self._total_documents = stats.total_documents
        logger.info(f"已加载ChromaDB索引，文档数: {self._total_documents}")

class FAISSVectorStore(VectorStoreBase):
    """FAISS向量存储实现"""
    
    def __init__(self, store_path: str, index_type: str = "IVFFlat"):
        """
        初始化FAISS存储
        
        Args:
            store_path: 存储路径
            index_type: 索引类型 ("IVFFlat", "IndexFlatIP", "IndexFlatL2")
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装faiss: pip install faiss-cpu 或 pip install faiss-gpu")
        
        self.faiss = faiss
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type
        
        self.index = None
        self.documents = {}  # doc_id -> Document
        self.id_to_index = {}  # doc_id -> faiss_index
        self.index_to_id = {}  # faiss_index -> doc_id
        self._embedding_dimension = None
        self._next_index = 0
        
        # 文件路径
        self.index_file = self.store_path / "faiss.index"
        self.metadata_file = self.store_path / "metadata.json"
        
        logger.info(f"FAISS向量存储初始化完成，索引类型: {index_type}")
    
    def _create_faiss_index(self, dimension: int) -> None:
        """创建FAISS索引"""
        if self.index_type == "IndexFlatIP":
            self.index = self.faiss.IndexFlatIP(dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = self.faiss.IndexFlatL2(dimension)
        elif self.index_type == "IVFFlat":
            # 创建IVF索引（适合大规模数据）
            nlist = max(100, int(np.sqrt(1000)))  # 聚类中心数量
            quantizer = self.faiss.IndexFlatIP(dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # 默认使用平坦索引
            self.index = self.faiss.IndexFlatIP(dimension)
        
        logger.info(f"已创建FAISS索引，类型: {self.index_type}, 维度: {dimension}")
    
    def create_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """创建索引"""
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与嵌入数量({len(embeddings)})不匹配")
        
        if not embeddings:
            raise ValueError("嵌入向量列表不能为空")
        
        # 获取嵌入维度
        self._embedding_dimension = embeddings[0].shape[0]
        
        # 创建FAISS索引
        self._create_faiss_index(self._embedding_dimension)
        
        # 准备嵌入矩阵
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        
        # 对于IVF索引，需要训练
        if hasattr(self.index, 'train'):
            logger.info("正在训练IVF索引...")
            self.index.train(embedding_matrix)
        
        # 添加向量到索引
        self.index.add(embedding_matrix)
        
        # 保存文档映射
        self.documents.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        for i, doc in enumerate(documents):
            self.documents[doc.doc_id] = doc
            self.id_to_index[doc.doc_id] = i
            self.index_to_id[i] = doc.doc_id
        
        self._next_index = len(documents)
        
        logger.info(f"FAISS索引创建完成，文档数: {len(documents)}")
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """添加文档"""
        if len(documents) != len(embeddings):
            raise ValueError(f"文档数量({len(documents)})与嵌入数量({len(embeddings)})不匹配")
        
        if self.index is None:
            self.create_index(documents, embeddings)
            return
        
        # 准备嵌入矩阵
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        
        # 添加到索引
        self.index.add(embedding_matrix)
        
        # 更新文档映射
        for i, doc in enumerate(documents):
            faiss_idx = self._next_index + i
            self.documents[doc.doc_id] = doc
            self.id_to_index[doc.doc_id] = faiss_idx
            self.index_to_id[faiss_idx] = doc.doc_id
        
        self._next_index += len(documents)
        
        logger.info(f"已添加 {len(documents)} 个文档到FAISS索引")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, **kwargs) -> List[SearchResult]:
        """搜索相似文档"""
        if self.index is None:
            return []
        
        # 准备查询向量
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # 执行搜索
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # 转换结果
        search_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示无效结果
                continue
            
            doc_id = self.index_to_id.get(idx)
            if doc_id and doc_id in self.documents:
                document = self.documents[doc_id]
                search_results.append(SearchResult(
                    document=document,
                    score=float(score),
                    embedding=None  # FAISS不返回原始嵌入
                ))
        
        logger.debug(f"FAISS搜索完成，返回 {len(search_results)} 个结果")
        return search_results
    
    def update_document(self, doc_id: str, document: Document, embedding: np.ndarray) -> bool:
        """更新文档（FAISS不直接支持更新，需要重建索引）"""
        if doc_id in self.documents:
            self.documents[doc_id] = document
            logger.warning("FAISS索引已更新文档内容，但嵌入向量需要重建索引才能更新")
            return True
        return False
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档（FAISS不直接支持删除，需要重建索引）"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.id_to_index:
                idx = self.id_to_index[doc_id]
                del self.id_to_index[doc_id]
                if idx in self.index_to_id:
                    del self.index_to_id[idx]
            
            logger.warning(f"已删除文档 {doc_id} 的元数据，但需要重建索引以完全删除")
            return True
        return False
    
    def get_statistics(self) -> VectorStoreStats:
        """获取统计信息"""
        # 计算存储大小
        storage_size = 0
        if self.store_path.exists():
            for file_path in self.store_path.rglob("*"):
                if file_path.is_file():
                    storage_size += file_path.stat().st_size
        
        return VectorStoreStats(
            total_documents=len(self.documents),
            embedding_dimension=self._embedding_dimension or 0,
            storage_type="FAISS",
            storage_size_mb=storage_size / (1024 * 1024),
            index_type=self.index_type,
            last_updated=time.time(),
            metadata={
                "index_total": self.index.ntotal if self.index else 0,
                "store_path": str(self.store_path)
            }
        )
    
    def save_index(self, path: str = None) -> None:
        """保存索引"""
        save_path = Path(path) if path else self.store_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        if self.index is not None:
            index_file = save_path / "faiss.index"
            self.faiss.write_index(self.index, str(index_file))
        
        # 保存元数据
        metadata = {
            "documents": {doc_id: asdict(doc) for doc_id, doc in self.documents.items()},
            "id_to_index": self.id_to_index,
            "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
            "embedding_dimension": self._embedding_dimension,
            "next_index": self._next_index,
            "index_type": self.index_type
        }
        
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"FAISS索引已保存到: {save_path}")
    
    def load_index(self, path: str = None) -> None:
        """加载索引"""
        load_path = Path(path) if path else self.store_path
        
        # 加载FAISS索引
        index_file = load_path / "faiss.index"
        if index_file.exists():
            self.index = self.faiss.read_index(str(index_file))
        else:
            logger.warning(f"未找到FAISS索引文件: {index_file}")
            return
        
        # 加载元数据
        metadata_file = load_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 重建文档对象
            self.documents = {}
            for doc_id, doc_data in metadata["documents"].items():
                self.documents[doc_id] = Document(
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    doc_id=doc_data["doc_id"]
                )
            
            self.id_to_index = metadata["id_to_index"]
            self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
            self._embedding_dimension = metadata.get("embedding_dimension")
            self._next_index = metadata.get("next_index", 0)
            self.index_type = metadata.get("index_type", "IndexFlatIP")
        
        logger.info(f"已加载FAISS索引，文档数: {len(self.documents)}")

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, 
                 store_type: str = VECTOR_STORE_TYPE,
                 store_path: str = str(VECTOR_STORE_PATH),
                 **kwargs):
        """
        初始化向量存储管理器
        
        Args:
            store_type: 存储类型 ("chroma", "faiss")
            store_path: 存储路径
            **kwargs: 其他参数
        """
        self.store_type = store_type.lower()
        self.store_path = store_path
        
        # 创建具体的存储实现
        if self.store_type == "chroma":
            self.store = ChromaVectorStore(
                store_path=store_path,
                collection_name=kwargs.get("collection_name", "documents")
            )
        elif self.store_type == "faiss":
            self.store = FAISSVectorStore(
                store_path=store_path,
                index_type=kwargs.get("index_type", "IndexFlatIP")
            )
            # FAISS需要主动加载索引（如果存在）
            try:
                self.store.load_index()
                logger.info(f"已自动加载FAISS索引，文档数: {len(self.store.documents)}")
            except Exception as e:
                logger.debug(f"FAISS索引文件不存在或加载失败，将在首次使用时创建: {e}")
        else:
            raise ValueError(f"不支持的存储类型: {store_type}")
        
        logger.info(f"向量存储管理器初始化完成，类型: {store_type}")
    
    def create_index(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """创建索引"""
        return self.store.create_index(documents, embeddings)
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]) -> None:
        """添加文档"""
        return self.store.add_documents(documents, embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, **kwargs) -> List[SearchResult]:
        """搜索相似文档"""
        return self.store.search(query_embedding, top_k, **kwargs)
    
    def update_document(self, doc_id: str, document: Document, embedding: np.ndarray) -> bool:
        """更新文档"""
        return self.store.update_document(doc_id, document, embedding)
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        return self.store.delete_document(doc_id)
    
    def get_statistics(self) -> VectorStoreStats:
        """获取统计信息"""
        return self.store.get_statistics()
    
    def save_index(self, path: str = None) -> None:
        """保存索引"""
        return self.store.save_index(path)
    
    def load_index(self, path: str = None) -> None:
        """加载索引"""
        return self.store.load_index(path)
    
    def search_by_text(self, 
                      query_text: str, 
                      embedding_engine,
                      top_k: int = 10, 
                      **kwargs) -> List[SearchResult]:
        """
        通过文本搜索（需要嵌入引擎）
        
        Args:
            query_text: 查询文本
            embedding_engine: 嵌入引擎
            top_k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            List[SearchResult]: 搜索结果
        """
        query_embedding = embedding_engine.embed_query(query_text)
        return self.search(query_embedding, top_k, **kwargs)

def main():
    """主函数，用于测试向量存储模块"""
    import sys
    import os
    from pathlib import Path
    
    # 添加src目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("=== 向量存储模块测试 ===")
        
        # 1. 创建测试数据
        print("1. 创建测试数据...")
        test_documents = [
            Document("材料分析测试内容1", {"type": "analysis", "id": 1}, "doc_001"),
            Document("性能测试内容2", {"type": "performance", "id": 2}, "doc_002"),
            Document("质量检测内容3", {"type": "quality", "id": 3}, "doc_003"),
        ]
        
        # 创建模拟嵌入向量
        test_embeddings = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
            np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32),
        ]
        
        # 2. 测试ChromaDB存储（在可用时）
        print("\n2. 测试ChromaDB存储...")
        temp_chroma_path = "test_chroma_store"
        
        try:
            chroma_manager = VectorStoreManager(
                store_type="chroma",  # 保持用于测试比较
                store_path=temp_chroma_path
            )
            
            # 创建索引
            chroma_manager.create_index(test_documents, test_embeddings)
            print("   ChromaDB索引创建成功")
            
            # 搜索测试
            query_embedding = np.array([0.15, 0.25, 0.35, 0.45, 0.55], dtype=np.float32)
            results = chroma_manager.search(query_embedding, top_k=2)
            print(f"   搜索结果数: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"   结果{i+1}: {result.document.content[:30]}... (分数: {result.score:.4f})")
            
            # 统计信息
            stats = chroma_manager.get_statistics()
            print(f"   统计信息: {stats.total_documents} 个文档, {stats.storage_type}")
            
        except ImportError:
            print("   ChromaDB未安装，跳过测试")
        except Exception as e:
            print(f"   ChromaDB测试失败: {e}")
        
        # 3. 测试FAISS存储（使用当前配置）
        print("\n3. 测试FAISS存储...")
        temp_faiss_path = "test_faiss_store"
        
        try:
            faiss_manager = VectorStoreManager(
                store_type=VECTOR_STORE_TYPE,  # 使用配置
                store_path=temp_faiss_path,
                index_type="IndexFlatIP"
            )
            
            # 创建索引
            faiss_manager.create_index(test_documents, test_embeddings)
            print("   FAISS索引创建成功")
            
            # 搜索测试
            results = faiss_manager.search(query_embedding, top_k=2)
            print(f"   搜索结果数: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"   结果{i+1}: {result.document.content[:30]}... (分数: {result.score:.4f})")
            
            # 保存和加载测试
            faiss_manager.save_index()
            print("   FAISS索引保存成功")
            
            # 创建新实例并加载
            new_faiss_manager = VectorStoreManager(
                store_type="faiss",
                store_path=temp_faiss_path
            )
            new_faiss_manager.load_index()
            print("   FAISS索引加载成功")
            
            # 统计信息
            stats = faiss_manager.get_statistics()
            print(f"   统计信息: {stats.total_documents} 个文档, {stats.storage_type}")
            
        except ImportError:
            print("   FAISS未安装，跳过测试")
        except Exception as e:
            print(f"   FAISS测试失败: {e}")
        
        # 4. 测试文档操作
        print("\n4. 测试文档操作...")
        try:
            # 使用FAISS进行测试
            manager = VectorStoreManager(store_type="faiss", store_path="test_ops")
            manager.create_index(test_documents, test_embeddings)
            
            # 添加新文档
            new_doc = Document("新添加的文档内容", {"type": "new"}, "doc_004")
            new_embedding = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float32)
            manager.add_documents([new_doc], [new_embedding])
            print("   文档添加成功")
            
            # 更新文档
            updated_doc = Document("更新后的文档内容", {"type": "updated"}, "doc_001")
            updated_embedding = np.array([2.1, 2.2, 2.3, 2.4, 2.5], dtype=np.float32)
            success = manager.update_document("doc_001", updated_doc, updated_embedding)
            print(f"   文档更新: {'成功' if success else '失败'}")
            
            # 删除文档
            success = manager.delete_document("doc_003")
            print(f"   文档删除: {'成功' if success else '失败'}")
            
        except Exception as e:
            print(f"   文档操作测试失败: {e}")
        
        print("\n✅ 向量存储模块测试完成！")
        
        # 清理测试文件
        import shutil
        for test_dir in ["test_chroma_store", "test_faiss_store", "test_ops"]:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
                print(f"   已清理测试目录: {test_dir}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 