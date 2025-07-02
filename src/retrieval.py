"""
检索模块
负责高级检索功能，包括语义搜索、混合搜索、过滤搜索等
"""
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
import json
from pathlib import Path

from data_processing import Document
from vector_store import VectorStoreManager, SearchResult
from embedding import EmbeddingEngine
from config import TOP_K, RERANK_TOP_K, VECTOR_STORE_PATH, VECTOR_STORE_TYPE

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果对象"""
    document: Document
    score: float
    retrieval_type: str  # "semantic", "hybrid", "filtered"
    rank: int
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    processed_query: str
    query_type: str  # "simple", "complex", "filtered"
    keywords: List[str]
    filters: Dict[str, Any]
    intent: str  # "search", "question", "comparison"
    confidence: float

@dataclass
class RetrievalStats:
    """检索统计信息"""
    query_count: int
    avg_response_time: float
    total_documents_searched: int
    cache_hit_rate: float
    top_queries: List[str]
    retrieval_accuracy: float
    last_updated: float

class QueryProcessor:
    """查询预处理器"""
    
    def __init__(self):
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'
        }
        
        # 过滤关键词映射
        self.filter_keywords = {
            '价格': ['price', '费用', '成本', '报价'],
            '周期': ['period', '时间', '工期', '天数'],
            '标准': ['standard', '规范', '要求'],
            '实验室': ['lab', 'laboratory', '检测机构'],
            'CMA': ['cma', '计量认证'],
            'CNAS': ['cnas', '认可'],
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        分析查询内容
        
        Args:
            query: 原始查询文本
            
        Returns:
            QueryAnalysis: 查询分析结果
        """
        # 基本清理
        processed_query = self._clean_query(query)
        
        # 提取关键词
        keywords = self._extract_keywords(processed_query)
        
        # 检测过滤条件
        filters = self._detect_filters(processed_query)
        
        # 判断查询类型
        query_type = self._classify_query_type(processed_query, filters)
        
        # 判断意图
        intent = self._detect_intent(processed_query)
        
        # 计算置信度
        confidence = self._calculate_confidence(processed_query, keywords, filters)
        
        return QueryAnalysis(
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            keywords=keywords,
            filters=filters,
            intent=intent,
            confidence=confidence
        )
    
    def _clean_query(self, query: str) -> str:
        """清理查询文本"""
        # 移除多余空格
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 移除特殊字符（保留中文、英文、数字、基本标点）
        query = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()（）【】""''、。，；：！？-]', '', query)
        
        return query
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单分词（基于空格和标点）
        words = re.findall(r'[\u4e00-\u9fff\w]+', query)
        
        # 过滤停用词
        keywords = [word for word in words if word.lower() not in self.stopwords and len(word) > 1]
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _detect_filters(self, query: str) -> Dict[str, Any]:
        """检测过滤条件"""
        filters = {}
        
        # 检测价格相关
        price_pattern = r'(\d+)\s*[元块钱]|价格.*?(\d+)|(\d+)\s*元以下'
        price_match = re.search(price_pattern, query)
        if price_match:
            price_value = next(g for g in price_match.groups() if g)
            filters['price_range'] = {'max': int(price_value)}
        
        # 检测周期相关
        period_pattern = r'(\d+)\s*[天日]|周期.*?(\d+)|(\d+)\s*天以内'
        period_match = re.search(period_pattern, query)
        if period_match:
            period_value = next(g for g in period_match.groups() if g)
            filters['period_range'] = {'max': int(period_value)}
        
        # 检测认证要求
        if any(keyword in query.upper() for keyword in ['CMA', '计量认证']):
            filters['CMA'] = True
        
        if any(keyword in query.upper() for keyword in ['CNAS', '认可']):
            filters['CNAS'] = True
        
        # 检测样品类别
        sample_categories = ['金属', '塑料', '橡胶', '涂料', '纺织', '食品', '环境', '建材']
        for category in sample_categories:
            if category in query:
                filters['sample_category'] = category
                break
        
        return filters
    
    def _classify_query_type(self, query: str, filters: Dict[str, Any]) -> str:
        """分类查询类型"""
        if filters:
            return "filtered"
        elif len(query.split()) > 5:
            return "complex"
        else:
            return "simple"
    
    def _detect_intent(self, query: str) -> str:
        """检测查询意图"""
        question_words = ['什么', '如何', '怎么', '为什么', '哪里', '哪个', '多少', 'what', 'how', 'why', 'where', 'which']
        comparison_words = ['比较', '对比', '区别', '差异', 'vs', 'versus', 'compare']
        
        if any(word in query for word in question_words) or '?' in query or '？' in query:
            return "question"
        elif any(word in query for word in comparison_words):
            return "comparison"
        else:
            return "search"
    
    def _calculate_confidence(self, query: str, keywords: List[str], filters: Dict[str, Any]) -> float:
        """计算查询置信度"""
        confidence = 0.5  # 基础置信度
        
        # 关键词质量
        if keywords:
            confidence += 0.2
            if len(keywords) >= 2:
                confidence += 0.1
        
        # 过滤条件
        if filters:
            confidence += 0.1 * len(filters)
        
        # 查询长度
        query_length = len(query.split())
        if 3 <= query_length <= 10:
            confidence += 0.1
        
        return min(confidence, 1.0)

class HybridRetriever:
    """混合检索器（向量+关键词）"""
    
    def __init__(self):
        self.keyword_weights = {
            'exact_match': 1.0,
            'partial_match': 0.7,
            'synonym_match': 0.5
        }
    
    def keyword_search(self, query_keywords: List[str], documents: List[Document], top_k: int = 50) -> List[Tuple[Document, float]]:
        """
        基于关键词的搜索
        
        Args:
            query_keywords: 查询关键词列表
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 关键词匹配分数)
        """
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_keyword_score(query_keywords, doc)
            if score > 0:
                scored_docs.append((doc, score))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def _calculate_keyword_score(self, query_keywords: List[str], document: Document) -> float:
        """计算关键词匹配分数"""
        content = document.content.lower()
        total_score = 0.0
        
        for keyword in query_keywords:
            keyword_lower = keyword.lower()
            
            # 精确匹配
            if keyword_lower in content:
                total_score += self.keyword_weights['exact_match']
            
            # 部分匹配
            elif any(keyword_lower in word for word in content.split()):
                total_score += self.keyword_weights['partial_match']
        
        # 归一化分数
        if query_keywords:
            total_score /= len(query_keywords)
        
        return total_score
    
    def combine_scores(self, semantic_results: List[SearchResult], keyword_results: List[Tuple[Document, float]], alpha: float = 0.7) -> List[RetrievalResult]:
        """
        合并语义搜索和关键词搜索结果
        
        Args:
            semantic_results: 语义搜索结果
            keyword_results: 关键词搜索结果
            alpha: 语义搜索权重（0-1）
            
        Returns:
            List[RetrievalResult]: 合并后的结果
        """
        # 创建文档ID到分数的映射
        semantic_scores = {result.document.doc_id: result.score for result in semantic_results}
        keyword_scores = {doc.doc_id: score for doc, score in keyword_results}
        
        # 收集所有文档
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            semantic_score = semantic_scores.get(doc_id, 0.0)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            
            # 归一化分数
            semantic_score = min(semantic_score, 1.0)
            keyword_score = min(keyword_score, 1.0)
            
            # 线性组合
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
            
            # 获取文档对象
            document = None
            for result in semantic_results:
                if result.document.doc_id == doc_id:
                    document = result.document
                    break
            
            if not document:
                for doc, _ in keyword_results:
                    if doc.doc_id == doc_id:
                        document = doc
                        break
            
            if document:
                combined_results.append(RetrievalResult(
                    document=document,
                    score=combined_score,
                    retrieval_type="hybrid",
                    rank=0,  # 稍后设置
                    metadata={
                        'semantic_score': semantic_score,
                        'keyword_score': keyword_score,
                        'alpha': alpha
                    }
                ))
        
        # 按分数排序并设置排名
        combined_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results

class RetrievalEngine:
    """高级检索引擎"""
    
    def __init__(self, vector_store: VectorStoreManager, embedding_engine: EmbeddingEngine):
        """
        初始化检索引擎
        
        Args:
            vector_store: 向量存储管理器
            embedding_engine: 嵌入引擎
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.query_processor = QueryProcessor()
        self.hybrid_retriever = HybridRetriever()
        
        # 统计信息
        self.stats = RetrievalStats(
            query_count=0,
            avg_response_time=0.0,
            total_documents_searched=0,
            cache_hit_rate=0.0,
            top_queries=[],
            retrieval_accuracy=0.0,
            last_updated=time.time()
        )
        
        # 查询缓存
        self.query_cache = {}
        self.cache_size_limit = 1000
        
        logger.info("检索引擎初始化完成")
    
    def semantic_search(self, query: str, top_k: int = TOP_K, **kwargs) -> List[RetrievalResult]:
        """
        语义相似度检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            List[RetrievalResult]: 检索结果
        """
        start_time = time.time()
        
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_engine.embed_query(query)
            
            # 执行向量搜索
            search_results = self.vector_store.search(query_embedding, top_k, **kwargs)
            
            # 转换为检索结果
            retrieval_results = []
            for i, result in enumerate(search_results):
                retrieval_results.append(RetrievalResult(
                    document=result.document,
                    score=result.score,
                    retrieval_type="semantic",
                    rank=i + 1,
                    metadata={'embedding_similarity': result.score}
                ))
            
            # 更新统计
            self._update_stats(start_time, len(search_results))
            
            logger.debug(f"语义搜索完成，查询: {query[:50]}..., 结果数: {len(retrieval_results)}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = TOP_K, alpha: float = 0.7, **kwargs) -> List[RetrievalResult]:
        """
        混合检索（语义+关键词）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            alpha: 语义搜索权重（0-1）
            **kwargs: 其他搜索参数
            
        Returns:
            List[RetrievalResult]: 检索结果
        """
        start_time = time.time()
        
        try:
            # 分析查询
            query_analysis = self.query_processor.analyze_query(query)
            
            # 语义搜索
            query_embedding = self.embedding_engine.embed_query(query)
            semantic_results = self.vector_store.search(query_embedding, top_k * 2, **kwargs)
            
            # 关键词搜索（需要获取更多文档进行关键词匹配）
            # 这里简化实现，实际应该从完整文档集合中搜索
            documents = [result.document for result in semantic_results]
            keyword_results = self.hybrid_retriever.keyword_search(query_analysis.keywords, documents, top_k * 2)
            
            # 合并结果
            hybrid_results = self.hybrid_retriever.combine_scores(semantic_results, keyword_results, alpha)
            
            # 限制结果数量
            final_results = hybrid_results[:top_k]
            
            # 更新统计
            self._update_stats(start_time, len(final_results))
            
            logger.debug(f"混合搜索完成，查询: {query[:50]}..., 结果数: {len(final_results)}")
            return final_results
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    def filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = TOP_K, **kwargs) -> List[RetrievalResult]:
        """
        过滤搜索
        
        Args:
            query: 查询文本
            filters: 过滤条件
            top_k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            List[RetrievalResult]: 检索结果
        """
        start_time = time.time()
        
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_engine.embed_query(query)
            
            # 转换过滤条件为向量存储格式
            vector_filters = self._convert_filters_to_vector_format(filters)
            
            # 合并过滤条件
            search_kwargs = {**kwargs, **vector_filters}
            
            # 执行过滤搜索
            search_results = self.vector_store.search(query_embedding, top_k, **search_kwargs)
            
            # 转换为检索结果
            retrieval_results = []
            for i, result in enumerate(search_results):
                retrieval_results.append(RetrievalResult(
                    document=result.document,
                    score=result.score,
                    retrieval_type="filtered",
                    rank=i + 1,
                    metadata={
                        'filters_applied': filters,
                        'embedding_similarity': result.score
                    }
                ))
            
            # 更新统计
            self._update_stats(start_time, len(retrieval_results))
            
            logger.debug(f"过滤搜索完成，查询: {query[:50]}..., 过滤条件: {filters}, 结果数: {len(retrieval_results)}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"过滤搜索失败: {e}")
            return []
    
    def multi_query_search(self, queries: List[str], top_k: int = TOP_K, fusion_method: str = "rrf") -> List[RetrievalResult]:
        """
        多查询融合搜索
        
        Args:
            queries: 多个查询文本
            top_k: 返回结果数量
            fusion_method: 融合方法 ("rrf", "weighted", "max")
            
        Returns:
            List[RetrievalResult]: 融合后的检索结果
        """
        start_time = time.time()
        
        try:
            all_results = []
            
            # 对每个查询进行搜索
            for query in queries:
                results = self.semantic_search(query, top_k * 2)
                all_results.append(results)
            
            # 融合结果
            if fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(all_results, top_k)
            elif fusion_method == "weighted":
                weights = [1.0 / len(queries)] * len(queries)  # 等权重
                fused_results = self._weighted_fusion(all_results, weights, top_k)
            elif fusion_method == "max":
                fused_results = self._max_score_fusion(all_results, top_k)
            else:
                raise ValueError(f"不支持的融合方法: {fusion_method}")
            
            # 更新检索类型
            for result in fused_results:
                result.retrieval_type = "multi_query"
                if result.metadata is None:
                    result.metadata = {}
                result.metadata['fusion_method'] = fusion_method
                result.metadata['query_count'] = len(queries)
            
            # 更新统计
            self._update_stats(start_time, len(fused_results))
            
            logger.debug(f"多查询搜索完成，查询数: {len(queries)}, 融合方法: {fusion_method}, 结果数: {len(fused_results)}")
            return fused_results
            
        except Exception as e:
            logger.error(f"多查询搜索失败: {e}")
            return []
    
    def smart_search(self, query: str, top_k: int = TOP_K, **kwargs) -> List[RetrievalResult]:
        """
        智能搜索（自动选择最佳检索策略）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            List[RetrievalResult]: 检索结果
        """
        start_time = time.time()
        
        try:
            # 分析查询
            query_analysis = self.query_processor.analyze_query(query)
            
            # 根据查询分析选择检索策略
            if query_analysis.filters:
                # 有过滤条件，使用过滤搜索
                results = self.filtered_search(query, query_analysis.filters, top_k, **kwargs)
            elif query_analysis.query_type == "complex" or len(query_analysis.keywords) > 3:
                # 复杂查询，使用混合搜索
                results = self.hybrid_search(query, top_k, alpha=0.6, **kwargs)
            else:
                # 简单查询，使用语义搜索
                results = self.semantic_search(query, top_k, **kwargs)
            
            # 添加查询分析信息到元数据
            for result in results:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata['query_analysis'] = asdict(query_analysis)
                result.metadata['auto_strategy'] = True
            
            logger.debug(f"智能搜索完成，查询类型: {query_analysis.query_type}, 策略: {results[0].retrieval_type if results else 'none'}")
            return results
            
        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            return []
    
    def _convert_filters_to_vector_format(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """将过滤条件转换为向量存储格式"""
        vector_filters = {}
        
        # ChromaDB格式的where子句
        where_conditions = {}
        
        # 处理价格范围
        if 'price_range' in filters:
            price_range = filters['price_range']
            if 'max' in price_range:
                # 这需要根据实际元数据结构调整
                where_conditions['raw_价格'] = {'$lte': str(price_range['max'])}
        
        # 处理周期范围
        if 'period_range' in filters:
            period_range = filters['period_range']
            if 'max' in period_range:
                where_conditions['raw_普通周期'] = {'$lte': str(period_range['max'])}
        
        # 处理认证要求
        if filters.get('CMA'):
            where_conditions['raw_CMA'] = {'$eq': 'Y'}
        
        if filters.get('CNAS'):
            where_conditions['raw_CNAS'] = {'$eq': 'Y'}
        
        # 处理样品类别
        if 'sample_category' in filters:
            where_conditions['raw_样品类别'] = {'$eq': filters['sample_category']}
        
        if where_conditions:
            vector_filters['where'] = where_conditions
        
        return vector_filters
    
    def _reciprocal_rank_fusion(self, all_results: List[List[RetrievalResult]], top_k: int, k: int = 60) -> List[RetrievalResult]:
        """倒数排名融合（RRF）"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for results in all_results:
            for rank, result in enumerate(results, 1):
                doc_id = result.document.doc_id
                doc_scores[doc_id] += 1.0 / (k + rank)
                doc_objects[doc_id] = result.document
        
        # 排序并创建结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            fused_results.append(RetrievalResult(
                document=doc_objects[doc_id],
                score=score,
                retrieval_type="multi_query",
                rank=i + 1,
                metadata={'rrf_score': score}
            ))
        
        return fused_results
    
    def _weighted_fusion(self, all_results: List[List[RetrievalResult]], weights: List[float], top_k: int) -> List[RetrievalResult]:
        """加权融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for results, weight in zip(all_results, weights):
            for result in results:
                doc_id = result.document.doc_id
                doc_scores[doc_id] += result.score * weight
                doc_objects[doc_id] = result.document
        
        # 排序并创建结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            fused_results.append(RetrievalResult(
                document=doc_objects[doc_id],
                score=score,
                retrieval_type="multi_query",
                rank=i + 1,
                metadata={'weighted_score': score}
            ))
        
        return fused_results
    
    def _max_score_fusion(self, all_results: List[List[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
        """最大分数融合"""
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for results in all_results:
            for result in results:
                doc_id = result.document.doc_id
                doc_scores[doc_id] = max(doc_scores[doc_id], result.score)
                doc_objects[doc_id] = result.document
        
        # 排序并创建结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            fused_results.append(RetrievalResult(
                document=doc_objects[doc_id],
                score=score,
                retrieval_type="multi_query",
                rank=i + 1,
                metadata={'max_score': score}
            ))
        
        return fused_results
    
    def _update_stats(self, start_time: float, result_count: int):
        """更新统计信息"""
        duration = time.time() - start_time
        
        self.stats.query_count += 1
        
        # 更新平均响应时间
        total_time = self.stats.avg_response_time * (self.stats.query_count - 1) + duration
        self.stats.avg_response_time = total_time / self.stats.query_count
        
        self.stats.total_documents_searched += result_count
        self.stats.last_updated = time.time()
    
    def get_statistics(self) -> RetrievalStats:
        """获取检索统计信息"""
        return self.stats
    
    def clear_cache(self):
        """清空查询缓存"""
        self.query_cache.clear()
        logger.info("查询缓存已清空")

def main():
    """主函数，用于测试检索模块"""
    import sys
    import os
    from pathlib import Path
    
    # 添加src目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("=== 检索模块测试 ===")
        
        # 1. 初始化组件
        print("1. 初始化检索引擎...")
        
        # 初始化向量存储（假设已有索引）
        vector_store = VectorStoreManager(
            store_type=VECTOR_STORE_TYPE,
            store_path=str(VECTOR_STORE_PATH)
        )
        
        # 初始化嵌入引擎
        embedding_engine = EmbeddingEngine()
        
        # 初始化检索引擎
        retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
        
        print("   检索引擎初始化完成")
        
        # 2. 测试查询处理
        print("\n2. 测试查询处理...")
        test_query = "金属材料拉伸强度测试价格不超过500元"
        query_analysis = retrieval_engine.query_processor.analyze_query(test_query)
        
        print(f"   原始查询: {query_analysis.original_query}")
        print(f"   处理后查询: {query_analysis.processed_query}")
        print(f"   查询类型: {query_analysis.query_type}")
        print(f"   关键词: {query_analysis.keywords}")
        print(f"   过滤条件: {query_analysis.filters}")
        print(f"   意图: {query_analysis.intent}")
        print(f"   置信度: {query_analysis.confidence:.2f}")
        
        # 3. 测试不同检索方法
        print("\n3. 测试检索功能...")
        
        test_queries = [
            "材料强度测试",
            "金属成分分析CMA认证",
            "塑料老化试验周期",
            "环境检测项目价格"
        ]
        
        for query in test_queries:
            print(f"\n   查询: {query}")
            
            # 语义搜索
            semantic_results = retrieval_engine.semantic_search(query, top_k=3)
            print(f"   语义搜索结果数: {len(semantic_results)}")
            
            # 混合搜索
            hybrid_results = retrieval_engine.hybrid_search(query, top_k=3)
            print(f"   混合搜索结果数: {len(hybrid_results)}")
            
            # 智能搜索
            smart_results = retrieval_engine.smart_search(query, top_k=3)
            print(f"   智能搜索结果数: {len(smart_results)}")
            
            # 显示第一个结果
            if smart_results:
                first_result = smart_results[0]
                print(f"   最佳结果: {first_result.document.content[:100]}...")
                print(f"   分数: {first_result.score:.4f}")
                print(f"   检索类型: {first_result.retrieval_type}")
        
        # 4. 测试多查询融合
        print("\n4. 测试多查询融合...")
        multi_queries = ["金属测试", "强度检测", "力学性能"]
        fusion_results = retrieval_engine.multi_query_search(multi_queries, top_k=5)
        print(f"   融合结果数: {len(fusion_results)}")
        
        # 5. 显示统计信息
        print("\n5. 检索统计信息:")
        stats = retrieval_engine.get_statistics()
        print(f"   查询总数: {stats.query_count}")
        print(f"   平均响应时间: {stats.avg_response_time:.4f}秒")
        print(f"   搜索文档总数: {stats.total_documents_searched}")
        
        print("\n✅ 检索模块测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 