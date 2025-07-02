"""
重排序模块
基于gte-rerank-v2模型提供文档重排序功能，提升检索精度
"""
import time
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from data_processing import Document
from retrieval import RetrievalResult
from config import RERANK_MODEL, RERANK_TOP_K, QWEN_API_KEY, QWEN_BASE_URL

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RerankResult:
    """重排序结果对象"""
    document: Document
    relevance_score: float
    original_rank: int
    new_rank: int
    score_improvement: float  # 重排序前后分数变化
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RerankStats:
    """重排序统计信息"""
    total_requests: int
    avg_processing_time: float
    total_documents_reranked: int
    avg_score_improvement: float
    cache_hit_rate: float
    model_name: str
    last_updated: float

class RerankCache:
    """重排序结果缓存"""
    
    def __init__(self, cache_size: int = 1000, cache_dir: str = "./cache"):
        """
        初始化缓存
        
        Args:
            cache_size: 缓存大小限制
            cache_dir: 缓存目录
        """
        self.cache_size = cache_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_cache = {}  # 内存缓存
        self.cache_file = self.cache_dir / "rerank_cache.json"
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        
        # 加载持久化缓存
        self._load_cache()
        
        logger.info(f"重排序缓存初始化完成，内存缓存大小: {len(self.memory_cache)}")
    
    def _generate_cache_key(self, query: str, documents: List[Document], top_k: int) -> str:
        """生成缓存键"""
        # 基于查询、文档内容和top_k生成唯一键
        doc_contents = [doc.content for doc in documents]
        content_str = f"{query}|{json.dumps(doc_contents, sort_keys=True)}|{top_k}"
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def get(self, query: str, documents: List[Document], top_k: int) -> Optional[List[RerankResult]]:
        """从缓存获取重排序结果"""
        cache_key = self._generate_cache_key(query, documents, top_k)
        
        if cache_key in self.memory_cache:
            self.hits += 1
            logger.debug(f"缓存命中: {cache_key[:8]}...")
            
            # 反序列化结果
            cached_data = self.memory_cache[cache_key]
            results = []
            for item in cached_data['results']:
                document = Document(
                    content=item['document']['content'],
                    metadata=item['document']['metadata'],
                    doc_id=item['document']['doc_id']
                )
                results.append(RerankResult(
                    document=document,
                    relevance_score=item['relevance_score'],
                    original_rank=item['original_rank'],
                    new_rank=item['new_rank'],
                    score_improvement=item['score_improvement'],
                    metadata=item.get('metadata')
                ))
            return results
        
        self.misses += 1
        return None
    
    def set(self, query: str, documents: List[Document], top_k: int, results: List[RerankResult]):
        """设置缓存"""
        cache_key = self._generate_cache_key(query, documents, top_k)
        
        # 序列化结果
        cached_data = {
            'results': [asdict(result) for result in results],
            'timestamp': time.time()
        }
        
        # 内存缓存大小控制
        if len(self.memory_cache) >= self.cache_size:
            # 删除最旧的缓存项
            oldest_key = min(self.memory_cache.keys(), 
                            key=lambda k: self.memory_cache[k]['timestamp'])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = cached_data
        logger.debug(f"缓存已更新: {cache_key[:8]}...")
        
        # 定期保存到磁盘
        if len(self.memory_cache) % 10 == 0:
            self._save_cache()
    
    def _load_cache(self):
        """从磁盘加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    disk_cache = json.load(f)
                
                # 只加载最近的缓存项
                current_time = time.time()
                valid_cache = {}
                
                for key, value in disk_cache.items():
                    # 缓存有效期为24小时
                    if current_time - value['timestamp'] < 24 * 3600:
                        valid_cache[key] = value
                
                self.memory_cache = valid_cache
                logger.info(f"从磁盘加载 {len(valid_cache)} 个缓存项")
                
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.memory_cache = {}
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_cache, f, ensure_ascii=False, indent=2)
            logger.debug("缓存已保存到磁盘")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.memory_cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.memory_cache.clear()
        self.hits = 0
        self.misses = 0
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("缓存已清空")

class RerankingEngine:
    """重排序引擎"""
    
    def __init__(self, 
                 model_name: str = RERANK_MODEL,
                 api_key: str = None,
                 base_url: str = QWEN_BASE_URL,
                 cache_enabled: bool = True,
                 max_workers: int = 4):
        """
        初始化重排序引擎
        
        Args:
            model_name: 重排序模型名称
            api_key: API密钥，如果为None则从环境变量获取
            base_url: API基础URL
            cache_enabled: 是否启用缓存
            max_workers: 并发处理的最大线程数
        """
        self.model_name = model_name
        
        # 智能获取API密钥
        if api_key is None:
            # 优先从环境变量获取
            import os
            api_key = os.getenv("DASHSCOPE_API_KEY") or QWEN_API_KEY
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.max_workers = max_workers
        
        # 初始化缓存
        if cache_enabled:
            self.cache = RerankCache()
        else:
            self.cache = None
        
        # 统计信息
        self.stats = RerankStats(
            total_requests=0,
            avg_processing_time=0.0,
            total_documents_reranked=0,
            avg_score_improvement=0.0,
            cache_hit_rate=0.0,
            model_name=model_name,
            last_updated=time.time()
        )
        
        # API客户端设置
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"重排序引擎初始化完成，模型: {model_name}")
    
    def rerank(self, 
               query: str, 
               documents: List[Document], 
               top_k: int = RERANK_TOP_K,
               return_documents: bool = True) -> List[RerankResult]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的top文档数量
            return_documents: 是否返回文档内容
            
        Returns:
            List[RerankResult]: 重排序结果列表
        """
        start_time = time.time()
        
        if not documents:
            return []
        
        # 限制输入文档数量
        max_docs = min(len(documents), 100)  # API限制
        input_documents = documents[:max_docs]
        
        # 检查缓存
        if self.cache:
            cached_results = self.cache.get(query, input_documents, top_k)
            if cached_results:
                logger.debug(f"从缓存返回重排序结果，查询: {query[:50]}...")
                return cached_results[:top_k]
        
        try:
            # 调用重排序API
            rerank_scores = self._call_rerank_api(query, input_documents)
            
            # 处理结果
            results = self._process_rerank_results(
                query, input_documents, rerank_scores, top_k
            )
            
            # 更新缓存
            if self.cache:
                self.cache.set(query, input_documents, top_k, results)
            
            # 更新统计
            self._update_stats(start_time, len(input_documents), results)
            
            logger.debug(f"重排序完成，查询: {query[:50]}..., 结果数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 降级处理：返回原始顺序
            return self._fallback_ranking(input_documents, top_k)
    
    def batch_rerank(self, 
                    queries: List[str], 
                    documents_list: List[List[Document]], 
                    top_k: int = RERANK_TOP_K) -> List[List[RerankResult]]:
        """
        批量重排序
        
        Args:
            queries: 查询列表
            documents_list: 文档列表的列表
            top_k: 每个查询返回的top文档数量
            
        Returns:
            List[List[RerankResult]]: 批量重排序结果
        """
        if len(queries) != len(documents_list):
            raise ValueError("查询数量与文档列表数量不匹配")
        
        logger.info(f"开始批量重排序，查询数: {len(queries)}")
        
        results = []
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.rerank, query, docs, top_k): i
                for i, (query, docs) in enumerate(zip(queries, documents_list))
            }
            
            # 按原始顺序收集结果
            index_results = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    index_results[index] = result
                except Exception as e:
                    logger.error(f"批量重排序第{index}个查询失败: {e}")
                    index_results[index] = []
            
            # 按索引顺序组织结果
            for i in range(len(queries)):
                results.append(index_results.get(i, []))
        
        logger.info(f"批量重排序完成")
        return results
    
    def rerank_retrieval_results(self, 
                                query: str, 
                                retrieval_results: List[RetrievalResult], 
                                top_k: int = RERANK_TOP_K) -> List[RerankResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            retrieval_results: 检索结果列表
            top_k: 返回的top文档数量
            
        Returns:
            List[RerankResult]: 重排序结果列表
        """
        # 提取文档
        documents = [result.document for result in retrieval_results]
        
        # 执行重排序
        rerank_results = self.rerank(query, documents, top_k)
        
        # 添加原始检索信息到元数据
        for rerank_result in rerank_results:
            # 找到对应的原始检索结果
            for i, retrieval_result in enumerate(retrieval_results):
                if retrieval_result.document.doc_id == rerank_result.document.doc_id:
                    if rerank_result.metadata is None:
                        rerank_result.metadata = {}
                    
                    rerank_result.metadata.update({
                        'original_retrieval_score': retrieval_result.score,
                        'original_retrieval_rank': retrieval_result.rank,
                        'retrieval_type': retrieval_result.retrieval_type
                    })
                    break
        
        return rerank_results
    
    def calculate_relevance_scores(self, 
                                  query: str, 
                                  documents: List[Document]) -> List[float]:
        """
        计算文档与查询的相关性分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            List[float]: 相关性分数列表
        """
        if not documents:
            return []
        
        try:
            # 调用重排序API获取分数
            scores = self._call_rerank_api(query, documents)
            return scores
            
        except Exception as e:
            logger.error(f"计算相关性分数失败: {e}")
            # 返回默认分数
            return [0.5] * len(documents)
    
    def _call_rerank_api(self, query: str, documents: List[Document]) -> List[float]:
        """
        调用重排序API
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            List[float]: 重排序分数列表
        """
        # 准备API请求数据
        doc_texts = [doc.content for doc in documents]
        
        # 检查API密钥
        if not self.api_key or self.api_key.strip() == "":
            raise Exception("API密钥未配置，请设置DASHSCOPE_API_KEY环境变量")
        
        # 构建请求
        if "gte-rerank" in self.model_name:
            # 阿里云DashScope重排序API
            api_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            
            request_data = {
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": doc_texts
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": len(doc_texts)
                }
            }
            
            # 使用DashScope专用的请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
        else:
            # 通用重排序API格式（如Cohere等）
            api_url = f"{self.base_url}/v1/rerank"
            
            request_data = {
                "model": self.model_name,
                "query": query,
                "documents": doc_texts,
                "top_n": len(doc_texts),
                "return_documents": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        
        # 发送请求
        response = requests.post(api_url, json=request_data, headers=headers, timeout=30)
        
        if response.status_code != 200:
            # 记录详细的错误信息
            logger.debug(f"API请求URL: {api_url}")
            logger.debug(f"请求数据: {request_data}")
            raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
        
        result = response.json()
        logger.debug(f"API响应: {result}")
        
        # 解析DashScope响应格式
        if "output" in result and "results" in result["output"]:
            # DashScope格式响应
            results_list = result["output"]["results"]
            scores = [0.0] * len(doc_texts)  # 初始化分数数组
            
            for result_item in results_list:
                index = result_item.get("index", 0)
                score = result_item.get("relevance_score", 0.0)
                if 0 <= index < len(scores):
                    scores[index] = float(score)
            
            return scores
        
        # 解析标准格式响应
        elif "results" in result:
            # 标准格式响应
            scores = []
            ranked_results = sorted(result["results"], key=lambda x: x.get("index", 0))
            
            for item in ranked_results:
                score = item.get("relevance_score", item.get("score", 0.0))
                scores.append(float(score))
            
            return scores
        
        elif "scores" in result:
            # 简化格式响应
            return [float(score) for score in result["scores"]]
        
        else:
            raise Exception(f"无法解析API响应格式: {result}")
    
    def _process_rerank_results(self, 
                               query: str, 
                               documents: List[Document], 
                               scores: List[float], 
                               top_k: int) -> List[RerankResult]:
        """
        处理重排序结果
        
        Args:
            query: 查询文本
            documents: 原始文档列表
            scores: 重排序分数列表
            top_k: 返回结果数量
            
        Returns:
            List[RerankResult]: 处理后的重排序结果
        """
        if len(documents) != len(scores):
            raise ValueError(f"文档数量({len(documents)})与分数数量({len(scores)})不匹配")
        
        # 创建(文档, 分数, 原始排名)的元组列表
        doc_score_tuples = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            doc_score_tuples.append((doc, score, i))
        
        # 按分数降序排序
        doc_score_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # 构建重排序结果
        results = []
        for new_rank, (doc, score, original_rank) in enumerate(doc_score_tuples[:top_k]):
            # 计算分数提升（这里使用排名变化作为代理指标）
            rank_improvement = original_rank - new_rank
            score_improvement = rank_improvement / len(documents) if len(documents) > 0 else 0
            
            result = RerankResult(
                document=doc,
                relevance_score=score,
                original_rank=original_rank,
                new_rank=new_rank,
                score_improvement=score_improvement,
                metadata={
                    'query': query,
                    'rerank_model': self.model_name,
                    'rerank_timestamp': time.time()
                }
            )
            results.append(result)
        
        return results
    
    def _fallback_ranking(self, documents: List[Document], top_k: int) -> List[RerankResult]:
        """
        降级处理：返回原始顺序的结果
        
        Args:
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            List[RerankResult]: 降级排序结果
        """
        results = []
        for i, doc in enumerate(documents[:top_k]):
            result = RerankResult(
                document=doc,
                relevance_score=0.5,  # 默认分数
                original_rank=i,
                new_rank=i,
                score_improvement=0.0,
                metadata={
                    'fallback': True,
                    'reason': 'API调用失败'
                }
            )
            results.append(result)
        
        return results
    
    def _update_stats(self, start_time: float, num_documents: int, results: List[RerankResult]):
        """更新统计信息"""
        duration = time.time() - start_time
        
        self.stats.total_requests += 1
        self.stats.total_documents_reranked += num_documents
        
        # 更新平均处理时间
        total_time = self.stats.avg_processing_time * (self.stats.total_requests - 1) + duration
        self.stats.avg_processing_time = total_time / self.stats.total_requests
        
        # 更新平均分数提升
        if results:
            current_improvement = sum(r.score_improvement for r in results) / len(results)
            total_improvement = self.stats.avg_score_improvement * (self.stats.total_requests - 1) + current_improvement
            self.stats.avg_score_improvement = total_improvement / self.stats.total_requests
        
        # 更新缓存命中率
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            self.stats.cache_hit_rate = cache_stats['hit_rate']
        
        self.stats.last_updated = time.time()
    
    def get_statistics(self) -> RerankStats:
        """获取重排序统计信息"""
        return self.stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear_cache()
            logger.info("重排序缓存已清空")

def main():
    """主函数，用于测试重排序模块"""
    import sys
    import os
    from pathlib import Path
    
    # 添加src目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("=== 重排序模块测试 ===")
        
        # 1. 初始化重排序引擎
        print("1. 初始化重排序引擎...")
        reranking_engine = RerankingEngine()
        print("   重排序引擎初始化完成")
        
        # 2. 创建测试文档
        print("\n2. 创建测试文档...")
        test_documents = [
            Document(
                content="金属材料拉伸强度测试是评估材料机械性能的重要方法",
                metadata={"type": "测试方法", "category": "金属"},
                doc_id="doc_1"
            ),
            Document(
                content="塑料制品的成分分析包括定性和定量分析两个方面",
                metadata={"type": "分析方法", "category": "塑料"},
                doc_id="doc_2"
            ),
            Document(
                content="金属疲劳试验用于研究材料在循环载荷下的性能表现",
                metadata={"type": "疲劳测试", "category": "金属"},
                doc_id="doc_3"
            ),
            Document(
                content="环境检测中重金属污染物的检测方法和标准",
                metadata={"type": "环境检测", "category": "重金属"},
                doc_id="doc_4"
            ),
            Document(
                content="建筑材料强度测试包括压缩强度、抗弯强度等多项指标",
                metadata={"type": "强度测试", "category": "建材"},
                doc_id="doc_5"
            )
        ]
        
        print(f"   创建了 {len(test_documents)} 个测试文档")
        
        # 3. 测试基础重排序功能
        print("\n3. 测试基础重排序功能...")
        query = "金属材料强度测试方法"
        
        results = reranking_engine.rerank(query, test_documents, top_k=3)
        
        print(f"   查询: {query}")
        print(f"   重排序结果数: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. 分数: {result.relevance_score:.4f} | 原排名: {result.original_rank} -> 新排名: {result.new_rank}")
            print(f"      内容: {result.document.content[:50]}...")
        
        # 4. 测试相关性分数计算
        print("\n4. 测试相关性分数计算...")
        scores = reranking_engine.calculate_relevance_scores(query, test_documents)
        
        print(f"   相关性分数: {[f'{s:.4f}' for s in scores]}")
        
        # 5. 测试批量重排序
        print("\n5. 测试批量重排序...")
        queries = [
            "金属测试方法",
            "塑料成分分析",
            "环境污染检测"
        ]
        
        documents_list = [test_documents[:3], test_documents[1:4], test_documents[2:]]
        
        batch_results = reranking_engine.batch_rerank(queries, documents_list, top_k=2)
        
        for i, (query, results) in enumerate(zip(queries, batch_results)):
            print(f"   查询{i+1}: {query}")
            print(f"   结果数: {len(results)}")
            if results:
                best_result = results[0]
                print(f"   最佳结果: 分数={best_result.relevance_score:.4f}")
        
        # 6. 显示统计信息
        print("\n6. 重排序统计信息:")
        stats = reranking_engine.get_statistics()
        print(f"   总请求数: {stats.total_requests}")
        print(f"   平均处理时间: {stats.avg_processing_time:.4f}秒")
        print(f"   重排序文档总数: {stats.total_documents_reranked}")
        print(f"   平均分数提升: {stats.avg_score_improvement:.4f}")
        print(f"   缓存命中率: {stats.cache_hit_rate:.2%}")
        
        print("\n✅ 重排序模块测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 