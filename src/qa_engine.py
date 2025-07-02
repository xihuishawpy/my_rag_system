"""
问答模块 (QA Engine)
集成检索和重排序功能，使用qwen-max模型生成最终答案
"""
import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from pathlib import Path

# OpenAI兼容客户端
from openai import OpenAI

# 导入其他模块
from data_processing import Document
from retrieval import RetrievalEngine, RetrievalResult
from reranking import RerankingEngine, RerankResult
from config import QA_MODEL, QWEN_API_KEY, QWEN_BASE_URL, VECTOR_STORE_TYPE, VECTOR_STORE_PATH

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """引用信息"""
    doc_id: str
    content: str
    relevance_score: float
    source_info: Dict[str, Any]
    rank: int

@dataclass
class QAResult:
    """问答结果对象"""
    question: str
    answer: str
    citations: List[Citation]
    retrieval_time: float
    rerank_time: float
    generation_time: float
    total_time: float
    confidence_score: float
    context_used: int
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationHistory:
    """对话历史"""
    question: str
    answer: str
    timestamp: datetime
    citations: List[Citation]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class QAStats:
    """问答统计信息"""
    total_questions: int
    avg_response_time: float
    avg_context_length: int
    avg_confidence_score: float
    total_retrieval_time: float
    total_rerank_time: float
    total_generation_time: float
    model_name: str
    last_updated: float

class QACache:
    """问答结果缓存"""
    
    def __init__(self, cache_dir: str = "./qa_cache", max_cache_size: int = 1000):
        """
        初始化问答缓存
        
        Args:
            cache_dir: 缓存目录
            max_cache_size: 最大缓存数量
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        
        self.memory_cache = {}
        self.cache_file = self.cache_dir / "qa_cache.json"
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        
        # 加载缓存
        self._load_cache()
        
        logger.info(f"问答缓存初始化完成，缓存大小: {len(self.memory_cache)}")
    
    def _generate_cache_key(self, question: str, context_ids: List[str]) -> str:
        """生成缓存键"""
        content = f"{question}|{json.dumps(sorted(context_ids))}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, question: str, context_ids: List[str]) -> Optional[QAResult]:
        """获取缓存的问答结果"""
        cache_key = self._generate_cache_key(question, context_ids)
        
        if cache_key in self.memory_cache:
            self.hits += 1
            cached_data = self.memory_cache[cache_key]
            
            # 检查缓存是否过期（24小时）
            if time.time() - cached_data['timestamp'] < 24 * 3600:
                logger.debug(f"问答缓存命中: {cache_key[:8]}...")
                return self._deserialize_qa_result(cached_data['result'])
            else:
                # 删除过期缓存
                del self.memory_cache[cache_key]
        
        self.misses += 1
        return None
    
    def set(self, question: str, context_ids: List[str], result: QAResult):
        """设置缓存"""
        cache_key = self._generate_cache_key(question, context_ids)
        
        # 缓存大小控制
        if len(self.memory_cache) >= self.max_cache_size:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k]['timestamp'])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = {
            'result': self._serialize_qa_result(result),
            'timestamp': time.time()
        }
        
        logger.debug(f"问答结果已缓存: {cache_key[:8]}...")
    
    def _serialize_qa_result(self, result: QAResult) -> Dict[str, Any]:
        """序列化QA结果"""
        return asdict(result)
    
    def _deserialize_qa_result(self, data: Dict[str, Any]) -> QAResult:
        """反序列化QA结果"""
        # 重构Citation对象
        citations = []
        for citation_data in data['citations']:
            # 检查是否已经是Citation对象
            if isinstance(citation_data, Citation):
                citations.append(citation_data)
            elif isinstance(citation_data, dict):
                citations.append(Citation(**citation_data))
            else:
                # 兼容其他格式，尝试转换为字典
                try:
                    if hasattr(citation_data, '__dict__'):
                        citations.append(Citation(**citation_data.__dict__))
                    else:
                        logger.warning(f"无法反序列化Citation对象: {type(citation_data)}")
                        continue
                except Exception as e:
                    logger.warning(f"Citation反序列化失败: {e}")
                    continue
        
        data['citations'] = citations
        return QAResult(**data)
    
    def _load_cache(self):
        """加载磁盘缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.memory_cache = json.load(f)
                logger.info(f"加载磁盘缓存: {len(self.memory_cache)} 条记录")
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
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.memory_cache),
            'max_cache_size': self.max_cache_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.memory_cache.clear()
        self.hits = 0
        self.misses = 0
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("问答缓存已清空")

class QAEngine:
    """问答引擎"""
    
    def __init__(self,
                 retrieval_engine: RetrievalEngine,
                 reranking_engine: RerankingEngine,
                 model_name: str = QA_MODEL,
                 api_key: str = None,
                 base_url: str = QWEN_BASE_URL,
                 cache_enabled: bool = True,
                 max_context_length: int = 4000,
                 temperature: float = 0.6):
        """
        初始化问答引擎
        
        Args:
            retrieval_engine: 检索引擎
            reranking_engine: 重排序引擎
            model_name: QA模型名称
            api_key: API密钥
            base_url: API基础URL
            cache_enabled: 是否启用缓存
            max_context_length: 最大上下文长度
            temperature: 生成温度
        """
        self.retrieval_engine = retrieval_engine
        self.reranking_engine = reranking_engine
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.temperature = temperature
        
        # 智能获取API密钥
        if api_key is None:
            import os
            api_key = os.getenv("DASHSCOPE_API_KEY") or QWEN_API_KEY
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 初始化缓存
        if cache_enabled:
            self.cache = QACache()
        else:
            self.cache = None
        
        # 对话历史
        self.conversation_history: List[ConversationHistory] = []
        
        # 统计信息
        self.stats = QAStats(
            total_questions=0,
            avg_response_time=0.0,
            avg_context_length=0,
            avg_confidence_score=0.0,
            total_retrieval_time=0.0,
            total_rerank_time=0.0,
            total_generation_time=0.0,
            model_name=model_name,
            last_updated=time.time()
        )
        
        logger.info(f"问答引擎初始化完成，模型: {model_name}")
    
    def answer_question(self, 
                       question: str, 
                       context_limit: int = 5,
                       search_method: str = "hybrid",
                       include_history: bool = False) -> QAResult:
        """
        回答问题
        
        Args:
            question: 用户问题
            context_limit: 上下文文档数量限制
            search_method: 搜索方法 (semantic, hybrid, smart)
            include_history: 是否包含对话历史
            
        Returns:
            QAResult: 问答结果
        """
        start_time = time.time()
        
        try:
            # 1. 检索相关文档
            retrieval_start = time.time()
            if search_method == "semantic":
                retrieval_results = self.retrieval_engine.semantic_search(question, top_k=context_limit*2)
            elif search_method == "hybrid":
                retrieval_results = self.retrieval_engine.hybrid_search(question, top_k=context_limit*2)
            else:  # smart
                retrieval_results = self.retrieval_engine.smart_search(question, top_k=context_limit*2)
            
            retrieval_time = time.time() - retrieval_start
            
            if not retrieval_results:
                return QAResult(
                    question=question,
                    answer="抱歉，我没有找到相关的信息来回答您的问题。",
                    citations=[],
                    retrieval_time=retrieval_time,
                    rerank_time=0.0,
                    generation_time=0.0,
                    total_time=time.time() - start_time,
                    confidence_score=0.0,
                    context_used=0
                )
            
            # 2. 重排序
            rerank_start = time.time()
            rerank_results = self.reranking_engine.rerank_retrieval_results(
                question, retrieval_results, top_k=context_limit
            )
            rerank_time = time.time() - rerank_start
            
            # 3. 检查缓存
            context_ids = [result.document.doc_id for result in rerank_results]
            if self.cache:
                cached_result = self.cache.get(question, context_ids)
                if cached_result:
                    logger.debug(f"从缓存返回问答结果: {question[:50]}...")
                    return cached_result
            
            # 4. 构建上下文
            context_docs = [result.document for result in rerank_results]
            context_text = self._build_context(context_docs)
            
            # 5. 生成答案
            generation_start = time.time()
            answer, confidence = self._generate_answer(question, context_text, include_history)
            generation_time = time.time() - generation_start
            
            # 6. 构建引用信息
            citations = []
            for i, rerank_result in enumerate(rerank_results):
                citation = Citation(
                    doc_id=rerank_result.document.doc_id,
                    content=rerank_result.document.content,  # 保留完整内容，不截断
                    relevance_score=rerank_result.relevance_score,
                    source_info=rerank_result.document.metadata or {},
                    rank=i + 1
                )
                citations.append(citation)
            
            # 7. 构建结果
            total_time = time.time() - start_time
            
            result = QAResult(
                question=question,
                answer=answer,
                citations=citations,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                generation_time=generation_time,
                total_time=total_time,
                confidence_score=confidence,
                context_used=len(context_docs),
                metadata={
                    'search_method': search_method,
                    'context_length': len(context_text),
                    'model': self.model_name,
                    'timestamp': time.time()
                }
            )
            
            # 8. 更新缓存和统计
            if self.cache:
                self.cache.set(question, context_ids, result)
            
            self._update_stats(result)
            
            # 9. 添加到对话历史
            if include_history:
                history_item = ConversationHistory(
                    question=question,
                    answer=answer,
                    timestamp=datetime.now(),
                    citations=citations
                )
                self.conversation_history.append(history_item)
                
                # 保持历史长度限制
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"问答完成，耗时: {total_time:.3f}秒")
            return result
            
        except Exception as e:
            logger.error(f"问答过程出错: {e}")
            return QAResult(
                question=question,
                answer=f"处理您的问题时发生错误: {str(e)}",
                citations=[],
                retrieval_time=0.0,
                rerank_time=0.0,
                generation_time=0.0,
                total_time=time.time() - start_time,
                confidence_score=0.0,
                context_used=0,
                metadata={'error': str(e)}
            )
    
    def batch_answer(self, 
                    questions: List[str], 
                    context_limit: int = 5,
                    search_method: str = "smart") -> List[QAResult]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            context_limit: 上下文限制
            search_method: 搜索方法
            
        Returns:
            List[QAResult]: 批量问答结果
        """
        logger.info(f"开始批量问答，问题数: {len(questions)}")
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{len(questions)}: {question[:50]}...")
            result = self.answer_question(question, context_limit, search_method)
            results.append(result)
        
        logger.info(f"批量问答完成")
        return results
    
    def conversational_qa(self, 
                         question: str, 
                         context_limit: int = 5) -> QAResult:
        """
        对话式问答，考虑历史上下文
        
        Args:
            question: 当前问题
            context_limit: 上下文限制
            
        Returns:
            QAResult: 问答结果
        """
        return self.answer_question(question, context_limit, "hybrid", include_history=True)
    
    def generate_with_citations(self, question: str, context_limit: int = 5) -> QAResult:
        """
        生成带详细引用的答案
        
        Args:
            question: 问题
            context_limit: 上下文限制
            
        Returns:
            QAResult: 带引用的问答结果
        """
        result = self.answer_question(question, context_limit)
        
        # 在答案中添加引用标记
        if result.citations:
            cited_answer = result.answer
            
            # 添加引用列表
            cited_answer += "\n\n**参考资料:**\n"
            for citation in result.citations:
                cited_answer += f"{citation.rank}. {citation.content}\n"
                cited_answer += f"   (相关性分数: {citation.relevance_score:.3f})\n\n"
            
            result.answer = cited_answer
        
        return result
    
    def _build_context(self, documents: List[Document]) -> str:
        """构建上下文文本"""
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"文档{i+1}: {doc.content}"
            
            # 检查长度限制
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, 
                        question: str, 
                        context: str, 
                        include_history: bool = False) -> Tuple[str, float]:
        """
        生成答案
        
        Args:
            question: 问题
            context: 上下文
            include_history: 是否包含历史
            
        Returns:
            Tuple[str, float]: (答案, 置信度)
        """
        # 构建系统提示
        system_prompt = """你是一个专业的实验室检测能力问答助手。请基于提供的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文信息
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 如果涉及技术参数或标准，请保持准确性
5. 可以引用具体的实验室能力或服务项目"""
        
        # 构建用户提示
        user_prompt = f"""上下文信息：
{context}

用户问题：{question}

请基于上述上下文信息回答用户问题："""
        
        # 添加对话历史
        if include_history and self.conversation_history:
            history_text = "\n".join([
                f"Q: {hist.question}\nA: {hist.answer}" 
                for hist in self.conversation_history[-3:]  # 最近3轮对话
            ])
            user_prompt = f"对话历史：\n{history_text}\n\n{user_prompt}"
        
        try:
            # 调用qwen-max API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 计算置信度（基于答案长度和是否包含"不确定"等词）
            confidence = self._calculate_confidence(answer, context)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"抱歉，生成答案时遇到技术问题：{str(e)}", 0.0
    
    def _calculate_confidence(self, answer: str, context: str) -> float:
        """计算答案置信度"""
        confidence = 0.8  # 基础置信度
        
        # 长度因子
        if len(answer) < 20:
            confidence -= 0.2
        elif len(answer) > 100:
            confidence += 0.1
        
        # 不确定性词汇
        uncertainty_words = ["不确定", "可能", "也许", "大概", "似乎", "抱歉", "没有找到"]
        for word in uncertainty_words:
            if word in answer:
                confidence -= 0.2
                break
        
        # 具体性指标
        specific_words = ["测试", "检测", "分析", "标准", "规程", "方法"]
        specific_count = sum(1 for word in specific_words if word in answer)
        confidence += specific_count * 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def _update_stats(self, result: QAResult):
        """更新统计信息"""
        self.stats.total_questions += 1
        
        # 更新平均值
        total = self.stats.total_questions
        
        self.stats.avg_response_time = (
            (self.stats.avg_response_time * (total - 1) + result.total_time) / total
        )
        
        self.stats.avg_context_length = (
            (self.stats.avg_context_length * (total - 1) + result.context_used) / total
        )
        
        self.stats.avg_confidence_score = (
            (self.stats.avg_confidence_score * (total - 1) + result.confidence_score) / total
        )
        
        # 累计时间
        self.stats.total_retrieval_time += result.retrieval_time
        self.stats.total_rerank_time += result.rerank_time
        self.stats.total_generation_time += result.generation_time
        
        self.stats.last_updated = time.time()
    
    def get_statistics(self) -> QAStats:
        """获取统计信息"""
        return self.stats
    
    def get_conversation_history(self) -> List[ConversationHistory]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear_cache()
            logger.info("问答缓存已清空")

def main():
    """主函数，用于测试问答模块"""
    import sys
    import os
    from pathlib import Path
    
    # 添加src目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print("=== 问答模块测试 ===")
        
        # 1. 导入必要模块
        print("1. 导入模块...")
        from embedding import EmbeddingEngine
        from vector_store import VectorStoreManager
        from retrieval import RetrievalEngine
        from reranking import RerankingEngine
        
        print("   ✅ 模块导入成功")
        
        # 2. 初始化各个组件
        print("\n2. 初始化组件...")
        
        # 嵌入引擎
        embedding_engine = EmbeddingEngine()
        print("   ✅ 嵌入引擎初始化完成")
        
        # 向量存储
        vector_store = VectorStoreManager(VECTOR_STORE_TYPE, str(VECTOR_STORE_PATH))
        print("   ✅ 向量存储初始化完成")
        
        # 检索引擎
        retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
        print("   ✅ 检索引擎初始化完成")
        
        # 重排序引擎
        reranking_engine = RerankingEngine()
        print("   ✅ 重排序引擎初始化完成")
        
        # 问答引擎
        qa_engine = QAEngine(retrieval_engine, reranking_engine)
        print("   ✅ 问答引擎初始化完成")
        
        # 3. 测试基础问答
        print("\n3. 测试基础问答功能...")
        test_questions = [
            "金属强度测试方法",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   测试问题 {i}: {question}")
            
            result = qa_engine.answer_question(question, context_limit=3)
            
            print(f"   回答: {result.answer[:100]}...")
            print(f"   置信度: {result.confidence_score:.3f}")
            print(f"   检索时间: {result.retrieval_time:.3f}秒")
            print(f"   重排序时间: {result.rerank_time:.3f}秒")
            print(f"   生成时间: {result.generation_time:.3f}秒")
            print(f"   总时间: {result.total_time:.3f}秒")
            print(f"   引用数量: {len(result.citations)}")
        
        # 4. 测试对话式问答
        print("\n4. 测试对话式问答...")
        conv_questions = [
            "什么是材料性能测试？",
            "具体包括哪些测试项目？"
        ]
        
        for i, question in enumerate(conv_questions, 1):
            print(f"\n   对话问题 {i}: {question}")
            result = qa_engine.conversational_qa(question)
            print(f"   回答: {result.answer[:100]}...")
        
        # 5. 显示统计信息
        print("\n5. 问答引擎统计信息:")
        stats = qa_engine.get_statistics()
        print(f"   总问题数: {stats.total_questions}")
        print(f"   平均响应时间: {stats.avg_response_time:.3f}秒")
        print(f"   平均上下文长度: {stats.avg_context_length}")
        print(f"   平均置信度: {stats.avg_confidence_score:.3f}")
        
        print("\n✅ 问答模块测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 