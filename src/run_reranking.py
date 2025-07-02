"""
重排序模块运行脚本
测试和演示重排序功能
"""
import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor, Document
from embedding import EmbeddingEngine  
from vector_store import VectorStoreManager
from retrieval import RetrievalEngine
from reranking import RerankingEngine
from config import *

def create_test_documents():
    """创建测试文档"""
    documents = [
        Document(
            content="金属材料拉伸强度测试是评估材料机械性能的重要方法，通过拉伸试验可以确定材料的屈服强度、抗拉强度、延伸率等关键参数。",
            metadata={"type": "测试方法", "category": "金属", "test_type": "拉伸"},
            doc_id="doc_001"
        ),
        Document(
            content="塑料制品的成分分析包括定性和定量分析两个方面，常用的分析方法有红外光谱、核磁共振、质谱等技术手段。",
            metadata={"type": "分析方法", "category": "塑料", "analysis_type": "成分"},
            doc_id="doc_002"
        ),
        Document(
            content="金属疲劳试验用于研究材料在循环载荷下的性能表现，包括疲劳强度、疲劳寿命、裂纹扩展等关键指标。",
            metadata={"type": "疲劳测试", "category": "金属", "test_type": "疲劳"},
            doc_id="doc_003"
        ),
        Document(
            content="环境检测中重金属污染物的检测方法和标准，包括土壤、水体、大气中重金属元素的定量分析技术。",
            metadata={"type": "环境检测", "category": "重金属", "detection_type": "污染"},
            doc_id="doc_004"
        ),
        Document(
            content="建筑材料强度测试包括压缩强度、抗弯强度等多项指标，确保建筑物的结构安全和使用性能。",
            metadata={"type": "强度测试", "category": "建材", "test_type": "强度"},
            doc_id="doc_005"
        ),
        Document(
            content="高分子材料的热性能分析，包括玻璃化转变温度、熔点、热分解温度等热力学性质测定。",
            metadata={"type": "热分析", "category": "高分子", "analysis_type": "热性能"},
            doc_id="doc_006"
        ),
        Document(
            content="材料表面处理技术，包括电镀、喷涂、阳极氧化等表面改性方法，提高材料的耐腐蚀性和装饰性。",
            metadata={"type": "表面处理", "category": "通用", "treatment_type": "表面"},
            doc_id="doc_007"
        ),
        Document(
            content="金属腐蚀试验评估材料在特定环境下的腐蚀行为，包括均匀腐蚀、点蚀、应力腐蚀等腐蚀类型。",
            metadata={"type": "腐蚀测试", "category": "金属", "test_type": "腐蚀"},
            doc_id="doc_008"
        )
    ]
    return documents

def test_basic_reranking():
    """测试基础重排序功能"""
    print("=== 测试基础重排序功能 ===")
    
    # 初始化重排序引擎
    print("1. 初始化重排序引擎...")
    reranking_engine = RerankingEngine()
    
    # 创建测试文档
    documents = create_test_documents()
    
    # 测试查询
    queries = [
        "金属材料强度测试方法",
        "塑料成分分析技术", 
        "环境污染检测标准",
        "建筑材料性能评估"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. 查询: {query}")
        print("   原始文档顺序:")
        for j, doc in enumerate(documents[:5]):
            print(f"      {j+1}. {doc.content[:40]}...")
        
        # 执行重排序
        start_time = time.time()
        results = reranking_engine.rerank(query, documents, top_k=3)
        duration = time.time() - start_time
        
        print("   重排序结果:")
        for j, result in enumerate(results, 1):
            print(f"      {j}. 分数: {result.relevance_score:.4f} | 原排名: {result.original_rank+1} -> 新排名: {j}")
            print(f"         内容: {result.document.content[:40]}...")
        
        print(f"   处理时间: {duration:.3f}秒")

def test_retrieval_reranking():
    """测试检索+重排序集成功能"""
    print("\n=== 测试检索+重排序集成功能 ===")
    
    try:
        # 初始化组件
        print("1. 初始化核心组件...")
        data_processor = DataProcessor()
        embedding_engine = EmbeddingEngine()
        vector_store = VectorStoreManager(VECTOR_STORE_TYPE)
        retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
        reranking_engine = RerankingEngine()
        
        # 加载实际数据 
        print("2. 加载文档数据...")
        documents = data_processor.process_excel_to_documents()[:50]  # 使用前50个文档测试
        print(f"   加载了 {len(documents)} 个文档")
        
        # 向量化
        print("3. 文档向量化...")
        embeddings = embedding_engine.embed_documents(documents)
        print(f"   向量化完成，维度: {len(embeddings[0])}")
        
        # 创建或加载向量索引
        print("4. 创建向量索引...")
        vector_store.add_documents(documents, embeddings)
        print("   向量索引创建完成")
        
        # 测试查询
        test_queries = [
            "金属材料拉伸试验测试",
            "塑料成分定量分析方法",
            "建筑材料强度性能检测"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 查询: {query}")
            
            # 第一阶段：检索
            print("   第一阶段 - 检索:")
            retrieval_results = retrieval_engine.semantic_search(query, top_k=10)
            
            print(f"   检索到 {len(retrieval_results)} 个结果")
            for j, result in enumerate(retrieval_results[:3], 1):
                print(f"      {j}. 相似度: {result.score:.4f} - {result.document.content[:40]}...")
            
            # 第二阶段：重排序
            print("   第二阶段 - 重排序:")
            rerank_results = reranking_engine.rerank_retrieval_results(
                query, retrieval_results, top_k=3
            )
            
            print(f"   重排序后 {len(rerank_results)} 个结果")
            for j, result in enumerate(rerank_results, 1):
                print(f"      {j}. 相关性: {result.relevance_score:.4f} (原排名: {result.original_rank+1})")
                print(f"         内容: {result.document.content[:40]}...")
                
                # 显示原始检索信息
                if result.metadata and 'original_retrieval_score' in result.metadata:
                    orig_score = result.metadata['original_retrieval_score']
                    print(f"         原始相似度: {orig_score:.4f}")
        
    except Exception as e:
        print(f"   集成测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_batch_reranking():
    """测试批量重排序"""
    print("\n=== 测试批量重排序 ===")
    
    reranking_engine = RerankingEngine()
    documents = create_test_documents()
    
    # 批量查询
    queries = [
        "金属强度测试",
        "塑料分析方法", 
        "环境检测技术",
        "建材性能评估",
        "表面处理工艺"
    ]
    
    # 为每个查询准备不同的文档子集
    documents_list = [
        documents[:4],  # 前4个文档
        documents[1:6], # 文档2-6
        documents[2:7], # 文档3-7
        documents[3:8], # 文档4-8
        documents[4:]   # 后4个文档
    ]
    
    print(f"批量重排序 {len(queries)} 个查询...")
    
    start_time = time.time()
    batch_results = reranking_engine.batch_rerank(queries, documents_list, top_k=2)
    duration = time.time() - start_time
    
    print(f"批量处理完成，总耗时: {duration:.3f}秒")
    
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        print(f"\n{i+1}. 查询: {query}")
        print(f"   结果数: {len(results)}")
        
        for j, result in enumerate(results, 1):
            print(f"   {j}. 分数: {result.relevance_score:.4f} - {result.document.content[:30]}...")

def test_performance_analysis():
    """性能分析测试"""
    print("\n=== 性能分析测试 ===")
    
    reranking_engine = RerankingEngine()
    documents = create_test_documents()
    
    # 测试不同数量的文档
    doc_counts = [3, 5, 8]
    query = "金属材料性能测试分析"
    
    print("测试不同文档数量的重排序性能:")
    
    for count in doc_counts:
        test_docs = documents[:count]
        
        # 执行多次测试取平均值
        times = []
        for _ in range(3):
            start_time = time.time()
            results = reranking_engine.rerank(query, test_docs, top_k=min(3, count))
            duration = time.time() - start_time
            times.append(duration)
        
        avg_time = sum(times) / len(times)
        print(f"   文档数: {count}, 平均耗时: {avg_time:.4f}秒")
    
    # 测试缓存性能
    print("\n测试缓存性能:")
    
    # 首次查询（无缓存）
    start_time = time.time()
    reranking_engine.rerank(query, documents, top_k=3)
    first_time = time.time() - start_time
    
    # 重复查询（有缓存）
    start_time = time.time()
    reranking_engine.rerank(query, documents, top_k=3)
    cached_time = time.time() - start_time
    
    print(f"   首次查询: {first_time:.4f}秒")
    print(f"   缓存查询: {cached_time:.4f}秒")
    print(f"   性能提升: {(first_time/cached_time):.1f}x")

def show_statistics():
    """显示统计信息"""
    print("\n=== 重排序引擎统计信息 ===")
    
    reranking_engine = RerankingEngine()
    documents = create_test_documents()
    
    # 执行一些操作来生成统计数据
    queries = ["测试查询1", "测试查询2", "测试查询3"]
    for query in queries:
        reranking_engine.rerank(query, documents[:5], top_k=3)
    
    # 获取统计信息
    stats = reranking_engine.get_statistics()
    
    print(f"模型名称: {stats.model_name}")
    print(f"总请求数: {stats.total_requests}")
    print(f"平均处理时间: {stats.avg_processing_time:.4f}秒")
    print(f"重排序文档总数: {stats.total_documents_reranked}")
    print(f"平均分数提升: {stats.avg_score_improvement:.4f}")
    print(f"缓存命中率: {stats.cache_hit_rate:.2%}")
    print(f"最后更新时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.last_updated))}")
    
    # 缓存统计
    if hasattr(reranking_engine, 'cache') and reranking_engine.cache:
        cache_stats = reranking_engine.cache.get_cache_stats()
        print(f"\n缓存统计:")
        print(f"缓存命中: {cache_stats['hits']}")
        print(f"缓存未命中: {cache_stats['misses']}")
        print(f"缓存大小: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")

def main():
    """主函数"""
    print("🔄 重排序模块演示程序")
    print("=" * 50)
    
    try:
        # 测试1：基础重排序功能
        test_basic_reranking()
        
        # 测试2：检索+重排序集成
        test_retrieval_reranking()
        
        # 测试3：批量重排序
        test_batch_reranking()
        
        # 测试4：性能分析
        test_performance_analysis()
        
        # 显示统计信息
        show_statistics()
        
        print("\n✅ 重排序模块演示完成！")
        print("\n📊 重排序模块功能特点:")
        print("   ✓ 基于gte-rerank-v2模型的高精度重排序")
        print("   ✓ 智能缓存机制，提升重复查询性能")
        print("   ✓ 批量处理支持，提高处理效率")
        print("   ✓ 与检索模块无缝集成")
        print("   ✓ 完整的性能监控和统计")
        print("   ✓ 错误处理和降级机制")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 