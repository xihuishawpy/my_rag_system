"""
检索模块运行脚本
演示和测试检索功能
"""
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import RetrievalEngine, QueryProcessor, RetrievalResult
from vector_store import VectorStoreManager
from embedding import EmbeddingEngine
from config import VECTOR_STORE_PATH, EXCEL_FILE_PATH, TOP_K, VECTOR_STORE_TYPE

def test_query_processing():
    """测试查询处理功能"""
    print("=== 测试查询处理 ===")
    
    processor = QueryProcessor()
    
    test_queries = [
        "金属材料拉伸强度测试",
        "塑料成分分析价格不超过300元",
        "环境检测CMA认证实验室",
        "橡胶老化试验周期7天以内",
        "什么是金属疲劳测试？",
        "比较不同材料的硬度测试方法"
    ]
    
    print("查询分析结果:")
    print("-" * 80)
    
    for query in test_queries:
        analysis = processor.analyze_query(query)
        print(f"原始查询: {analysis.original_query}")
        print(f"处理后: {analysis.processed_query}")
        print(f"类型: {analysis.query_type} | 意图: {analysis.intent} | 置信度: {analysis.confidence:.2f}")
        print(f"关键词: {analysis.keywords}")
        if analysis.filters:
            print(f"过滤条件: {analysis.filters}")
        print("-" * 80)

def test_semantic_search(retrieval_engine: RetrievalEngine):
    """测试语义搜索"""
    print("\n=== 测试语义搜索 ===")
    
    test_queries = [
        "金属材料力学性能测试",
        "塑料制品成分检测",
        "环境污染物检测",
        "食品安全检测项目",
        "建筑材料强度测试"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        start_time = time.time()
        
        results = retrieval_engine.semantic_search(query, top_k=5)
        
        duration = time.time() - start_time
        print(f"搜索时间: {duration:.3f}秒")
        print(f"结果数量: {len(results)}")
        
        if results:
            print("前3个结果:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. 分数: {result.score:.4f}")
                print(f"     内容: {result.document.content[:100]}...")
                if result.document.metadata:
                    sample_type = result.document.metadata.get('raw_样品类别', 'N/A')
                    test_item = result.document.metadata.get('raw_测试项目', 'N/A')
                    print(f"     样品类别: {sample_type} | 测试项目: {test_item}")
                print()

def test_hybrid_search(retrieval_engine: RetrievalEngine):
    """测试混合搜索"""
    print("\n=== 测试混合搜索 ===")
    
    test_queries = [
        "金属拉伸强度测试方法",
        "塑料材料老化性能检测",
        "环境水质重金属分析"
    ]
    
    alphas = [0.3, 0.5, 0.7, 0.9]  # 不同的语义权重
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        for alpha in alphas:
            start_time = time.time()
            results = retrieval_engine.hybrid_search(query, top_k=3, alpha=alpha)
            duration = time.time() - start_time
            
            print(f"  α={alpha} | 时间: {duration:.3f}秒 | 结果数: {len(results)}")
            
            if results:
                best_result = results[0]
                print(f"    最佳: {best_result.document.content[:80]}... (分数: {best_result.score:.4f})")
                if best_result.metadata:
                    semantic_score = best_result.metadata.get('semantic_score', 0)
                    keyword_score = best_result.metadata.get('keyword_score', 0)
                    print(f"    语义: {semantic_score:.4f} | 关键词: {keyword_score:.4f}")

def test_filtered_search(retrieval_engine: RetrievalEngine):
    """测试过滤搜索"""
    print("\n=== 测试过滤搜索 ===")
    
    test_cases = [
        {
            "query": "材料强度测试",
            "filters": {"sample_category": "金属"},
            "description": "金属类材料"
        },
        {
            "query": "成分分析",
            "filters": {"CMA": True},
            "description": "CMA认证"
        },
        {
            "query": "环境检测",
            "filters": {"CNAS": True},
            "description": "CNAS认可"
        }
    ]
    
    for case in test_cases:
        print(f"\n测试案例: {case['description']}")
        print(f"查询: {case['query']}")
        print(f"过滤条件: {case['filters']}")
        
        start_time = time.time()
        results = retrieval_engine.filtered_search(case['query'], case['filters'], top_k=5)
        duration = time.time() - start_time
        
        print(f"搜索时间: {duration:.3f}秒")
        print(f"结果数量: {len(results)}")
        
        if results:
            print("结果预览:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result.document.content[:80]}...")
                print(f"     分数: {result.score:.4f}")
                
                # 验证过滤条件
                metadata = result.document.metadata
                if metadata:
                    if 'sample_category' in case['filters']:
                        actual_category = metadata.get('raw_样品类别', 'N/A')
                        print(f"     样品类别: {actual_category}")
                    if case['filters'].get('CMA'):
                        cma_status = metadata.get('raw_CMA', 'N/A')
                        print(f"     CMA认证: {cma_status}")
                    if case['filters'].get('CNAS'):
                        cnas_status = metadata.get('raw_CNAS', 'N/A')
                        print(f"     CNAS认可: {cnas_status}")

def test_multi_query_search(retrieval_engine: RetrievalEngine):
    """测试多查询融合搜索"""
    print("\n=== 测试多查询融合搜索 ===")
    
    test_cases = [
        {
            "queries": ["金属测试", "力学性能", "强度检测"],
            "description": "金属力学性能相关"
        },
        {
            "queries": ["塑料检测", "成分分析", "物理性能"],
            "description": "塑料检测相关"
        },
        {
            "queries": ["环境监测", "水质检测", "污染物分析"],
            "description": "环境检测相关"
        }
    ]
    
    fusion_methods = ["rrf", "weighted", "max"]
    
    for case in test_cases:
        print(f"\n测试案例: {case['description']}")
        print(f"查询列表: {case['queries']}")
        
        for method in fusion_methods:
            start_time = time.time()
            results = retrieval_engine.multi_query_search(
                case['queries'], 
                top_k=5, 
                fusion_method=method
            )
            duration = time.time() - start_time
            
            print(f"  融合方法: {method} | 时间: {duration:.3f}秒 | 结果数: {len(results)}")
            
            if results:
                best_result = results[0]
                print(f"    最佳结果: {best_result.document.content[:60]}...")
                print(f"    分数: {best_result.score:.4f}")

def test_smart_search(retrieval_engine: RetrievalEngine):
    """测试智能搜索"""
    print("\n=== 测试智能搜索 ===")
    
    test_queries = [
        "材料测试",  # 简单查询
        "金属材料拉伸强度测试价格不超过500元",  # 带过滤条件
        "什么是塑料老化试验，如何进行测试，周期多长？",  # 复杂查询
        "环境检测CMA认证实验室",  # 带认证要求
        "比较不同材料的硬度测试方法和标准"  # 比较类查询
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        start_time = time.time()
        results = retrieval_engine.smart_search(query, top_k=3)
        duration = time.time() - start_time
        
        print(f"搜索时间: {duration:.3f}秒")
        print(f"结果数量: {len(results)}")
        
        if results:
            first_result = results[0]
            print(f"策略: {first_result.retrieval_type}")
            print(f"最佳结果: {first_result.document.content[:100]}...")
            print(f"分数: {first_result.score:.4f}")
            
            # 显示查询分析信息
            if first_result.metadata and 'query_analysis' in first_result.metadata:
                analysis = first_result.metadata['query_analysis']
                print(f"查询分析: 类型={analysis['query_type']}, 意图={analysis['intent']}, 置信度={analysis['confidence']:.2f}")

def benchmark_retrieval_performance(retrieval_engine: RetrievalEngine):
    """性能基准测试"""
    print("\n=== 检索性能基准测试 ===")
    
    # 不同复杂度的查询
    benchmark_queries = {
        "简单查询": [
            "金属测试", "塑料检测", "环境监测", "食品检验", "建材检测"
        ],
        "中等查询": [
            "金属材料力学性能测试", "塑料制品成分分析", "环境水质污染物检测",
            "食品添加剂安全检验", "建筑材料强度测试"
        ],
        "复杂查询": [
            "金属材料拉伸强度和硬度测试CMA认证实验室价格",
            "塑料制品老化性能和化学成分分析CNAS认可机构",
            "环境土壤重金属污染物检测周期和标准要求"
        ]
    }
    
    methods = [
        ("语义搜索", lambda q: retrieval_engine.semantic_search(q, top_k=10)),
        ("混合搜索", lambda q: retrieval_engine.hybrid_search(q, top_k=10)),
        ("智能搜索", lambda q: retrieval_engine.smart_search(q, top_k=10))
    ]
    
    results = {}
    
    for complexity, queries in benchmark_queries.items():
        print(f"\n{complexity}:")
        results[complexity] = {}
        
        for method_name, method_func in methods:
            times = []
            result_counts = []
            
            for query in queries:
                start_time = time.time()
                search_results = method_func(query)
                duration = time.time() - start_time
                
                times.append(duration)
                result_counts.append(len(search_results))
            
            avg_time = sum(times) / len(times)
            avg_results = sum(result_counts) / len(result_counts)
            
            results[complexity][method_name] = {
                'avg_time': avg_time,
                'avg_results': avg_results
            }
            
            print(f"  {method_name}: 平均时间={avg_time:.3f}秒, 平均结果数={avg_results:.1f}")
    
    return results

def save_test_results(results: Dict[str, Any], output_path: str = "retrieval_test_results.json"):
    """保存测试结果"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 测试结果已保存到: {output_path}")
    except Exception as e:
        print(f"\n❌ 保存测试结果失败: {e}")

def main():
    """主函数"""
    print("🔍 检索模块功能演示")
    print("=" * 60)
    
    try:
        # 1. 测试查询处理
        test_query_processing()
        
        # 2. 初始化检索引擎
        print("\n=== 初始化检索引擎 ===")
        
        # 检查向量存储是否存在
        if not Path(VECTOR_STORE_PATH).exists():
            print("❌ 向量存储不存在，请先运行 run_vector_store.py")
            return
        
        # 初始化组件
        print("加载向量存储...")
        vector_store = VectorStoreManager(
            store_type=VECTOR_STORE_TYPE,
            store_path=str(VECTOR_STORE_PATH)
        )
        
        print("初始化嵌入引擎...")
        embedding_engine = EmbeddingEngine()
        
        print("初始化检索引擎...")
        retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
        
        # 获取向量存储统计信息
        stats = vector_store.get_statistics()
        print(f"向量存储统计:")
        print(f"  文档总数: {stats.total_documents}")
        print(f"  嵌入维度: {stats.embedding_dimension}")
        print(f"  存储类型: {stats.storage_type}")
        print(f"  存储大小: {stats.storage_size_mb:.2f} MB")
        
        # 3. 运行各种测试
        test_semantic_search(retrieval_engine)
        test_hybrid_search(retrieval_engine)
        test_filtered_search(retrieval_engine)
        test_multi_query_search(retrieval_engine)
        test_smart_search(retrieval_engine)
        
        # 4. 性能基准测试
        benchmark_results = benchmark_retrieval_performance(retrieval_engine)
        
        # 5. 显示最终统计
        print("\n=== 检索引擎统计信息 ===")
        retrieval_stats = retrieval_engine.get_statistics()
        print(f"总查询数: {retrieval_stats.query_count}")
        print(f"平均响应时间: {retrieval_stats.avg_response_time:.4f}秒")
        print(f"搜索文档总数: {retrieval_stats.total_documents_searched}")
        
        # 6. 保存测试结果
        all_results = {
            "timestamp": time.time(),
            "vector_store_stats": {
                "total_documents": stats.total_documents,
                "embedding_dimension": stats.embedding_dimension,
                "storage_type": stats.storage_type,
                "storage_size_mb": stats.storage_size_mb
            },
            "retrieval_stats": {
                "query_count": retrieval_stats.query_count,
                "avg_response_time": retrieval_stats.avg_response_time,
                "total_documents_searched": retrieval_stats.total_documents_searched
            },
            "benchmark_results": benchmark_results
        }
        
        save_test_results(all_results)
        
        print("\n✅ 检索模块测试完成！")
        print("\n🎯 检索模块主要功能:")
        print("  ✓ 语义相似度搜索")
        print("  ✓ 混合搜索（语义+关键词）") 
        print("  ✓ 过滤搜索（元数据过滤）")
        print("  ✓ 多查询融合搜索")
        print("  ✓ 智能搜索（自动策略选择）")
        print("  ✓ 查询预处理和分析")
        print("  ✓ 性能监控和统计")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 