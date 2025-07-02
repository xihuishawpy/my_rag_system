"""
问答引擎演示脚本
展示完整的RAG问答流程：检索 → 重排序 → 答案生成
"""
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """检查运行环境"""
    print("🔧 检查运行环境...")
    
    # 导入配置
    from config import QWEN_API_KEY
    
    # 检查API密钥
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    config_key = QWEN_API_KEY
    
    has_api_key = bool(dashscope_key or config_key)
    
    print(f"   DASHSCOPE_API_KEY: {'已设置' if dashscope_key else '未设置'}")
    print(f"   配置文件API密钥: {'已配置' if config_key else '未配置'}")
    print(f"   API密钥状态: {'✅ 可用' if has_api_key else '❌ 未配置'}")
    
    # 检查向量数据库
    vector_store_path = Path("./vector_store")
    print(f"   向量数据库: {'存在' if vector_store_path.exists() else '不存在'}")
    
    if not has_api_key:
        print("\n⚠️  警告：API密钥未配置")
        print("   请设置环境变量：export DASHSCOPE_API_KEY='your_api_key'")
        print("   或者在config.py中直接配置QWEN_API_KEY")
        print("\n💡 提示：系统将启用演示模式，使用模拟数据")
        return "demo_mode"  # 返回演示模式标识
    
    if not vector_store_path.exists():
        print("\n⚠️  警告：向量数据库不存在，请先运行向量存储模块")
        print("   可以运行：python run_vector_store_faiss.py")
        return False
    
    print("   ✅ 环境检查通过")
    return True

def initialize_rag_system():
    """初始化RAG系统"""
    print("\n🚀 初始化RAG系统...")
    
    try:
        # 导入必要模块
        from embedding import EmbeddingEngine
        from vector_store import VectorStoreManager
        from retrieval import RetrievalEngine
        from reranking import RerankingEngine
        from qa_engine import QAEngine
        from config import VECTOR_STORE_TYPE, VECTOR_STORE_PATH
        
        print("   ✅ 模块导入成功")
        
        # 1. 初始化嵌入引擎
        print("   📊 初始化嵌入引擎...")
        embedding_engine = EmbeddingEngine(cache_enabled=True)
        
        # 2. 初始化向量存储
        print("   🗄️  初始化向量存储...")
        vector_store = VectorStoreManager(VECTOR_STORE_TYPE, str(VECTOR_STORE_PATH))
        
        # 3. 初始化检索引擎
        print("   🔍 初始化检索引擎...")
        retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
        
        # 4. 初始化重排序引擎
        print("   📈 初始化重排序引擎...")
        reranking_engine = RerankingEngine(cache_enabled=True)
        
        # 5. 初始化问答引擎
        print("   🤖 初始化问答引擎...")
        qa_engine = QAEngine(
            retrieval_engine=retrieval_engine,
            reranking_engine=reranking_engine,
            cache_enabled=True,
            max_context_length=3000,
            temperature=0.7
        )
        
        print("   ✅ RAG系统初始化完成")
        return qa_engine
        
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        raise

def demonstrate_basic_qa(qa_engine):
    """演示基础问答功能"""
    print("\n📝 基础问答功能演示")
    print("=" * 50)
    
    # 测试问题集
    test_questions = [
        {
            "question": "金属材料的拉伸强度如何测试？",
            "description": "技术方法查询"
        },
        {
            "question": "实验室提供哪些环境检测服务？",
            "description": "服务项目查询"
        },
        {
            "question": "塑料制品成分分析有什么要求？",
            "description": "标准要求查询"
        },
        {
            "question": "金属疲劳试验的周期是多少？",
            "description": "时间参数查询"
        },
        {
            "question": "实验室的质量认证情况如何？",
            "description": "认证信息查询"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        description = test_case["description"]
        
        print(f"\n🔸 测试 {i}: {description}")
        print(f"   问题: {question}")
        
        # 执行问答
        result = qa_engine.answer_question(
            question=question,
            context_limit=3,
            search_method="hybrid"
        )
        
        print(f"   回答: {result.answer}")
        print(f"   置信度: {result.confidence_score:.3f}")
        print(f"   性能指标:")
        print(f"     - 检索: {result.retrieval_time:.3f}秒")
        print(f"     - 重排序: {result.rerank_time:.3f}秒") 
        print(f"     - 生成: {result.generation_time:.3f}秒")
        print(f"     - 总时间: {result.total_time:.3f}秒")
        print(f"   引用文档: {len(result.citations)}个")
        
        # 显示引用信息
        if result.citations:
            print(f"   引用文档:")
            for j, citation in enumerate(result.citations, 1):
                print(f"     {j}. 相关性: {citation.relevance_score:.3f}")
                # 显示引用内容摘要（保留前80字符用于命令行显示）
                content_preview = citation.content[:80] + "..." if len(citation.content) > 80 else citation.content
                print(f"        内容: {content_preview}")

def demonstrate_conversational_qa(qa_engine):
    """演示对话式问答"""
    print("\n💬 对话式问答演示")
    print("=" * 50)
    
    # 对话序列
    conversation = [
        "什么是材料性能测试？",
        "具体包括哪些测试项目？",
        "这些测试的价格大概是多少？",
        "测试周期通常需要多久？"
    ]
    
    print("开始对话式问答演示...")
    
    for i, question in enumerate(conversation, 1):
        print(f"\n👤 用户: {question}")
        
        # 使用对话式问答
        result = qa_engine.conversational_qa(question, context_limit=3)
        
        print(f"🤖 助手: {result.answer}")
        print(f"   (置信度: {result.confidence_score:.3f}, 耗时: {result.total_time:.3f}秒)")
    
    # 显示对话历史
    print(f"\n📚 对话历史记录: {len(qa_engine.get_conversation_history())} 轮")

def demonstrate_cited_answers(qa_engine):
    """演示带引用的答案"""
    print("\n📎 带引用答案演示")
    print("=" * 50)
    
    question = "金属材料检测有哪些标准和方法？"
    print(f"问题: {question}")
    
    # 生成带引用的答案
    result = qa_engine.generate_with_citations(question, context_limit=4)
    
    print(f"\n答案:\n{result.answer}")
    print(f"\n性能: 置信度={result.confidence_score:.3f}, 总时间={result.total_time:.3f}秒")

def demonstrate_batch_qa(qa_engine):
    """演示批量问答"""
    print("\n📦 批量问答演示")
    print("=" * 50)
    
    batch_questions = [
        "实验室的营业时间是什么？",
        "如何联系实验室预约检测？",
        "实验室的地址在哪里？",
        "检测报告多久可以出具？"
    ]
    
    print(f"批量处理 {len(batch_questions)} 个问题...")
    
    # 执行批量问答
    results = qa_engine.batch_answer(
        questions=batch_questions,
        context_limit=2,
        search_method="hybrid"
    )
    
    print(f"\n批量问答结果:")
    total_time = 0
    for i, (question, result) in enumerate(zip(batch_questions, results), 1):
        print(f"\n  {i}. 问题: {question}")
        print(f"     回答: {result.answer[:100]}{'...' if len(result.answer) > 100 else ''}")
        print(f"     置信度: {result.confidence_score:.3f}")
        print(f"     耗时: {result.total_time:.3f}秒")
        total_time += result.total_time
    
    print(f"\n  总耗时: {total_time:.3f}秒")
    print(f"  平均耗时: {total_time/len(batch_questions):.3f}秒/问题")

def show_system_statistics(qa_engine):
    """显示系统统计信息"""
    print("\n📊 系统统计信息")
    print("=" * 50)
    
    # 问答引擎统计
    qa_stats = qa_engine.get_statistics()
    print(f"问答引擎:")
    print(f"  总问题数: {qa_stats.total_questions}")
    print(f"  平均响应时间: {qa_stats.avg_response_time:.3f}秒")
    print(f"  平均上下文长度: {qa_stats.avg_context_length}")
    print(f"  平均置信度: {qa_stats.avg_confidence_score:.3f}")
    print(f"  总检索时间: {qa_stats.total_retrieval_time:.3f}秒")
    print(f"  总重排序时间: {qa_stats.total_rerank_time:.3f}秒")
    print(f"  总生成时间: {qa_stats.total_generation_time:.3f}秒")
    
    # 检索引擎统计  
    retrieval_stats = qa_engine.retrieval_engine.get_statistics()
    print(f"\n检索引擎:")
    print(f"  查询次数: {retrieval_stats.query_count}")
    print(f"  平均响应时间: {retrieval_stats.avg_response_time:.3f}秒")
    print(f"  搜索文档总数: {retrieval_stats.total_documents_searched}")
    print(f"  缓存命中率: {retrieval_stats.cache_hit_rate:.2%}")
    print(f"  检索准确率: {retrieval_stats.retrieval_accuracy:.3f}")
    if retrieval_stats.top_queries:
        print(f"  热门查询: {', '.join(retrieval_stats.top_queries[:3])}")
    
    # 重排序引擎统计
    rerank_stats = qa_engine.reranking_engine.get_statistics()
    print(f"\n重排序引擎:")
    print(f"  总请求数: {rerank_stats.total_requests}")
    print(f"  平均处理时间: {rerank_stats.avg_processing_time:.3f}秒")
    print(f"  缓存命中率: {rerank_stats.cache_hit_rate:.2%}")
    
    # 问答缓存统计
    if qa_engine.cache:
        cache_stats = qa_engine.cache.get_cache_stats()
        print(f"\n问答缓存:")
        print(f"  命中率: {cache_stats['hit_rate']:.2%}")
        print(f"  缓存大小: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")

def interactive_qa_session(qa_engine):
    """交互式问答会话"""
    print("\n💭 交互式问答会话")
    print("=" * 50)
    print("输入问题开始对话，输入 'exit' 或 'quit' 结束会话")
    print("输入 'history' 查看对话历史")
    print("输入 'stats' 查看统计信息")
    
    while True:
        try:
            question = input("\n❓ 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', '退出']:
                print("👋 再见！")
                break
            
            if question.lower() == 'history':
                history = qa_engine.get_conversation_history()
                if history:
                    print(f"\n📚 对话历史 ({len(history)} 轮):")
                    for i, conv in enumerate(history, 1):
                        print(f"  {i}. Q: {conv.question}")
                        print(f"     A: {conv.answer[:100]}...")
                else:
                    print("📚 暂无对话历史")
                continue
            
            if question.lower() == 'stats':
                show_system_statistics(qa_engine)
                continue
            
            # 执行问答
            print("🤔 思考中...")
            result = qa_engine.conversational_qa(question)
            
            print(f"\n💡 回答: {result.answer}")
            print(f"📊 置信度: {result.confidence_score:.3f} | 耗时: {result.total_time:.3f}秒")
            
            if result.citations:
                print(f"📄 参考了 {len(result.citations)} 个相关文档")
            
        except KeyboardInterrupt:
            print("\n\n👋 会话已中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 处理问题时出错: {e}")

def main():
    """主函数"""
    print("🔄 RAG问答系统演示")
    print("=" * 50)
    
    # 1. 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请检查配置")
        return
    
    try:
        # 2. 初始化系统
        qa_engine = initialize_rag_system()
        
        # 3. 功能演示
        demonstrate_basic_qa(qa_engine)
        
        demonstrate_conversational_qa(qa_engine)
        
        demonstrate_cited_answers(qa_engine)
        
        demonstrate_batch_qa(qa_engine)
        
        # 4. 显示统计信息
        show_system_statistics(qa_engine)
        
        print("\n🎉 RAG问答系统演示完成！")
        
        # 5. 可选的交互式会话
        choice = input("\n是否进入交互式问答会话？(y/N): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_qa_session(qa_engine)
        
    except Exception as e:
        print(f"\n❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 