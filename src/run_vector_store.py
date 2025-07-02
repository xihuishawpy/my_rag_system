"""
向量存储模块运行脚本
演示向量存储的基本功能
"""
import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStoreManager
from data_processing import DataProcessor
from embedding import EmbeddingEngine
from config import DATA_DIR, VECTOR_STORE_PATH, VECTOR_STORE_TYPE

def demo_vector_store():
    """演示向量存储功能"""
    print("=== 向量存储模块演示 ===\n")
    
    try:
        # 1. 加载数据
        print("1. 加载Excel数据...")
        from config import EXCEL_FILE_PATH
        processor = DataProcessor(EXCEL_FILE_PATH)
        documents = processor.process_excel_to_documents()
        print(f"   已加载 {len(documents)} 个文档")
        
        # 2. 初始化嵌入引擎
        print("\n2. 初始化嵌入引擎...")
        embedding_engine = EmbeddingEngine()
        print("   嵌入引擎初始化完成")
        
        # 3. 生成文档嵌入
        print("\n3. 生成文档嵌入...")
        embeddings = []
        batch_size = 10
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embedding_engine.embed_documents(batch_docs)
            embeddings.extend(batch_embeddings)
            print(f"   已处理 {min(i+batch_size, len(documents))}/{len(documents)} 个文档")
        
        print(f"   嵌入维度: {len(embeddings[0]) if embeddings else 0}")
        
        # 4. 初始化向量存储
        print("\n4. 初始化向量存储...")
        vector_store = VectorStoreManager(
            store_type=VECTOR_STORE_TYPE,  
            store_path=str(VECTOR_STORE_PATH)
        )
        print("   向量存储初始化完成")
        
        # 5. 创建索引
        print("\n5. 创建向量索引...")
        vector_store.create_index(documents, embeddings)
        print("   索引创建完成")
        
        # 6. 获取统计信息
        print("\n6. 向量存储统计信息:")
        stats = vector_store.get_statistics()
        print(f"   文档总数: {stats.total_documents}")
        print(f"   嵌入维度: {stats.embedding_dimension}")
        print(f"   存储类型: {stats.storage_type}")
        print(f"   存储大小: {stats.storage_size_mb:.2f} MB")
        
        # 7. 搜索测试
        print("\n7. 搜索功能测试:")
        test_queries = [
            "材料分析测试",
            "性能检测方法",
            "质量控制标准",
            "环境测试条件"
        ]
        
        for query in test_queries:
            print(f"\n   查询: {query}")
            
            # 生成查询嵌入
            query_embedding = embedding_engine.embed_query(query)
            
            # 搜索相似文档
            results = vector_store.search(query_embedding, top_k=3)
            
            print(f"   找到 {len(results)} 个相关文档:")
            for i, result in enumerate(results):
                content_preview = result.document.content[:50] + "..." if len(result.document.content) > 50 else result.document.content
                print(f"     {i+1}. {content_preview} (相似度: {result.score:.4f})")
        
        # 8. 测试元数据过滤搜索（如果支持）
        print("\n8. 元数据过滤搜索测试:")
        try:
            # 获取第一个查询的嵌入
            query_embedding = embedding_engine.embed_query("材料测试")
            
            # 如果文档有元数据，尝试过滤搜索
            if documents and documents[0].metadata:
                # 获取可用的元数据键
                metadata_keys = list(documents[0].metadata.keys())
                print(f"   可用元数据字段: {metadata_keys}")
                
                # 尝试按第一个元数据字段过滤
                if metadata_keys:
                    first_key = metadata_keys[0]
                    first_value = documents[0].metadata[first_key]
                    
                    # 注意：ChromaDB和FAISS的过滤语法可能不同
                    if hasattr(vector_store.store, 'collection'):  # ChromaDB
                        results = vector_store.search(
                            query_embedding, 
                            top_k=5,
                            where={first_key: first_value}
                        )
                        print(f"   按 {first_key}={first_value} 过滤，找到 {len(results)} 个文档")
                    else:
                        print("   当前存储后端不支持元数据过滤搜索")
            else:
                print("   文档没有元数据，跳过过滤搜索测试")
                
        except Exception as e:
            print(f"   元数据过滤搜索失败: {e}")
        
        # 9. 保存索引
        print("\n9. 保存向量索引...")
        vector_store.save_index()
        print("   索引保存完成")
        
        print("\n✅ 向量存储模块演示完成！")
        print(f"\n💾 向量数据库已保存到: {VECTOR_STORE_PATH}")
        print("📝 您可以通过以下方式使用向量存储:")
        print("   1. 加载现有索引进行搜索")
        print("   2. 添加新文档到现有索引")
        print("   3. 更新或删除特定文档")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_operations():
    """测试向量存储的各种操作"""
    print("\n=== 向量存储操作测试 ===\n")
    
    try:
        # 创建测试文档
        from data_processing import Document
        
        test_documents = [
            Document("材料强度测试标准", {"type": "standard", "category": "strength"}, "test_001"),
            Document("化学成分分析方法", {"type": "method", "category": "chemistry"}, "test_002"),
            Document("物理性能检测流程", {"type": "process", "category": "physics"}, "test_003"),
        ]
        
        # 创建测试嵌入
        test_embeddings = [
            np.random.rand(384).astype(np.float32),  # 模拟嵌入向量
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        
        # 初始化向量存储
        vector_store = VectorStoreManager(
            store_type="faiss",  # 使用FAISS进行操作测试
            store_path="test_operations"
        )
        
        # 创建索引
        print("1. 创建测试索引...")
        vector_store.create_index(test_documents, test_embeddings)
        print("   索引创建完成")
        
        # 添加文档
        print("\n2. 添加新文档...")
        new_doc = Document("环境测试条件标准", {"type": "standard", "category": "environment"}, "test_004")
        new_embedding = np.random.rand(384).astype(np.float32)
        vector_store.add_documents([new_doc], [new_embedding])
        print("   文档添加完成")
        
        # 搜索测试
        print("\n3. 搜索测试...")
        query_embedding = np.random.rand(384).astype(np.float32)
        results = vector_store.search(query_embedding, top_k=2)
        print(f"   找到 {len(results)} 个结果")
        
        # 更新文档
        print("\n4. 更新文档...")
        updated_doc = Document("更新后的材料强度测试标准", {"type": "standard", "category": "strength", "version": "2.0"}, "test_001")
        updated_embedding = np.random.rand(384).astype(np.float32)
        success = vector_store.update_document("test_001", updated_doc, updated_embedding)
        print(f"   更新结果: {'成功' if success else '失败'}")
        
        # 删除文档
        print("\n5. 删除文档...")
        success = vector_store.delete_document("test_003")
        print(f"   删除结果: {'成功' if success else '失败'}")
        
        # 最终统计
        stats = vector_store.get_statistics()
        print(f"\n6. 最终统计: {stats.total_documents} 个文档")
        
        # 清理
        import shutil
        if Path("test_operations").exists():
            shutil.rmtree("test_operations")
            print("   测试文件已清理")
        
        print("\n✅ 操作测试完成！")
        
    except Exception as e:
        print(f"\n❌ 操作测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    success = demo_vector_store()
    
    if success:
        # 运行操作测试
        test_vector_store_operations() 