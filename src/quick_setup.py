#!/usr/bin/env python3
"""
快速设置脚本 - 用少量数据快速测试RAG系统
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStoreManager
from embedding import EmbeddingEngine
from data_processing import Document
from config import VECTOR_STORE_TYPE, VECTOR_STORE_PATH

def quick_setup():
    """快速设置测试数据"""
    print("🚀 RAG系统快速设置")
    print("=" * 50)
    
    try:
        # 创建测试文档
        print("1. 创建测试文档...")
        test_documents = [
            Document(
                content="金属材料拉伸强度测试按照GB/T 228.1-2010标准执行，包括屈服强度、抗拉强度等指标",
                metadata={"样品类别": "金属材料", "测试项目": "拉伸强度", "测试标准": "GB/T 228.1-2010", "价格": "300"},
                doc_id="doc_001"
            ),
            Document(
                content="塑料制品成分分析采用红外光谱法和热重分析法，可检测聚合物类型和添加剂含量",
                metadata={"样品类别": "塑料制品", "测试项目": "成分分析", "测试标准": "GB/T 1040-2018", "价格": "450"},
                doc_id="doc_002"
            ),
            Document(
                content="金属疲劳试验测试材料在循环载荷下的疲劳寿命，试验周期通常为5-7个工作日",
                metadata={"样品类别": "金属材料", "测试项目": "疲劳试验", "测试标准": "GB/T 3075-2008", "价格": "800"},
                doc_id="doc_003"
            ),
            Document(
                content="环境检测服务包括重金属污染、有机污染物检测，检测项目覆盖土壤、水质、大气",
                metadata={"样品类别": "环境样品", "测试项目": "污染物检测", "测试标准": "GB 15618-2018", "价格": "200"},
                doc_id="doc_004"
            ),
            Document(
                content="实验室具有CMA和CNAS双重认证资质，可提供具有法律效力的检测报告",
                metadata={"服务类型": "认证信息", "资质": "CMA+CNAS", "报告": "具有法律效力", "价格": "0"},
                doc_id="doc_005"
            ),
            Document(
                content="建筑材料检测包括水泥、混凝土、钢筋等强度测试，普通周期3-5天，加急1-2天",
                metadata={"样品类别": "建筑材料", "测试项目": "强度测试", "普通周期": "3-5天", "价格": "150"},
                doc_id="doc_006"
            ),
            Document(
                content="纺织品检测项目包括甲醛含量、色牢度、纤维成分分析，符合国家纺织标准",
                metadata={"样品类别": "纺织品", "测试项目": "质量检测", "测试标准": "GB 18401-2010", "价格": "250"},
                doc_id="doc_007"
            ),
            Document(
                content="食品安全检测涵盖农药残留、重金属、微生物指标，检测周期2-3个工作日",
                metadata={"样品类别": "食品", "测试项目": "安全检测", "测试周期": "2-3天", "价格": "180"},
                doc_id="doc_008"
            ),
        ]
        
        print(f"   创建了 {len(test_documents)} 个测试文档")
        
        # 初始化嵌入引擎
        print("\n2. 初始化嵌入引擎...")
        embedding_engine = EmbeddingEngine(cache_enabled=True)
        
        # 生成嵌入
        print("\n3. 生成文档嵌入...")
        embeddings = embedding_engine.embed_documents(test_documents)
        print(f"   生成 {len(embeddings)} 个嵌入向量，维度: {len(embeddings[0])}")
        
        # 初始化向量存储（清空现有数据）
        print("\n4. 初始化向量存储（覆盖模式）...")
        vector_store = VectorStoreManager(VECTOR_STORE_TYPE, str(VECTOR_STORE_PATH))
        
        # 删除现有集合（如果存在）
        try:
            # 尝试删除现有数据（FAISS会重新创建索引）
            if hasattr(vector_store.store, 'delete_collection'):
                vector_store.store.delete_collection("documents")
                print("   已删除现有集合")
            else:
                print("   FAISS将重新创建索引")
        except:
            print("   无现有集合需要删除")
        
        # 创建新索引
        print("\n5. 创建向量索引...")
        vector_store.create_index(test_documents, embeddings)
        print("   索引创建完成")
        
        # 验证数据
        print("\n6. 验证数据...")
        stats = vector_store.get_statistics()
        print(f"   文档总数: {stats.total_documents}")
        print(f"   嵌入维度: {stats.embedding_dimension}")
        
        # 测试搜索
        print("\n7. 测试搜索功能...")
        test_queries = [
            "金属材料强度测试",
            "塑料成分分析",
            "实验室认证",
            "检测周期",
            "价格费用"
        ]
        
        for query in test_queries:
            print(f"\n   查询: '{query}'")
            query_embedding = embedding_engine.embed_query(query)
            results = vector_store.search(query_embedding, top_k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    score = result.score  # 使用相似度分数
                    print(f"     {i}. 相似度: {score:.3f}")
                    print(f"        内容: {result.document.content[:80]}...")
            else:
                print("     ❌ 无搜索结果")
        
        print("\n✅ 快速设置完成！")
        print("   向量数据库已就绪，可以进行问答测试")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 快速设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_setup() 