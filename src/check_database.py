#!/usr/bin/env python3
"""
数据库内容检查脚本
直接查看chroma数据库中的数据
"""

import chromadb
from pathlib import Path

def check_database():
    """检查数据库内容"""
    print("🗄️ 数据库内容检查")
    print("=" * 50)
    
    try:
        # 连接到chroma数据库
        db_path = Path("./vector_store")
        print(f"数据库路径: {db_path.absolute()}")
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        # 列出所有集合
        collections = client.list_collections()
        print(f"集合数量: {len(collections)}")
        
        for collection_info in collections:
            # Chroma v0.6.0+ 兼容性修复
            collection_name = collection_info if isinstance(collection_info, str) else collection_info.name
            print(f"\n📚 集合: {collection_name}")
            
            # 获取集合
            collection = client.get_collection(collection_name)
            
            # 获取集合统计信息
            count = collection.count()
            print(f"   文档数量: {count}")
            
            if count > 0:
                # 获取一些样本数据
                sample_data = collection.get(limit=3, include=['documents', 'metadatas', 'embeddings'])
                
                print(f"   样本数据:")
                for i, (doc_id, document, metadata) in enumerate(zip(
                    sample_data['ids'], 
                    sample_data['documents'], 
                    sample_data['metadatas']
                ), 1):
                    print(f"     {i}. ID: {doc_id}")
                    print(f"        内容: {document[:100] if document else 'None'}...")
                    print(f"        元数据: {metadata}")
                    
                    # 检查向量
                    if sample_data.get('embeddings') and sample_data['embeddings'][i-1]:
                        embedding = sample_data['embeddings'][i-1]
                        print(f"        向量维度: {len(embedding)}")
                        print(f"        向量预览: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
                    else:
                        print(f"        向量: 未找到")
                    print()
            else:
                print("   ❌ 集合为空")
                
        if not collections:
            print("❌ 未找到任何集合")
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database() 