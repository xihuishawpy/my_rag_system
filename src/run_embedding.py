"""
运行向量化模块的脚本（使用Qwen API）
"""
import sys
import os
from pathlib import Path
import json

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding import EmbeddingEngine
from data_processing import DataProcessor
from config import EXCEL_FILE_PATH, EMBEDDING_MODEL

def check_api_key():
    """检查API密钥配置"""
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    qwen_key = os.getenv("QWEN_API_KEY")
    
    if not dashscope_key and not qwen_key:
        print("❌ 未找到API密钥配置!")
        print("\n请设置以下环境变量之一：")
        print("1. DASHSCOPE_API_KEY (推荐)")
        print("2. QWEN_API_KEY")
        print("\n设置方法：")
        print("Windows:")
        print("  set DASHSCOPE_API_KEY=your_api_key_here")
        print("Linux/Mac:")
        print("  export DASHSCOPE_API_KEY=your_api_key_here")
        print("\n或者在.env文件中设置：")
        print("  DASHSCOPE_API_KEY=your_api_key_here")
        print("\n如何获取API密钥：")
        print("1. 访问 https://dashscope.console.aliyun.com/")
        print("2. 登录阿里云账号")
        print("3. 创建API-KEY")
        return False
    
    if dashscope_key:
        print(f"✅ 找到DASHSCOPE_API_KEY: {dashscope_key[:8]}***")
    elif qwen_key:
        print(f"✅ 找到QWEN_API_KEY: {qwen_key[:8]}***")
    
    return True

def main():
    """主函数"""
    print("=== RAG系统向量化模块 (Qwen API) ===")
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    
    # 检查API密钥
    if not check_api_key():
        return
    
    try:
        # 1. 加载预处理的文档数据
        print("\n1. 加载文档数据...")
        
        # 检查是否存在预处理的JSON文件
        processed_file = Path("processed_documents.json")
        if processed_file.exists():
            print("   发现预处理文件，直接加载...")
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            from data_processing import Document
            documents = []
            for item in processed_data:
                doc = Document(
                    content=item["content"],
                    metadata=item["metadata"],
                    doc_id=item["doc_id"]
                )
                documents.append(doc)
        else:
            print("   未找到预处理文件，开始处理Excel...")
            processor = DataProcessor(EXCEL_FILE_PATH)
            documents = processor.process_excel_to_documents()
        
        print(f"   加载了 {len(documents)} 个文档")
        
        # 2. 初始化嵌入引擎
        print("\n2. 初始化Qwen嵌入引擎...")
        embedding_engine = EmbeddingEngine(
            model_name=EMBEDDING_MODEL,
            cache_enabled=True,
            batch_size=20  # 降低批处理大小以避免API限制
        )
        print("   引擎初始化完成")
        
        # 3. 选择处理模式
        print("\n请选择处理模式：")
        print("1. 完整处理（所有文档）")
        print("2. 快速测试（前10个文档）")
        print("3. 单个文档测试")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # 完整处理
            print(f"\n3. 开始处理所有 {len(documents)} 个文档...")
            target_documents = documents
            output_file = "all_embeddings.json"
            
        elif choice == "2":
            # 快速测试
            target_documents = documents[:10]
            output_file = "test_embeddings.json"
            print(f"\n3. 快速测试模式，处理前 {len(target_documents)} 个文档...")
            
        elif choice == "3":
            # 单个文档测试
            print("\n3. 单个文档测试...")
            test_doc = documents[0]
            print(f"   测试文档: {test_doc.content[:100]}...")
            
            embedding = embedding_engine.embed_text(test_doc.content)
            print(f"   嵌入向量维度: {embedding.shape}")
            print(f"   向量前5个值: {embedding[:5]}")
            print("   单个文档测试完成!")
            return
            
        else:
            print("无效选择，退出")
            return
        
        # 4. 执行嵌入处理
        print("\n   开始向量化处理...")
        embeddings = embedding_engine.embed_documents(target_documents)
        
        # 5. 显示处理结果
        print("\n4. 处理结果:")
        stats = embedding_engine.get_embedding_stats(embeddings)
        for key, value in stats.items():
            if key == "cache_stats" and isinstance(value, dict):
                print(f"   缓存统计:")
                for cache_key, cache_value in value.items():
                    if isinstance(cache_value, float):
                        print(f"     {cache_key}: {cache_value:.2f}")
                    else:
                        print(f"     {cache_key}: {cache_value}")
            elif isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # 6. 保存结果
        print(f"\n5. 保存嵌入结果到: {output_file}")
        embedding_engine.save_embeddings(embeddings, target_documents, output_file)
        
        # 7. 验证保存的文件
        print("\n6. 验证保存的文件...")
        loaded_embeddings, loaded_documents = embedding_engine.load_embeddings(output_file)
        print(f"   验证完成，加载了 {len(loaded_embeddings)} 个嵌入向量")
        
        # 8. 展示示例结果
        print("\n7. 示例结果:")
        for i, (doc, emb) in enumerate(zip(target_documents[:3], embeddings[:3])):
            print(f"\n   文档 {i+1}:")
            print(f"   ID: {doc.doc_id}")
            print(f"   内容: {doc.content[:100]}...")
            print(f"   嵌入维度: {emb.shape}")
            print(f"   嵌入范数: {np.linalg.norm(emb):.4f}")
        
        print(f"\n✅ 向量化处理完成！")
        print(f"   处理文档数: {len(target_documents)}")
        print(f"   输出文件: {Path(output_file).absolute()}")
        print(f"   嵌入维度: {embeddings[0].shape[0] if embeddings else 'N/A'}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np  # 添加这个import
    main()
