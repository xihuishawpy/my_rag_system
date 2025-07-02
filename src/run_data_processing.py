"""
运行数据预处理的脚本
"""
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
from config import EXCEL_FILE_PATH

def main():
    """主函数"""
    print("=== RAG系统数据预处理模块 ===")
    print(f"Excel文件路径: {EXCEL_FILE_PATH}")
    
    # 检查文件是否存在
    if not EXCEL_FILE_PATH.exists():
        print(f"❌ Excel文件不存在: {EXCEL_FILE_PATH}")
        print("请确认文件路径是否正确")
        return
    
    try:
        # 创建数据处理器
        print("\n1. 初始化数据处理器...")
        processor = DataProcessor(EXCEL_FILE_PATH)
        
        # 处理Excel数据
        print("2. 处理Excel数据...")
        documents = processor.process_excel_to_documents()
        
        # 显示统计信息
        print("\n3. 统计信息:")
        stats = processor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # 显示前几个文档示例
        print("\n4. 文档示例:")
        for i, doc in enumerate(documents[:2]):
            print(f"\n   文档 {i+1} (ID: {doc.doc_id}):")
            print(f"   内容: {doc.content[:150]}...")
            print(f"   元数据字段: {len(doc.metadata)} 个")
        
        # 保存处理后的数据
        output_path = Path("processed_documents.json")
        print(f"\n5. 保存处理后的数据到: {output_path}")
        processor.save_processed_data(str(output_path))
        
        print("\n✅ 数据预处理完成！")
        print(f"   生成文档数量: {len(documents)}")
        print(f"   输出文件: {output_path.absolute()}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 