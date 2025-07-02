"""
数据预处理模块
负责读取Excel文件，清理数据，并转换为Document对象
"""
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """文档对象，包含文本内容和元数据"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

class DataProcessor:
    """数据预处理"""
    def __init__(self, excel_path: str):
        """
        初始化数据处理器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = Path(excel_path)
        self.documents: List[Document] = []
        
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
    
    def load_excel_data(self) -> pd.DataFrame:
        """
        加载Excel数据
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"成功加载Excel文件，共 {len(df)} 行数据")
            logger.info(f"列名: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"加载Excel文件失败: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        # 移除特殊字符（保留中文、英文、数字、常用标点）
        # text = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()（）【】""''、。，；：！？]', '', text)
    
        return text
    
    def create_document_from_row(self, row: pd.Series, row_index: int) -> Optional[Document]:
        """
        将DataFrame行转换为Document对象
        
        Args:
            row: pandas Series对象，表示一行数据
            row_index: 行索引
            
        Returns:
            Document: 文档对象，如果内容为空则返回None
        """
        # 将所有列的内容，合并为一个完整的语义单元
        content_parts = []
        
        for column_name, value in row.items():
            if pd.notna(value) and str(value).strip():
                cleaned_value = self.clean_text(str(value))
                if cleaned_value:
                    content_parts.append(f"{column_name}: {cleaned_value}")
        
        if not content_parts:
            logger.warning(f"第 {row_index + 1} 行数据为空，跳过")
            return None
        
        # 合并行内容
        content = " | ".join(content_parts)
        
        # 创建元数据
        metadata = {
                "source": str(self.excel_path),
                "row_index": row_index,
                "created_at": datetime.now().isoformat(),
                "columns": ",".join(list(row.index)), 
                "content_length": len(content),
                "column_count": len(row.index)  
            }
        
        # 添加原始数据到元数据
        for column_name, value in row.items():
            if pd.notna(value):
                metadata[f"raw_{column_name}"] = str(value)
        
        doc_id = f"doc_{row_index:04d}"
        
        return Document(
            content=content,
            metadata=metadata,
            doc_id=doc_id
        )
    
    def process_excel_to_documents(self) -> List[Document]:
        """
        处理Excel文件，转换为Document列表
        
        Returns:
            List[Document]: 文档列表
        """
        logger.info("开始处理Excel数据...")
        
        df = self.load_excel_data()
        
        documents = []
        skipped_count = 0
        
        for index, row in df.iterrows():
            document = self.create_document_from_row(row, index)
            if document:
                documents.append(document)
            else:
                skipped_count += 1
        
        logger.info(f"处理完成，生成 {len(documents)} 个文档，跳过 {skipped_count} 个空行")
        
        self.documents = documents
        return documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.documents:
            return {"total_documents": 0}
        
        content_lengths = [len(doc.content) for doc in self.documents]
        
        stats = {
            "total_documents": len(self.documents),
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "min_content_length": min(content_lengths),
            "max_content_length": max(content_lengths),
            "total_characters": sum(content_lengths)
        }
        
        return stats
    
    def save_processed_data(self, output_path: str) -> None:
        """
        保存处理后的数据到文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        data_to_save = []
        for doc in self.documents:
            data_to_save.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存处理后的数据到: {output_path}")

def main():
    from config import EXCEL_FILE_PATH
    
    try:
        # 创建数据处理器
        processor = DataProcessor(EXCEL_FILE_PATH)
        
        # 处理数据
        documents = processor.process_excel_to_documents()
        
        # 显示统计信息
        stats = processor.get_statistics()
        print("\n=== 数据处理统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # 显示前几个文档示例
        print("\n=== 文档示例 ===")
        for i, doc in enumerate(documents[:3]):
            print(f"\n文档 {i+1} (ID: {doc.doc_id}):")
            print(f"内容: {doc.content[:200]}...")
            print(f"元数据: {list(doc.metadata.keys())}")
        
        # 保存处理后的数据
        output_path = "processed_documents.json"
        processor.save_processed_data(output_path)
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise

if __name__ == "__main__":
    main() 