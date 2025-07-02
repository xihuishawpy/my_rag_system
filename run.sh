# 进入项目目录
cd /Volumes/mac/my_rag_system/src

# 第1步：数据预处理
echo "=== 第1步：数据预处理 ==="
python run_data_processing.py

# 第2步：向量化
echo "=== 第2步：向量化处理 ==="
python run_embedding.py

# 第3步：向量存储
echo "=== 第3步：向量存储 ==="
python run_vector_store.py

# 第4步：问答引擎
echo "=== 第4步：问答引擎 ==="
python run_qa_engine.py

