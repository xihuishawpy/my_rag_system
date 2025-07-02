#!/usr/bin/env python3
"""
RAG系统Web界面
基于Streamlit的用户友好图形化查询界面
"""

import streamlit as st
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding import EmbeddingEngine
from vector_store import VectorStoreManager
from retrieval import RetrievalEngine
from reranking import RerankingEngine
from qa_engine import QAEngine, QAResult
from config import QWEN_API_KEY, VECTOR_STORE_TYPE, VECTOR_STORE_PATH

# 页面配置
st.set_page_config(
    page_title="RAG知识库问答2.0",
    page_icon="😎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border-left: 4px solid #1f77b4;
}

.user-message {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}

.assistant-message {
    background-color: #f3e5f5;
    border-left-color: #9c27b0;
}

.citation-box {
    background-color: #fff3e0;
    padding: 0.8rem;
    border-radius: 0.3rem;
    margin: 0.3rem 0;
    border-left: 3px solid #ff9800;
    font-size: 0.9rem;
}

.status-success {
    color: #4caf50;
    font-weight: bold;
}

.status-warning {
    color: #ff9800;
    font-weight: bold;
}

.status-error {
    color: #f44336;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class RAGWebInterface:
    """RAG系统Web界面主类"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """初始化会话状态"""
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'selected_vector_db' not in st.session_state:
            st.session_state.selected_vector_db = "chroma"  # 默认使用ChromaDB
        if 'last_vector_db' not in st.session_state:
            st.session_state.last_vector_db = None  # 用于检测数据库类型变化
    
    def check_environment(self):
        """检查系统环境"""
        try:
            # 检查API密钥
            dashscope_key = os.getenv("DASHSCOPE_API_KEY")
            config_key = QWEN_API_KEY
            has_api_key = bool(dashscope_key or config_key)
            
            # 检查向量数据库
            vector_store_path = Path("./vector_store")
            has_vector_store = vector_store_path.exists()
            
            if not has_api_key:
                return False, "⚠️ API密钥未配置，请设置DASHSCOPE_API_KEY环境变量"
            
            if not has_vector_store:
                return False, "⚠️ 向量数据库不存在，请先运行数据处理脚本"
            
            return True, "✅ 环境检查通过"
            
        except Exception as e:
            return False, f"❌ 环境检查失败: {str(e)}"
    
    def initialize_rag_system(self):
        """初始化RAG系统"""
        try:
            selected_db = st.session_state.selected_vector_db
            with st.spinner(f"🚀 正在初始化RAG系统 (使用{selected_db.upper()})..."):
                # 初始化各个组件
                embedding_engine = EmbeddingEngine(cache_enabled=True)
                vector_store = VectorStoreManager(selected_db, str(VECTOR_STORE_PATH))
                retrieval_engine = RetrievalEngine(vector_store, embedding_engine)
                reranking_engine = RerankingEngine(cache_enabled=True)
                
                qa_engine = QAEngine(
                    retrieval_engine=retrieval_engine,
                    reranking_engine=reranking_engine,
                    cache_enabled=True,
                    max_context_length=3000,
                    temperature=0.7
                )
                
                st.session_state.rag_system = qa_engine
                st.session_state.system_ready = True
                
                # 获取系统统计信息
                self.update_system_stats()
                
                return True
                
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {str(e)}")
            return False
    
    def update_system_stats(self):
        """更新系统统计信息"""
        if st.session_state.rag_system:
            try:
                qa_stats = st.session_state.rag_system.get_statistics()
                retrieval_stats = st.session_state.rag_system.retrieval_engine.get_statistics()
                rerank_stats = st.session_state.rag_system.reranking_engine.get_statistics()
                vector_stats = st.session_state.rag_system.retrieval_engine.vector_store.get_statistics()
                
                st.session_state.system_stats = {
                    'qa_stats': qa_stats,
                    'retrieval_stats': retrieval_stats,
                    'rerank_stats': rerank_stats,
                    'vector_stats': vector_stats
                }
            except Exception as e:
                st.warning(f"获取系统统计信息失败: {str(e)}")
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.markdown("## 🎛️ 系统控制")
        
        # 环境状态
        env_ok, env_msg = self.check_environment()
        if env_ok:
            st.sidebar.markdown(f'<div class="status-success">{env_msg}</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f'<div class="status-error">{env_msg}</div>', unsafe_allow_html=True)
        
        # 向量数据库选择
        st.sidebar.markdown("### 🗃️ 向量数据库")
        selected_db = st.sidebar.selectbox(
            "选择向量数据库类型",
            ["faiss", "chroma"],
            index=0 if st.session_state.selected_vector_db == "faiss" else 1,
            help="FAISS: 高性能，支持多种索引类型\nChromaDB: 智能向量数据库，内置HNSW索引",
            key="vector_db_selector"
        )
        
        # 显示数据库特性说明
        if selected_db == "faiss":
            st.sidebar.markdown("""
            <div style="background-color: #e3f2fd; padding: 0.5rem; border-radius: 0.3rem; font-size: 0.8rem;">
                🚀 <strong>FAISS特性:</strong><br>
                • 高性能向量检索<br>
                • 支持多种索引类型<br>
                • 适合大规模数据
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style="background-color: #f3e5f5; padding: 0.5rem; border-radius: 0.3rem; font-size: 0.8rem;">
                🏗️ <strong>ChromaDB特性:</strong><br>
                • 智能向量数据库<br>
                • 内置HNSW索引<br>
                • 易于使用和管理
            </div>
            """, unsafe_allow_html=True)
        
        # 检测数据库类型变化
        if selected_db != st.session_state.selected_vector_db:
            st.session_state.selected_vector_db = selected_db
            # 如果系统已经初始化，需要重新初始化
            if st.session_state.system_ready:
                st.session_state.system_ready = False
                st.session_state.rag_system = None
                st.session_state.system_stats = {}
                st.sidebar.info(f"已切换到 {selected_db.upper()}，请重新初始化系统")
        
        # 系统初始化
        if not st.session_state.system_ready:
            if st.sidebar.button("🚀 初始化系统", type="primary"):
                if env_ok:
                    st.session_state.system_ready = self.initialize_rag_system()
                else:
                    st.sidebar.error("请先解决环境问题")
        else:
            st.sidebar.markdown('<div class="status-success">✅ 系统已就绪</div>', unsafe_allow_html=True)
        
        # 系统统计信息
        if st.session_state.system_ready and st.session_state.system_stats:
            st.sidebar.markdown("### 📊 系统状态")
            
            # 显示当前向量数据库类型
            vector_db_type = st.session_state.selected_vector_db.upper()
            vector_db_icon = "🚀" if vector_db_type == "FAISS" else "🏗️"
            st.sidebar.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 0.5rem; border-radius: 0.3rem; margin-bottom: 1rem;">
                <strong>{vector_db_icon} 当前使用:</strong> <span style="color: #2e7d32; font-weight: bold;">{vector_db_type}</span>
            </div>
            """, unsafe_allow_html=True)
            
            stats = st.session_state.system_stats
            if 'vector_stats' in stats:
                st.sidebar.metric("文档总数", stats['vector_stats'].total_documents)
                st.sidebar.metric("向量维度", stats['vector_stats'].embedding_dimension)
                st.sidebar.metric("存储大小", f"{stats['vector_stats'].storage_size_mb:.1f} MB")
            
            if 'qa_stats' in stats:
                st.sidebar.metric("总问题数", stats['qa_stats'].total_questions)
                st.sidebar.metric("平均响应时间", f"{stats['qa_stats'].avg_response_time:.3f}s")
        
        # 搜索设置
        st.sidebar.markdown("### ⚙️ 搜索设置")
        search_method = st.sidebar.selectbox(
            "检索策略",
            ["hybrid", "semantic", "smart", "filtered"],
            index=0,
            help="选择不同的检索策略"
        )
        
        context_limit = st.sidebar.slider(
            "检索文档数量",
            min_value=1,
            max_value=10,
            value=5,
            help="控制检索和显示的文档数量，设置多少就会显示多少个引用"
        )
        
        temperature = st.sidebar.slider(
            "创造性温度",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="控制回答的创造性，值越高越有创意"
        )
        
        return search_method, context_limit, temperature
    
    def render_main_header(self):
        """渲染主标题"""
        st.markdown('<div class="main-header">😎 RAG知识库问答</div>', unsafe_allow_html=True)
        
        # 显示当前向量数据库信息
        vector_db_type = st.session_state.selected_vector_db.upper()
        vector_db_icon = "🚀" if vector_db_type == "FAISS" else "🏗️"
        if vector_db_type == "FAISS":
            vector_db_desc = "高性能向量检索 - 支持多种索引类型"
        else:  # ChromaDB
            vector_db_desc = "智能向量数据库 - 内置HNSW索引"
        
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <div style="background: linear-gradient(90deg, #e3f2fd, #f3e5f5); padding: 0.8rem; border-radius: 0.5rem; display: inline-block;">
                <span style="font-size: 1.1rem; font-weight: bold; color: #1565c0;">
                    {vector_db_icon} 当前使用: {vector_db_type} ({vector_db_desc})
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_chat_interface(self, search_method, context_limit, temperature):
        """渲染聊天界面"""
        st.markdown('<div class="sub-header">💬 智能问答</div>', unsafe_allow_html=True)
        
        # 聊天历史显示
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user-message">
                    <strong>👤 您:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message assistant-message">
                    <strong>🤖 助手:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
                
                # 显示引用信息
                if "citations" in message and message["citations"]:
                    with st.expander(f"📚 查看引用 ({len(message['citations'])}个文档)"):
                        for j, citation in enumerate(message["citations"], 1):
                            # 完整显示引用内容
                            citation_content = citation.content
                            
                            # 统一显示格式，避免嵌套expander
                            st.markdown(f'''
                            <div class="citation-box">
                                <strong>引用 {j}:</strong> (相关性: {citation.relevance_score:.3f})<br>
                                <div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #f8f9fa; border-radius: 0.3rem; white-space: pre-wrap;">{citation_content}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # 添加分隔线（除了最后一个引用）
                            if j < len(message["citations"]):
                                st.markdown("---")
                
                # 显示性能指标
                if "metrics" in message:
                    metrics = message["metrics"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总耗时", f"{metrics['total_time']:.3f}s")
                    with col2:
                        st.metric("置信度", f"{metrics['confidence']:.3f}")
                    with col3:
                        st.metric("检索时间", f"{metrics['retrieval_time']:.3f}s")
                    with col4:
                        st.metric("生成时间", f"{metrics['generation_time']:.3f}s")
        
        # 输入区域
        st.markdown("### 💭 提问")
        question = st.text_input(
            "请输入您的问题:",
            placeholder="例如: 金属材料拉伸强度如何测试？",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ask_button = st.button("🔍 提问", type="primary")
        with col2:
            clear_button = st.button("🗑️ 清空历史")
        with col3:
            refresh_button = st.button("📊 刷新统计")
        
        # 处理按钮点击
        if clear_button:
            st.session_state.chat_history = []
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_conversation_history()
            st.rerun()
        
        if refresh_button:
            self.update_system_stats()
            st.rerun()
        
        if ask_button and question and st.session_state.system_ready:
            self.process_question(question, search_method, context_limit, temperature)
    
    def process_question(self, question, search_method, context_limit, temperature):
        """处理用户问题"""
        try:
            # 添加用户消息到历史
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # 显示处理状态
            with st.spinner("🤔 正在思考中..."):
                start_time = time.time()
                
                # 调用RAG系统
                result = st.session_state.rag_system.answer_question(
                    question=question,
                    context_limit=context_limit,
                    search_method=search_method
                )
                
                processing_time = time.time() - start_time
            
            # 添加助手回复到历史
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.answer,
                "citations": result.citations,
                "metrics": {
                    "total_time": result.total_time,
                    "confidence": result.confidence_score,
                    "retrieval_time": result.retrieval_time,
                    "generation_time": result.generation_time,
                    "processing_time": processing_time
                }
            })
            
            # 更新统计信息
            self.update_system_stats()
            
            # 刷新页面显示
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 处理问题时出错: {str(e)}")
    
    def render_search_interface(self, search_method, context_limit):
        """渲染搜索界面"""
        st.markdown('<div class="sub-header">🔍 文档搜索</div>', unsafe_allow_html=True)
        
        search_query = st.text_input(
            "搜索查询:",
            placeholder="输入关键词搜索相关文档",
            key="search_input"
        )
        
        if st.button("🔎 搜索文档"):
            if search_query and st.session_state.system_ready:
                self.perform_search(search_query, search_method, context_limit)
        
        # 显示搜索结果
        if st.session_state.search_results:
            st.markdown("### 📄 搜索结果")
            
            for i, result in enumerate(st.session_state.search_results, 1):
                with st.expander(f"文档 {i} (相关性: {result.score:.3f})"):
                    st.write(f"**内容:** {result.document.content}")
                    st.write(f"**检索类型:** {result.retrieval_type}")
                    
                    if result.document.metadata:
                        st.write("**元数据:**")
                        metadata_df = pd.DataFrame([result.document.metadata])
                        st.dataframe(metadata_df, use_container_width=True)
    
    def perform_search(self, query, search_method, context_limit):
        """执行文档搜索"""
        try:
            with st.spinner("🔍 正在搜索..."):
                retrieval_engine = st.session_state.rag_system.retrieval_engine
                
                if search_method == "semantic":
                    results = retrieval_engine.semantic_search(query, top_k=context_limit)
                elif search_method == "hybrid":
                    results = retrieval_engine.hybrid_search(query, top_k=context_limit)
                elif search_method == "smart":
                    results = retrieval_engine.smart_search(query, top_k=context_limit)
                else:
                    results = retrieval_engine.semantic_search(query, top_k=context_limit)
                
                st.session_state.search_results = results
                
        except Exception as e:
            st.error(f"❌ 搜索失败: {str(e)}")
    
    def render_analytics_dashboard(self):
        """渲染分析仪表板"""
        st.markdown('<div class="sub-header">📊 系统分析</div>', unsafe_allow_html=True)
        
        if not st.session_state.system_stats:
            st.info("暂无统计数据，请先使用系统进行问答")
            return
        
        stats = st.session_state.system_stats
        
        # 向量数据库详细信息
        st.markdown("#### 🚀 向量数据库详情")
        
        # 数据库基本信息
        vector_col1, vector_col2, vector_col3, vector_col4 = st.columns(4)
        
        with vector_col1:
            vector_db_type = st.session_state.selected_vector_db.upper()
            st.metric(
                "数据库类型",
                vector_db_type,
                help="当前使用的向量数据库类型"
            )
        
        with vector_col2:
            if 'vector_stats' in stats:
                # 根据数据库类型显示不同的信息
                if st.session_state.selected_vector_db.lower() == "faiss":
                    st.metric(
                        "索引类型",
                        stats['vector_stats'].index_type,
                        help="FAISS向量索引的具体类型"
                    )
                else:  # ChromaDB
                    collection_name = stats['vector_stats'].metadata.get('collection_name', 'documents')
                    st.metric(
                        "集合名称",
                        collection_name,
                        help="ChromaDB集合名称"
                    )
        
        with vector_col3:
            if 'vector_stats' in stats:
                st.metric(
                    "存储大小",
                    f"{stats['vector_stats'].storage_size_mb:.1f} MB",
                    help="向量数据库占用的存储空间"
                )
        
        with vector_col4:
            if 'vector_stats' in stats:
                efficiency = stats['vector_stats'].total_documents / max(stats['vector_stats'].storage_size_mb, 1)
                st.metric(
                    "存储效率",
                    f"{efficiency:.1f} docs/MB",
                    help="每MB存储的文档数量"
                )
        
        st.markdown("---")
        
        # 性能指标
        st.markdown("#### 📈 系统性能指标")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'qa_stats' in stats:
                st.metric(
                    "总问题数",
                    stats['qa_stats'].total_questions,
                    help="系统处理的问题总数"
                )
        
        with col2:
            if 'qa_stats' in stats:
                st.metric(
                    "平均响应时间",
                    f"{stats['qa_stats'].avg_response_time:.3f}s",
                    help="问答的平均响应时间"
                )
        
        with col3:
            if 'qa_stats' in stats:
                st.metric(
                    "平均置信度",
                    f"{stats['qa_stats'].avg_confidence_score:.3f}",
                    help="答案的平均置信度"
                )
        
        with col4:
            if 'rerank_stats' in stats:
                st.metric(
                    "重排序请求",
                    stats['rerank_stats'].total_requests,
                    help="重排序模块处理的请求数"
                )
        
        # 性能图表
        if st.session_state.chat_history:
            self.render_performance_charts()
    
    def render_performance_charts(self):
        """渲染性能图表"""
        # 提取聊天历史中的性能数据
        metrics_data = []
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "assistant" and "metrics" in message:
                metrics = message["metrics"]
                metrics_data.append({
                    "问题序号": i // 2 + 1,
                    "总耗时": metrics["total_time"],
                    "检索时间": metrics["retrieval_time"],
                    "生成时间": metrics["generation_time"],
                    "置信度": metrics["confidence"]
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # 响应时间趋势
            st.markdown("#### ⏱️ 响应时间分析")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_time = px.line(
                    df, 
                    x="问题序号", 
                    y=["总耗时", "检索时间", "生成时间"],
                    title="响应时间趋势",
                    labels={"value": "时间(秒)", "variable": "指标"}
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                fig_confidence = px.bar(
                    df,
                    x="问题序号",
                    y="置信度",
                    title="回答置信度",
                    labels={"置信度": "置信度分数"}
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
    
    def run(self):
        """运行Web界面"""
        # 渲染侧边栏
        search_method, context_limit, temperature = self.render_sidebar()
        
        # 渲染主界面
        self.render_main_header()
        
        # 选项卡
        tab1, tab2, tab3 = st.tabs(["💬 智能问答", "🔍 文档搜索", "📊 系统分析"])
        
        with tab1:
            self.render_chat_interface(search_method, context_limit, temperature)
        
        with tab2:
            self.render_search_interface(search_method, context_limit)
        
        with tab3:
            self.render_analytics_dashboard()

def main():
    """主函数"""
    interface = RAGWebInterface()
    interface.run()

if __name__ == "__main__":
    main() 