import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

logger = logging.getLogger('RAGHelper')

class RAGHelper:
    """RAG助手：将搜索结果转换为知识向量并进行检索增强生成"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dim: int = 384):
        """
        初始化RAG助手
        
        Args:
            model_name: 句子嵌入模型名称
            vector_dim: 向量维度
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        # 创建向量存储目录
        self.vector_store_dir = "vector_store"
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # 初始化嵌入模型
        self._load_embedding_model()
        
        logger.info(f"RAG助手初始化完成，使用模型: {model_name}")
    
    def _load_embedding_model(self):
        """加载句子嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _create_document_id(self, content: str, metadata: Dict) -> str:
        """创建文档ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{metadata.get('source', 'unknown')}_{content_hash[:8]}"
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """将长文本分块"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def add_search_results(self, search_results: List[Dict[str, Any]], 
                          search_term: str = None) -> int:
        """
        将搜索结果添加到知识库
        
        Args:
            search_results: 搜索结果列表
            search_term: 搜索关键词
            
        Returns:
            添加的文档数量
        """
        added_count = 0
        
        for result in search_results:
            # 提取文本内容
            title = result.get('title', '')
            description = result.get('description', '')
            url = result.get('url', '')
            
            # 合并文本内容
            content = f"标题: {title}\n摘要: {description}"
            
            # 创建元数据
            metadata = {
                'title': title,
                'url': url,
                'source': 'search_result',
                'search_term': search_term,
                'timestamp': datetime.now().isoformat()
            }
            
            # 检查是否已存在
            doc_id = self._create_document_id(content, metadata)
            existing_ids = [doc.get('id') for doc in self.document_metadata]
            
            if doc_id in existing_ids:
                logger.debug(f"文档已存在，跳过: {title[:50]}...")
                continue
            
            # 文本分块
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['id'] = f"{doc_id}_chunk_{i}"
                
                # 生成嵌入向量
                try:
                    embedding = self.embedding_model.encode([chunk])[0]
                    
                    # 添加到文档列表
                    self.documents.append(chunk)
                    self.document_metadata.append(chunk_metadata)
                    
                    # 添加到向量索引
                    if self.index is None:
                        self.index = faiss.IndexFlatIP(self.vector_dim)
                    
                    # 重塑向量并添加到索引
                    embedding_reshaped = embedding.reshape(1, -1).astype('float32')
                    self.index.add(embedding_reshaped)
                    
                    added_count += 1
                    
                except Exception as e:
                    logger.error(f"处理文档块失败: {e}")
                    continue
        
        logger.info(f"成功添加 {added_count} 个文档块到知识库")
        return added_count
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回的相似文档数量
            
        Returns:
            相似文档列表，包含内容和元数据
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("知识库为空，无法搜索")
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding_reshaped = query_embedding.reshape(1, -1).astype('float32')
            
            # 搜索相似向量
            scores, indices = self.index.search(query_embedding_reshaped, min(top_k, len(self.documents)))
            
            # 构建结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    result = {
                        'content': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    }
                    results.append(result)
            
            logger.info(f"搜索完成，找到 {len(results)} 个相似文档")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_context_for_llm(self, query: str, max_tokens: int = 4000) -> str:
        """
        为LLM生成上下文信息
        
        Args:
            query: 查询文本
            max_tokens: 最大token数
            
        Returns:
            格式化的上下文字符串
        """
        similar_docs = self.search_similar(query, top_k=10)
        
        if not similar_docs:
            return "暂无相关背景信息。"
        
        context_parts = []
        current_tokens = 0
        estimated_tokens_per_char = 0.25  # 粗略估计
        
        for doc in similar_docs:
            content = doc['content']
            metadata = doc['metadata']
            similarity = doc['similarity_score']
            
            # 估算token数
            estimated_tokens = len(content) * estimated_tokens_per_char
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            # 格式化文档信息
            doc_info = f"""
【相关文档 {doc['rank']}】相似度: {similarity:.3f}
来源: {metadata.get('title', '未知标题')}
URL: {metadata.get('url', '无链接')}
搜索关键词: {metadata.get('search_term', '未知')}
内容:
{content}
"""
            context_parts.append(doc_info)
            current_tokens += estimated_tokens
        
        if context_parts:
            return "\n".join(context_parts)
        else:
            return "暂无相关背景信息。"
    
    def save_vector_store(self, filename: str = None) -> str:
        """保存向量存储"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"vector_store_{timestamp}.pkl"
        
        filepath = os.path.join(self.vector_store_dir, filename)
        
        try:
            store_data = {
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'index': self.index,
                'model_name': self.model_name,
                'vector_dim': self.vector_dim,
                'created_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(store_data, f)
            
            logger.info(f"向量存储已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
            return ""
    
    def load_vector_store(self, filepath: str) -> bool:
        """加载向量存储"""
        try:
            with open(filepath, 'rb') as f:
                store_data = pickle.load(f)
            
            self.documents = store_data['documents']
            self.document_metadata = store_data['document_metadata']
            self.index = store_data['index']
            self.model_name = store_data['model_name']
            self.vector_dim = store_data['vector_dim']
            
            logger.info(f"向量存储已加载: {filepath}")
            logger.info(f"知识库统计: {len(self.documents)} 个文档块")
            return True
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if not self.documents:
            return {"total_documents": 0, "total_chunks": 0}
        
        # 统计搜索关键词
        search_terms = {}
        for metadata in self.document_metadata:
            term = metadata.get('search_term', 'unknown')
            search_terms[term] = search_terms.get(term, 0) + 1
        
        return {
            "total_documents": len(set([m.get('id', '').split('_chunk_')[0] for m in self.document_metadata])),
            "total_chunks": len(self.documents),
            "search_terms": search_terms,
            "model_name": self.model_name,
            "vector_dim": self.vector_dim
        } 