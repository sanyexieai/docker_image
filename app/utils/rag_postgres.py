import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from pgvector import Vector
import yaml

logger = logging.getLogger('RAGPostgres')

class RAGPostgresHelper:
    """基于PostgreSQL和pgvector的RAG助手"""
    
    def __init__(self, 
                 db_config: Dict[str, str] = None,
                 rag_config: Dict[str, Any] = None):
        """
        初始化PostgreSQL RAG助手
        
        Args:
            db_config: 数据库配置字典
            rag_config: RAG配置字典，包含模型名称、向量维度、设备等参数
        """
        # 从配置文件加载RAG配置
        if rag_config is None:
            from config.database_config import db_config as global_db_config
            rag_config = global_db_config.get_rag_config()
        print('RAGPostgresHelper收到的rag_config:', rag_config)  # 调试用
        self.model_name = rag_config.get('model_name', 'Qwen/Qwen3-Embedding-4B')
        self.vector_dim = rag_config.get('vector_dim', 1536)
        self.device = rag_config.get('device', 'cuda')
        self.chunk_size = rag_config.get('chunk_size', 500)
        self.chunk_overlap = rag_config.get('chunk_overlap', 50)
        self.max_tokens = rag_config.get('max_tokens', 4000)
        self.top_k = rag_config.get('top_k', 10)
        self.embedding_model = None
        
        # 数据库配置
        if db_config is None:
            from config.database_config import db_config as global_db_config
            self.db_config = global_db_config.get_postgres_config()
        else:
            self.db_config = db_config
        
        # 初始化嵌入模型
        self._load_embedding_model()
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"PostgreSQL RAG助手初始化完成，使用模型: {self.model_name}，设备: {self.device}")
    
    def _load_embedding_model(self):
        """加载句子嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name} 到设备: {self.device}")
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"嵌入模型加载成功，使用设备: {self.embedding_model.device}")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def _get_connection(self):
        """获取数据库连接"""
        try:
            conn = psycopg2.connect(**self.db_config)
            register_vector(conn)
            return conn
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 启用pgvector扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 检查表是否存在
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # 检查现有表的向量维度
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'documents' 
                    AND column_name = 'embedding';
                """)
                result = cursor.fetchone()
                
                if result and result[1]:  # 确保结果存在且data_type不为空
                    try:
                        # 解析向量维度，格式可能是 vector(1536) 或 vector
                        data_type = result[1]
                        if '(' in data_type and ')' in data_type:
                            current_dim = int(data_type.split('(')[1].split(')')[0])
                            if current_dim != self.vector_dim:
                                logger.warning(f"检测到向量维度不匹配: 当前 {current_dim}, 配置 {self.vector_dim}")
                                logger.info("删除旧表并重新创建...")
                                
                                # 删除旧表
                                cursor.execute("DROP TABLE IF EXISTS documents CASCADE;")
                                conn.commit()
                                table_exists = False
                        else:
                            logger.warning("无法解析向量维度，但保留现有数据，仅检查维度兼容性...")
                            # 不删除表，只检查是否有数据
                            cursor.execute("SELECT COUNT(*) FROM documents")
                            count = cursor.fetchone()[0]
                            if count > 0:
                                logger.info(f"现有数据库包含 {count} 条记录，保留现有数据")
                                table_exists = True
                            else:
                                logger.info("数据库为空，重新创建表结构...")
                                cursor.execute("DROP TABLE IF EXISTS documents CASCADE;")
                                conn.commit()
                                table_exists = False
                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析向量维度失败: {e}，但保留现有数据...")
                        # 不删除表，只检查是否有数据
                        cursor.execute("SELECT COUNT(*) FROM documents")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            logger.info(f"现有数据库包含 {count} 条记录，保留现有数据")
                            table_exists = True
                        else:
                            logger.info("数据库为空，重新创建表结构...")
                            cursor.execute("DROP TABLE IF EXISTS documents CASCADE;")
                            conn.commit()
                            table_exists = False
            
            if not table_exists:
                # 创建文档表
                cursor.execute(f"""
                    CREATE TABLE documents (
                        id SERIAL PRIMARY KEY,
                        doc_id VARCHAR(255) UNIQUE NOT NULL,
                        title TEXT,
                        content TEXT NOT NULL,
                        url TEXT,
                        search_term VARCHAR(255),
                        source VARCHAR(100) DEFAULT 'search_result',
                        chunk_id INTEGER DEFAULT 0,
                        embedding vector({self.vector_dim}),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX idx_documents_search_term 
                    ON documents(search_term);
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_documents_embedding 
                    ON documents USING hnsw (embedding vector_cosine_ops);
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_documents_created_at 
                    ON documents(created_at);
                """)
                
                logger.info(f"数据库表结构创建完成，向量维度: {self.vector_dim}")
            else:
                logger.info("数据库表结构已存在且维度匹配")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _create_document_id(self, content: str, metadata: Dict) -> str:
        """创建文档ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{metadata.get('source', 'unknown')}_{content_hash[:8]}"
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """将长文本分块"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
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
        将搜索结果添加到PostgreSQL知识库
        
        Args:
            search_results: 搜索结果列表
            search_term: 搜索关键词
            
        Returns:
            添加的文档数量
        """
        added_count = 0
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
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
                cursor.execute(
                    "SELECT id FROM documents WHERE doc_id = %s",
                    (doc_id,)
                )
                if cursor.fetchone():
                    logger.debug(f"文档已存在，跳过: {title[:50]}...")
                    continue
                
                # 文本分块
                chunks = self._chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    # 生成嵌入向量
                    try:
                        embedding = self.embedding_model.encode([chunk])[0]
                        
                        # 插入数据库
                        cursor.execute("""
                            INSERT INTO documents 
                            (doc_id, title, content, url, search_term, chunk_id, embedding, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            f"{doc_id}_chunk_{i}",
                            title,
                            chunk,
                            url,
                            search_term,
                            i,
                            Vector(embedding),
                            json.dumps(metadata)
                        ))
                        
                        added_count += 1
                        
                    except Exception as e:
                        logger.error(f"处理文档块失败: {e}")
                        continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"成功添加 {added_count} 个文档块到PostgreSQL知识库")
            return added_count
            
        except Exception as e:
            logger.error(f"添加搜索结果失败: {e}")
            return 0
    
    def search_similar(self, query: str, top_k: int = None, 
                      search_term: str = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回的相似文档数量，如果为None则使用配置文件中的默认值
            search_term: 可选的搜索关键词过滤
            
        Returns:
            相似文档列表，包含内容和元数据
        """
        if top_k is None:
            top_k = self.top_k
            
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query])[0]
            
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 构建SQL查询
            if search_term:
                sql = """
                    SELECT id, doc_id, title, content, url, search_term, metadata,
                           embedding <=> %s as similarity
                    FROM documents 
                    WHERE search_term = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """
                cursor.execute(sql, (Vector(query_embedding), search_term, 
                                   Vector(query_embedding), top_k))
            else:
                sql = """
                    SELECT id, doc_id, title, content, url, search_term, metadata,
                           embedding <=> %s as similarity
                    FROM documents 
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """
                cursor.execute(sql, (Vector(query_embedding), 
                                   Vector(query_embedding), top_k))
            
            results = []
            for i, row in enumerate(cursor.fetchall()):
                result = {
                    'content': row['content'],
                    'metadata': row['metadata'] if isinstance(row['metadata'], dict) 
                               else json.loads(row['metadata']) if row['metadata'] else {},
                    'similarity_score': float(row['similarity']),
                    'rank': i + 1,
                    'title': row['title'],
                    'url': row['url']
                }
                results.append(result)
            
            cursor.close()
            conn.close()
            
            logger.info(f"搜索完成，找到 {len(results)} 个相似文档")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_context_for_llm(self, query: str, max_tokens: int = None,
                           search_term: str = None) -> str:
        """
        为LLM生成上下文信息
        
        Args:
            query: 查询文本
            max_tokens: 最大token数，如果为None则使用配置文件中的默认值
            search_term: 可选的搜索关键词过滤
            
        Returns:
            格式化的上下文字符串
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        similar_docs = self.search_similar(query, top_k=None, search_term=search_term)
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 总文档数
            cursor.execute("SELECT COUNT(*) FROM documents;")
            total_chunks = cursor.fetchone()[0]
            
            # 唯一文档数
            cursor.execute("""
                SELECT COUNT(DISTINCT split_part(doc_id, '_chunk_', 1)) 
                FROM documents;
            """)
            total_documents = cursor.fetchone()[0]
            
            # 搜索关键词统计
            cursor.execute("""
                SELECT search_term, COUNT(*) 
                FROM documents 
                WHERE search_term IS NOT NULL 
                GROUP BY search_term 
                ORDER BY COUNT(*) DESC 
                LIMIT 10;
            """)
            search_terms = dict(cursor.fetchall())
            
            # 最新更新时间
            cursor.execute("""
                SELECT MAX(created_at) FROM documents;
            """)
            last_updated = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "search_terms": search_terms,
                "model_name": self.model_name,
                "vector_dim": self.vector_dim,
                "last_updated": last_updated.isoformat() if last_updated else None
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"total_documents": 0, "total_chunks": 0}
    
    def delete_old_documents(self, days: int = 30) -> int:
        """
        删除指定天数前的旧文档
        
        Args:
            days: 天数
            
        Returns:
            删除的文档数量
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM documents 
                WHERE created_at < NOW() - INTERVAL '%s days';
            """, (days,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"删除了 {deleted_count} 个旧文档")
            return deleted_count
            
        except Exception as e:
            logger.error(f"删除旧文档失败: {e}")
            return 0
    
    def export_knowledge_base(self, filepath: str) -> bool:
        """
        导出知识库到文件
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            是否成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT doc_id, title, content, url, search_term, metadata, created_at
                FROM documents
                ORDER BY created_at;
            """)
            
            documents = []
            for row in cursor.fetchall():
                doc = {
                    'doc_id': row['doc_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'url': row['url'],
                    'search_term': row['search_term'],
                    'metadata': row['metadata'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None
                }
                documents.append(doc)
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'model_name': self.model_name,
                'vector_dim': self.vector_dim,
                'total_documents': len(documents),
                'documents': documents
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            cursor.close()
            conn.close()
            
            logger.info(f"知识库已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出知识库失败: {e}")
            return False 