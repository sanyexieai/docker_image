"""
数据库配置文件
支持PostgreSQL和pgvector的RAG知识库配置
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# 确保在配置加载前加载环境变量，强制覆盖已存在的环境变量
load_dotenv(override=True)

class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # PostgreSQL配置
        self.postgres_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'rag_knowledge'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password'),
            'sslmode': os.getenv('POSTGRES_SSLMODE', 'prefer')
        }
        
        # RAG配置 - 使用通用的句子嵌入模型
        self.rag_config = {
            'model_name': os.getenv('RAG_MODEL_NAME', 'all-MiniLM-L6-v2'),
            'vector_dim': int(os.getenv('RAG_VECTOR_DIM', '384')),
            'chunk_size': int(os.getenv('RAG_CHUNK_SIZE', '500')),
            'chunk_overlap': int(os.getenv('RAG_CHUNK_OVERLAP', '50')),
            'max_tokens': int(os.getenv('RAG_MAX_TOKENS', '4000')),
            'top_k': int(os.getenv('RAG_TOP_K', '10')),
            'device': os.getenv('RAG_DEVICE', 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu')
        }
        
        # 连接池配置
        self.pool_config = {
            'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', '1')),
            'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', '10')),
            'connection_timeout': int(os.getenv('DB_CONNECTION_TIMEOUT', '30'))
        }
    
    def get_postgres_config(self) -> Dict[str, str]:
        """获取PostgreSQL配置"""
        return self.postgres_config.copy()
    
    def get_rag_config(self) -> Dict[str, Any]:
        """获取RAG配置"""
        return self.rag_config.copy()
    
    def get_pool_config(self) -> Dict[str, int]:
        """获取连接池配置"""
        return self.pool_config.copy()
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        required_fields = ['host', 'database', 'user', 'password']
        
        for field in required_fields:
            if not self.postgres_config.get(field):
                print(f"警告: 缺少PostgreSQL配置字段: {field}")
                return False
        
        return True
    
    def print_config(self):
        """打印配置信息（隐藏敏感信息）"""
        print("=== 数据库配置 ===")
        print(f"主机: {self.postgres_config['host']}:{self.postgres_config['port']}")
        print(f"数据库: {self.postgres_config['database']}")
        print(f"用户: {self.postgres_config['user']}")
        print(f"SSL模式: {self.postgres_config['sslmode']}")
        
        print("\n=== RAG配置 ===")
        print(f"模型: {self.rag_config['model_name']}")
        print(f"向量维度: {self.rag_config['vector_dim']}")
        print(f"设备: {self.rag_config['device']}")
        print(f"分块大小: {self.rag_config['chunk_size']}")
        print(f"分块重叠: {self.rag_config['chunk_overlap']}")
        print(f"最大Token: {self.rag_config['max_tokens']}")
        print(f"Top-K: {self.rag_config['top_k']}")
        
        print("\n=== 连接池配置 ===")
        print(f"最小连接数: {self.pool_config['min_connections']}")
        print(f"最大连接数: {self.pool_config['max_connections']}")
        print(f"连接超时: {self.pool_config['connection_timeout']}秒")

# 全局配置实例
db_config = DatabaseConfig() 