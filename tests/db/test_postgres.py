"""
PostgreSQL数据库连接和操作的单元测试

该模块测试PostgreSQL数据库的连接、初始化和基本操作
"""
import unittest
import logging
import json
import numpy as np
import sys
import os
from langchain.schema import Document

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from knowledgerag.db.postgres import PostgresManager
from knowledgerag.config.settings import DB_CONFIG

# 配置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPostgresManager(unittest.TestCase):
    """测试PostgresManager类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试用的数据库管理器
        self.db_manager = PostgresManager(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            table_name="test_document_embeddings"  # 使用测试表名
        )
        
        # 连接数据库
        self.db_manager.connect()
        
        # 初始化数据库
        self.db_manager.initialize_database()
        
        # 清空测试表
        self.db_manager.clear_table()
        
        # 准备测试数据
        self.test_documents = [
            Document(
                page_content="这是测试文档1",
                metadata={"source": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="这是测试文档2",
                metadata={"source": "test2.pdf", "page": 2}
            ),
            Document(
                page_content="这是一个完全不同的文档",
                metadata={"source": "test3.pdf", "page": 3}
            )
        ]
        
        # 准备测试向量
        self.test_embeddings = [
            [0.1] * 768,  # 文档1的向量
            [0.2] * 768,  # 文档2的向量
            [0.9] * 768   # 文档3的向量
        ]
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清空测试表
        if self.db_manager.conn:
            self.db_manager.clear_table()
            
            # 关闭数据库连接
            self.db_manager.close()
    
    def test_connect(self):
        """测试数据库连接"""
        # 重新创建连接
        self.db_manager.close()
        success = self.db_manager.connect()
        
        # 验证连接成功
        self.assertTrue(success)
        self.assertIsNotNone(self.db_manager.conn)
        self.assertIsNotNone(self.db_manager.engine)
    
    def test_initialize_database(self):
        """测试数据库初始化"""
        # 重新初始化数据库
        success = self.db_manager.initialize_database()
        
        # 验证初始化成功
        self.assertTrue(success)
        
        # 验证表是否存在
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{self.db_manager.table_name}')")
        table_exists = cursor.fetchone()[0]
        cursor.close()
        
        self.assertTrue(table_exists)
    
    def test_add_documents(self):
        """测试添加文档"""
        # 添加文档
        success = self.db_manager.add_documents(self.test_documents, self.test_embeddings)
        
        # 验证添加成功
        self.assertTrue(success)
        
        # 验证文档数量
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.db_manager.table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        
        self.assertEqual(count, len(self.test_documents))
    
    def test_similarity_search(self):
        """测试相似度搜索"""
        # 添加文档
        self.db_manager.add_documents(self.test_documents, self.test_embeddings)
        
        # 执行相似度搜索
        query_vector = [0.2] * 768  # 与文档2最相似
        results = self.db_manager.similarity_search(query_vector, k=2)
        
        # 验证搜索结果
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "这是测试文档2")  # 第一个结果应该是文档2
    
    def test_clear_table(self):
        """测试清空表"""
        # 添加文档
        self.db_manager.add_documents(self.test_documents, self.test_embeddings)
        
        # 清空表
        success = self.db_manager.clear_table()
        
        # 验证清空成功
        self.assertTrue(success)
        
        # 验证表是否为空
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.db_manager.table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        
        self.assertEqual(count, 0)

if __name__ == "__main__":
    unittest.main()
