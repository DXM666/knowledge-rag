"""
PostgreSQL数据库连接和操作模块

该模块负责PostgreSQL数据库的连接和基本操作
"""

import logging
from typing import List, Dict, Any, Tuple 
import psycopg2
from psycopg2.extras import execute_values, Json
from sqlalchemy import create_engine
from langchain.schema import Document
import faiss
import os

from knowledgerag.config.settings import DB_CONFIG


class PostgresManager:
    """
    PostgreSQL数据库管理器

    负责连接PostgreSQL数据库，创建表，以及执行基本的数据库操作

    属性:
        host (str): 数据库主机
        port (int): 数据库端口
        database (str): 数据库名称
        user (str): 数据库用户名
        password (str): 数据库密码
        table_name (str): 表名
        conn: 数据库连接
        engine: SQLAlchemy引擎
    """

    def __init__(
        self,
        host: str = DB_CONFIG["host"],
        port: int = DB_CONFIG["port"],
        database: str = DB_CONFIG["database"],
        user: str = DB_CONFIG["user"],
        password: str = DB_CONFIG["password"],
        table_name: str = DB_CONFIG["table_name"],
    ):
        """
        初始化PostgreSQL管理器

        参数:
            host (str): 数据库主机
            port (int): 数据库端口
            database (str): 数据库名称
            user (str): 数据库用户名
            password (str): 数据库密码
            table_name (str): 表名
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.table_name = table_name
        self.conn = None
        self.engine = None

    def connect(self) -> bool:
        """
        连接到PostgreSQL数据库

        返回:
            bool: 连接是否成功
        """
        try:
            # 创建连接
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )

            # 创建SQLAlchemy引擎
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)

            logging.info(
                f"成功连接到PostgreSQL数据库: {self.host}:{self.port}/{self.database}"
            )
            return True

        except Exception as e:
            logging.error(f"连接到PostgreSQL数据库时出错: {str(e)}")
            return False

    def initialize_database(self) -> bool:
        """
        初始化数据库，创建必要的扩展和表

        返回:
            bool: 初始化是否成功
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return False

            # 创建游标
            cursor = self.conn.cursor()

            # 创建pgvector扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 创建表
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(768)
            );
            """
            )

            # 创建向量索引
            cursor.execute(
                f"""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON {self.table_name} 
            USING ivfflat (embedding vector_l2_ops)
            WITH (lists = 100);
            """
            )

            # 创建FAISS索引表
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS faiss_indices (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                index_data BYTEA NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
            )

            # 提交事务
            self.conn.commit()

            logging.info(f"成功初始化数据库和表: {self.table_name}")
            return True

        except Exception as e:
            logging.error(f"初始化数据库时出错: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> bool:
        """
        添加文档和对应的嵌入向量到数据库

        参数:
            documents (List[Document]): 文档列表
            embeddings (List[List[float]]): 嵌入向量列表

        返回:
            bool: 添加是否成功
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return False

            # 创建游标
            cursor = self.conn.cursor()

            # 准备数据
            data = []
            for doc, embedding in zip(documents, embeddings):
                # 使用Json适配器处理元数据字典
                data.append((doc.page_content, Json(doc.metadata), embedding))

            # 批量插入数据
            execute_values(
                cursor,
                f"INSERT INTO {self.table_name} (content, metadata, embedding) VALUES %s",
                data,
                template="(%s, %s, %s::vector)",
            )

            # 提交事务
            self.conn.commit()

            logging.info(f"成功添加 {len(documents)} 个文档到数据库")
            return True

        except Exception as e:
            logging.error(f"添加文档到数据库时出错: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def similarity_search(
        self, query_vector: List[float], k: int = 4
    ) -> List[Document]:
        """
        使用向量相似度搜索文档

        参数:
            query_vector (List[float]): 查询向量
            k (int): 返回的文档数量

        返回:
            List[Document]: 相似文档列表
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return []

            # 创建游标
            cursor = self.conn.cursor()

            # 执行相似度搜索
            query = f"""
            SELECT content, metadata
            FROM {self.table_name}
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
            """
            cursor.execute(query, (query_vector, k))

            # 获取结果
            results = cursor.fetchall()

            # 转换为Document对象
            documents = []
            for content, metadata in results:
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents

        except Exception as e:
            logging.error(f"执行相似度搜索时出错: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()

    def save_faiss_index(self, index_name: str, faiss_index) -> bool:
        """
        将FAISS索引保存到PostgreSQL

        参数:
            index_name (str): 索引名称
            faiss_index: FAISS索引对象

        返回:
            bool: 保存是否成功
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return False

            # 创建游标
            cursor = self.conn.cursor()

            # 序列化FAISS索引
            temp_file = os.path.join(os.getcwd(), f"temp_{index_name}.index")
            try:
                # 先保存到临时文件
                faiss.write_index(faiss_index, temp_file)
                
                # 读取临时文件内容
                with open(temp_file, 'rb') as f:
                    index_bytes = f.read()
                    
                # 删除临时文件
                os.remove(temp_file)
            except Exception as e:
                logging.error(f"保存FAISS索引到临时文件时出错: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise

            # 检查索引是否已存在
            cursor.execute(
                "SELECT id FROM faiss_indices WHERE name = %s", (index_name,)
            )
            result = cursor.fetchone()

            if result:
                # 更新现有索引
                cursor.execute(
                    "UPDATE faiss_indices SET index_data = %s, updated_at = CURRENT_TIMESTAMP WHERE name = %s",
                    (psycopg2.Binary(index_bytes), index_name),
                )
                logging.info(f"成功更新FAISS索引: {index_name}")
            else:
                # 插入新索引
                cursor.execute(
                    "INSERT INTO faiss_indices (name, index_data) VALUES (%s, %s)",
                    (index_name, psycopg2.Binary(index_bytes)),
                )
                logging.info(f"成功创建FAISS索引: {index_name}")

            # 提交事务
            self.conn.commit()
            return True

        except Exception as e:
            logging.error(f"保存FAISS索引时出错: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def load_faiss_index(self, index_name: str):
        """
        从PostgreSQL加载FAISS索引

        参数:
            index_name (str): 索引名称

        返回:
            FAISS索引对象或None
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return None

            # 创建游标
            cursor = self.conn.cursor()

            # 查询索引数据
            cursor.execute(
                "SELECT index_data FROM faiss_indices WHERE name = %s", (index_name,)
            )
            result = cursor.fetchone()

            if not result:
                logging.warning(f"未找到FAISS索引: {index_name}")
                return None

            # 使用临时文件加载FAISS索引
            temp_file = os.path.join(os.getcwd(), f"temp_load_{index_name}.index")
            try:
                # 将二进制数据写入临时文件
                with open(temp_file, 'wb') as f:
                    f.write(result[0])
                
                # 从临时文件加载索引
                faiss_index = faiss.read_index(temp_file)
                
                # 删除临时文件
                os.remove(temp_file)
                
                logging.info(f"成功加载FAISS索引: {index_name}")
                return faiss_index
                
            except Exception as e:
                logging.error(f"从临时文件加载FAISS索引时出错: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise

        except Exception as e:
            logging.error(f"加载FAISS索引时出错: {str(e)}")
            return None
        finally:
            if cursor:
                cursor.close()

    def delete_faiss_index(self, index_name: str) -> bool:
        """
        从PostgreSQL删除FAISS索引

        参数:
            index_name (str): 索引名称

        返回:
            bool: 删除是否成功
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return False

            # 创建游标
            cursor = self.conn.cursor()

            # 删除索引
            cursor.execute("DELETE FROM faiss_indices WHERE name = %s", (index_name,))

            # 提交事务
            self.conn.commit()

            if cursor.rowcount > 0:
                logging.info(f"成功删除FAISS索引: {index_name}")
                return True
            else:
                logging.warning(f"未找到要删除的FAISS索引: {index_name}")
                return False

        except Exception as e:
            logging.error(f"删除FAISS索引时出错: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def list_faiss_indices(self) -> List[Dict[str, Any]]:
        """
        列出所有FAISS索引

        返回:
            List[Dict[str, Any]]: 索引信息列表
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return []

            # 创建游标
            cursor = self.conn.cursor()

            # 查询所有索引
            cursor.execute(
                "SELECT id, name, created_at, updated_at FROM faiss_indices ORDER BY name"
            )
            results = cursor.fetchall()

            # 转换为字典列表
            indices = []
            for id, name, created_at, updated_at in results:
                indices.append(
                    {
                        "id": id,
                        "name": name,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    }
                )

            return indices

        except Exception as e:
            logging.error(f"列出FAISS索引时出错: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()

    def get_all_documents_with_vectors(
        self,
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        获取所有文档和向量数据

        返回:
            Tuple[List[Document], List[List[float]]]: 文档列表和向量列表
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return [], []

            # 创建游标
            cursor = self.conn.cursor()

            # 查询所有文档和向量
            cursor.execute(
                f"""
                SELECT content, metadata, embedding
                FROM {self.table_name}
                ORDER BY id;
            """
            )

            # 获取结果
            results = cursor.fetchall()

            # 分离文档和向量
            documents = []
            vectors = []
            for content, metadata, embedding in results:
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
                # 处理向量数据
                # PostgreSQL的vector类型可能返回为字符串，如"[0.1,0.2,0.3]"
                # 需要将其转换为浮点数列表
                if isinstance(embedding, str):
                    # 如果是类似"[0.1,0.2,0.3]"的字符串
                    if embedding.startswith('[') and embedding.endswith(']'):
                        # 去掉方括号，按逗号分割，转换为浮点数
                        embedding = [float(x) for x in embedding[1:-1].split(',')]
                    # 如果是类似"0.1,0.2,0.3"的字符串
                    else:
                        embedding = [float(x) for x in embedding.split(',')]
                
                vectors.append(embedding)

            logging.info(f"从PostgreSQL获取了 {len(documents)} 个文档和向量")
            return documents, vectors

        except Exception as e:
            logging.error(f"获取所有文档和向量时出错: {str(e)}")
            return [], []
        finally:
            if cursor:
                cursor.close()

    def clear_table(self) -> bool:
        """
        清空表中的所有数据

        返回:
            bool: 清空是否成功
        """
        try:
            if not self.conn:
                success = self.connect()
                if not success:
                    return False

            # 创建游标
            cursor = self.conn.cursor()

            # 清空表
            cursor.execute(f"TRUNCATE TABLE {self.table_name};")

            # 提交事务
            self.conn.commit()

            logging.info(f"成功清空表: {self.table_name}")
            return True

        except Exception as e:
            logging.error(f"清空表时出错: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logging.info("已关闭PostgreSQL数据库连接")
