"""
向量存储模块

该模块负责向量存储的创建、加载和检索，使用FAISS进行高效向量检索，
并通过PostgreSQL进行持久化存储
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from knowledgerag.db.postgres import PostgresManager
from knowledgerag.embeddings.embedding_model import EmbeddingModel


class FaissVectorStore:
    """
    Faiss向量存储类

    负责创建、加载和检索Faiss向量存储，支持从PostgreSQL构建索引
    并提供混合查询能力

    属性:
        db_manager (PostgresManager): PostgreSQL数据库管理器
        index_name (str): FAISS索引名称
    """

    # 单例实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式
        """
        if cls._instance is None:
            cls._instance = super(FaissVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化Faiss向量存储

        """
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return

        self.index_name = "main"
        self.db_manager = PostgresManager()
        self.embedding_model = EmbeddingModel()
        self._initialized = True

    def load_or_create(self) -> bool:
        """
        从PostgreSQL加载或创建向量存储

        返回:
            bool: 加载或创建是否成功
        """
        try:
            # 尝试从PostgreSQL加载FAISS索引
            logging.info(f"尝试从PostgreSQL加载FAISS索引: {self.index_name}")
            faiss_index = self.db_manager.load_faiss_index(self.index_name)

            if faiss_index:
                # 创建FAISS向量存储
                self.vector_store = FAISS(
                    embedding_function=self.embedding_model.get_embeddings(),
                    index=faiss_index,
                    docstore={},  # 空文档存储，将在后续加载文档
                    index_to_docstore_id={},  # 索引到文档ID的映射
                )

                logging.info(f"成功从PostgreSQL加载FAISS索引: {self.index_name}")

                # 重建文档存储
                self._rebuild_docstore()
                return True
            else:
                logging.info(
                    f"PostgreSQL中未找到FAISS索引: {self.index_name}，将从文档数据构建新索引"
                )
                return self._build_from_postgres()

        except Exception as e:
            logging.error(f"加载或创建向量存储时出错: {str(e)}")
            return False

    def _rebuild_docstore(self) -> bool:
        """
        重建文档存储

        返回:
            bool: 重建是否成功
        """
        try:
            # 从PostgreSQL获取所有文档
            documents, _ = self.db_manager.get_all_documents_with_vectors()

            if not documents:
                logging.warning("PostgreSQL中没有文档数据，无法重建文档存储")
                return False

            # 重建文档存储
            docstore_dict = {}
            index_to_docstore_id = {}
            
            for i, doc in enumerate(documents):
                # 使用文档ID或生成新ID
                doc_id = doc.metadata.get("doc_id", f"doc_{i}")
                docstore_dict[doc_id] = doc
                index_to_docstore_id[i] = doc_id

            # 更新文档存储
            if self.vector_store:
                # 创建新的InMemoryDocstore实例
                docstore = InMemoryDocstore(docstore_dict)
                
                # 更新向量存储的docstore
                self.vector_store.docstore = docstore
                
                # 更新索引映射
                self.vector_store.index_to_docstore_id = index_to_docstore_id
                
                logging.info(f"成功重建文档存储，共 {len(docstore_dict)} 个文档，更新了索引映射")
                return True
            else:
                logging.error("向量存储未初始化，无法重建文档存储")
                return False

        except Exception as e:
            logging.error(f"重建文档存储时出错: {str(e)}")
            return False

    def _build_from_postgres(self) -> bool:
        """
        从PostgreSQL数据构建FAISS索引

        返回:
            bool: 构建是否成功
        """
        try:
            # 从PostgreSQL获取所有文档和向量
            documents, vectors = self.db_manager.get_all_documents_with_vectors()

            if not documents or not vectors:
                logging.warning("PostgreSQL中没有文档或向量数据，无法构建FAISS索引")
                return False

            # 检查向量数据类型并进行必要的转换
            processed_vectors = []
            for vec in vectors:
                # 如果向量是字符串类型，需要转换为浮点数列表
                if isinstance(vec, str):
                    try:
                        # 如果是类似"[0.1,0.2,0.3]"的字符串
                        if vec.startswith('[') and vec.endswith(']'):
                            # 去掉方括号，按逗号分割，转换为浮点数
                            vec = [float(x) for x in vec[1:-1].split(',')]
                        # 如果是类似"0.1,0.2,0.3"的字符串
                        else:
                            vec = [float(x) for x in vec.split(',')]
                    except ValueError as e:
                        logging.error(f"向量数据转换失败: {str(e)}, 原始数据: {vec[:100]}...")
                        continue
                
                processed_vectors.append(vec)
            
            if not processed_vectors:
                logging.warning("处理后的向量数据为空，无法构建FAISS索引")
                return False
            
            # 构建FAISS索引
            dimension = len(processed_vectors[0])
            index = faiss.IndexFlatL2(dimension)  # 创建基础索引

            # 将向量添加到索引
            vectors_array = np.array(processed_vectors).astype("float32")
            index.add(vectors_array)

            # 创建FAISS向量存储
            docstore = {}
            for i, doc in enumerate(documents[:len(processed_vectors)]):
                # 使用文档ID或生成新ID
                doc_id = doc.metadata.get("doc_id", f"doc_{i}")
                docstore[doc_id] = doc

            self.vector_store = FAISS(
                embedding_function=self.embedding_model.get_embeddings(),
                index=index,
                docstore=docstore,
                index_to_docstore_id={i: doc_id for i, doc_id in enumerate(docstore)},
            )

            # 保存索引到PostgreSQL
            success = self.db_manager.save_faiss_index(self.index_name, index)

            if success:
                logging.info(
                    f"成功从PostgreSQL数据构建FAISS索引并保存: {self.index_name}"
                )
                return True
            else:
                logging.error(f"保存FAISS索引到PostgreSQL失败: {self.index_name}")
                return False

        except Exception as e:
            logging.error(f"从PostgreSQL构建FAISS索引时出错: {str(e)}")
            return False

    def similarity_search(
        self, query: str, k: int = 4, use_postgres: bool = False
    ) -> List[Document]:
        """
        使用向量相似度搜索文档

        参数:
            query (str): 查询文本
            k (int): 返回的文档数量
            use_postgres (bool): 是否使用PostgreSQL进行搜索

        返回:
            List[Document]: 相似文档列表
        """
        try:
            if use_postgres:
                # 使用PostgreSQL进行搜索
                if not self.embedding_model:
                    logging.error("嵌入模型未初始化")
                    return []

                # 获取查询向量
                query_vector = self.embedding_model.get_embeddings().embed_query(query)

                # 使用PostgreSQL进行搜索
                return self.db_manager.similarity_search(query_vector, k)
            else:
                # 使用FAISS进行搜索
                if not self.vector_store:
                    success = self.load_or_create()
                    if not success:
                        return []

                # 执行相似度搜索
                documents = self.vector_store.similarity_search(query, k=k)
                return documents

        except Exception as e:
            logging.error(f"执行相似度搜索时出错: {str(e)}")
            return []

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        将向量存储转换为检索器

        参数:
            search_kwargs (Dict[str, Any]): 搜索参数

        返回:
            BaseRetriever: 检索器
        """
        if not self.vector_store:
            success = self.load_or_create()
            if not success:
                raise ValueError("无法创建检索器，向量存储未初始化")

        # 设置默认搜索参数
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        # 创建检索器
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def clear(self) -> bool:
        """
        清空向量存储

        返回:
            bool: 清空是否成功
        """
        try:
            if not self.embedding_model:
                logging.error("嵌入模型未初始化")
                return False

            # 删除PostgreSQL中的索引
            success = self.db_manager.delete_faiss_index(self.index_name)

            if success:
                logging.info(f"成功清空向量存储: {self.index_name}")
                return True
            else:
                logging.error(f"清空向量存储失败: {self.index_name}")
                return False

        except Exception as e:
            logging.error(f"清空向量存储时出错: {str(e)}")
            return False
