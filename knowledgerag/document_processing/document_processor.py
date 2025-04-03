"""
文档处理模块

该模块负责处理文档，包括加载、分割和向量化
"""

import logging
from typing import List
import os.path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from knowledgerag.config.settings import TEXT_SPLITTER_CONFIG
from knowledgerag.db.postgres import PostgresManager
from knowledgerag.retrieval.vector_store import FaissVectorStore
from knowledgerag.embeddings.embedding_model import EmbeddingModel


class DocumentProcessor:
    """
    文档处理器类

    负责加载PDF文档，将其分割成小块，并存储到向量数据库和PostgreSQL中
    """

    # 单例实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式
        """
        if cls._instance is None:
            cls._instance = super(DocumentProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化文档处理器
        """
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return

        # 使用递归分割器，更适合处理中文文档
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
            chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            length_function=len,
            separators=TEXT_SPLITTER_CONFIG["separators"],
        )

        # 初始化向量存储
        self.vector_store = FaissVectorStore()

        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel()

        # 初始化数据库管理器
        self.db_manager = PostgresManager()

        # 初始化数据库
        self._init_database()

        self._initialized = True

    def _init_database(self):
        """初始化数据库"""
        try:
            # 连接数据库
            success = self.db_manager.connect()
            if not success:
                logging.warning("无法连接到PostgreSQL数据库，将仅使用FAISS向量存储")
                return

            # 初始化数据库
            success = self.db_manager.initialize_database()
            if not success:
                logging.warning("初始化PostgreSQL数据库失败，将仅使用FAISS向量存储")
                return

            logging.info("PostgreSQL数据库初始化成功")

        except Exception as e:
            logging.error("初始化数据库时出错: %s", str(e))
            logging.warning("将仅使用FAISS向量存储")

    def process_document(self, file_path: str):
        """
        处理PDF文档

        参数:
            file_path (str): PDF文件路径
        """
        try:
            logging.info("加载文档: %s", file_path)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 检查文件是否为PDF
            if not file_path.lower().endswith(".pdf"):
                raise ValueError(f"不支持的文件格式，仅支持PDF文件: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            logging.info("分割文档...")
            documents = []
            for page in pages:
                page_texts = self.text_splitter.split_text(page.page_content)

                # 创建Document对象列表
                for text in page_texts:
                    doc = page.copy()
                    doc.page_content = text
                    documents.append(doc)

            logging.info("共分割出 %s 个文本块", len(documents))

            # 处理PostgreSQL存储
            self._process_postgres(documents)

            # 从PostgreSQL构建FAISS索引
            self._build_faiss_from_postgres()

            logging.info("处理完成！")

        except FileNotFoundError as e:
            logging.error("文件不存在: %s", str(e))
            raise
        except ValueError as e:
            logging.error("文件格式错误: %s", str(e))
            raise
        except ImportError as e:
            logging.error("导入模块错误: %s", str(e))
            raise
        except Exception as e:
            logging.error("处理文档时出错: %s", str(e))
            raise

    def _process_postgres(self, documents: List[Document]):
        """
        处理PostgreSQL存储

        参数:
            documents (List[Document]): 文档列表

        异常:
            Exception: 处理PostgreSQL存储时出错
        """
        try:
            if not self.db_manager.conn:
                logging.warning("PostgreSQL数据库未连接，跳过PostgreSQL存储")
                return

            # 清理文档内容，移除NUL字符
            cleaned_documents = []
            for doc in documents:
                # 移除NUL字符和其他不可打印字符
                cleaned_content = doc.page_content.replace("\x00", "")
                # 创建新的文档对象
                cleaned_doc = Document(
                    page_content=cleaned_content, metadata=doc.metadata
                )
                cleaned_documents.append(cleaned_doc)

            logging.info("将文档嵌入向量...")
            texts = [doc.page_content for doc in cleaned_documents]
            embeddings = self.embedding_model.embed_texts(texts)

            logging.info("添加文档到PostgreSQL...")
            success = self.db_manager.add_documents(cleaned_documents, embeddings)
            if not success:
                logging.error("添加文档到PostgreSQL失败")
                return

            logging.info("PostgreSQL存储更新成功")

        except ConnectionError as e:
            logging.error("数据库连接错误: %s", str(e))
            logging.warning("将仅使用FAISS向量存储")
        except ImportError as e:
            logging.error("模型加载错误: %s", str(e))
            logging.warning("将仅使用FAISS向量存储")
        except Exception as e:
            logging.error("处理PostgreSQL存储时出错: %s", str(e))
            logging.warning("将仅使用FAISS向量存储")

    def _build_faiss_from_postgres(self):
        """
        从PostgreSQL构建FAISS索引
        """
        try:
            logging.info("从PostgreSQL构建FAISS索引...")

            # 初始化或加载FAISS索引
            success = self.vector_store.load_or_create()
            if not success:
                logging.error("初始化FAISS索引失败")
                return

            logging.info("FAISS索引更新成功")

        except Exception as e:
            logging.error("从PostgreSQL构建FAISS索引时出错: %s", str(e))
            raise

    def clear_stores(self, clear_faiss: bool = True, clear_postgres: bool = True):
        """
        清空存储

        参数:
            clear_faiss (bool): 是否清空FAISS向量存储
            clear_postgres (bool): 是否清空PostgreSQL存储
        """
        if clear_faiss:
            logging.info("清空FAISS向量存储...")
            success = self.vector_store.clear()
            if success:
                logging.info("FAISS向量存储已清空")
            else:
                logging.error("清空FAISS向量存储失败")

        if clear_postgres and self.db_manager.conn:
            logging.info("清空PostgreSQL存储...")
            success = self.db_manager.clear_table()
            if success:
                logging.info("PostgreSQL存储已清空")
            else:
                logging.error("清空PostgreSQL存储失败")
