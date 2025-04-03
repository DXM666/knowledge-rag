"""
嵌入模型模块

该模块负责文本嵌入模型的加载和使用
"""
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings

from knowledgerag.config.settings import EMBEDDING_MODEL_CONFIG

class EmbeddingModel:
    """
    嵌入模型类
    
    负责加载和使用文本嵌入模型
    
    属性:
        model_name (str): 模型名称
        device (str): 设备类型，'cuda'或'cpu'
        max_length (int): 最大序列长度
        embeddings (HuggingFaceEmbeddings): 嵌入模型
    """

    # 单例实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式
        """
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_CONFIG["model_name"],
        device: str = EMBEDDING_MODEL_CONFIG["device"],
        max_length: int = EMBEDDING_MODEL_CONFIG["max_length"]
    ):
        """
        初始化嵌入模型
        
        参数:
            model_name (str): 模型名称
            device (str): 设备类型，'cuda'或'cpu'
            max_length (int): 最大序列长度
        """
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.embeddings = None
        self._initialized = True
        
    def load(self) -> Embeddings:
        """
        加载嵌入模型
        
        返回:
            Embeddings: 嵌入模型
        """
        try:
            logging.info(f"加载嵌入模型: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True, 'max_length': self.max_length}
            )
            logging.info(f"嵌入模型加载成功，使用设备: {self.device}")
            return self.embeddings
            
        except Exception as e:
            logging.error(f"加载嵌入模型时出错: {str(e)}")
            raise
    
    def get_embeddings(self) -> Embeddings:
        """
        获取嵌入模型
        
        返回:
            Embeddings: 嵌入模型
        """
        if not self.embeddings:
            return self.load()
        return self.embeddings
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文本列表
        
        参数:
            texts (List[str]): 文本列表
            
        返回:
            List[List[float]]: 嵌入向量列表
        """
        if not self.embeddings:
            self.load()
        
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        嵌入查询文本
        
        参数:
            query (str): 查询文本
            
        返回:
            List[float]: 嵌入向量
        """
        if not self.embeddings:
            self.load()
        
        return self.embeddings.embed_query(query)
