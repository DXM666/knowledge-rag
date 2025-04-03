"""
RAG链模块

该模块负责构建检索增强生成链
"""

import logging
from typing import Tuple, Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline

from knowledgerag.config.settings import PROMPT_TEMPLATE, RETRIEVER_CONFIG, GENERATION_CONFIG
from knowledgerag.retrieval.vector_store import FaissVectorStore
from knowledgerag.embeddings.embedding_model import EmbeddingModel
from knowledgerag.db.postgres import PostgresManager
from transformers import pipeline


class RAGChainBuilder:
    """
    RAG链构建器类

    负责构建检索增强生成链

    属性:
        model: 语言模型
        tokenizer: 分词器
        vector_store_path (str): 向量存储路径
        embedding_model (EmbeddingModel): 嵌入模型
        db_manager (PostgresManager): 数据库管理器
        top_k (int): 检索文档数量
        rag_chain: 构建的RAG链
    """

    def __init__(
        self,
        model,
        tokenizer,
        embedding_model: Optional[EmbeddingModel] = None,
        db_manager: Optional[PostgresManager] = None,
        top_k: int = RETRIEVER_CONFIG["top_k"],
    ):
        """
        初始化RAG链构建器

        参数:
            model: 语言模型
            tokenizer: 分词器
            vector_store_path (str): 向量存储路径
            embedding_model (EmbeddingModel): 嵌入模型
            db_manager (PostgresManager): 数据库管理器
            top_k (int): 检索文档数量
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model or EmbeddingModel()
        self.db_manager = db_manager or PostgresManager()
        self.top_k = top_k
        self.rag_chain = None

    def build(self):
        """
        构建RAG链

        返回:
            构建的RAG链

        异常:
            FileNotFoundError: 向量存储不存在
        """
        # 初始化向量存储和检索器
        try:
            logging.info("初始化向量检索器...")

            # 初始化向量存储
            vector_store = FaissVectorStore()

            # 加载向量存储
            success = vector_store.load_or_create()
            if not success:
                raise ValueError("加载向量存储失败")

            # 创建检索器
            retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

            logging.info("向量检索器初始化成功")

        except Exception as e:
            logging.error(f"初始化向量检索器时出错: {str(e)}")
            raise

        # 创建LLM管道
        logging.info("创建LLM管道...")
        
        # 使用已加载的模型和分词器创建管道
        llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                do_sample=GENERATION_CONFIG["do_sample"],
                return_full_text=False,  # 只返回新生成的文本
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        )
        
        logging.info("LLM管道创建成功")

        # 创建提示模板
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )

        # 构建RAG链
        def format_docs(docs):
            """格式化检索到的文档"""
            return "\n\n".join(
                [f"文档 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        return self.rag_chain

    def process_response(self, response: Dict[str, Any]) -> Tuple[str, str]:
        """
        处理RAG链的响应
        
        参数:
            response (Dict[str, Any]): RAG链的响应
            
        返回:
            Tuple[str, str]: (思考过程, 最终回答)
        """
        try:
            # 获取思考过程
            thinking = ""
            if "thinking" in response:
                thinking = response["thinking"]
            
            # 获取最终回答
            answer = ""
            if "answer" in response:
                answer = response["answer"]
                # 清理可能的混乱输出
                # 有时模型输出会包含 </think> 标记或其他噪声
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                
                # 移除可能的前缀
                prefixes = ["回答:", "答:", "答案:"]
                for prefix in prefixes:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                        break
            
            # 如果没有获取到回答，尝试从response本身获取
            if not answer and isinstance(response, str):
                answer = response
            
            # 确保返回的是干净的字符串
            thinking = thinking.strip() if thinking else ""
            answer = answer.strip() if answer else ""
            
            return thinking, answer
        except Exception as e:
            logging.error(f"处理响应时出错: {str(e)}")
            return "", f"处理响应时出错: {str(e)}"