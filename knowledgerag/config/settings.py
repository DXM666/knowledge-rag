"""
配置文件，存储项目中所有可配置参数
"""

import os
from pathlib import Path
import torch

# 项目根目录
ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 模型配置
MODEL_CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "load_in_4bit": True,
    "max_memory": {0: "8GiB", "cpu": "16GiB"},
    "device_map": "auto",
    "offload_folder": "offload",
}

# 嵌入模型配置
EMBEDDING_MODEL_CONFIG = {
    "model_name": "BAAI/bge-m3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_length": 512,
}

# 生成配置
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "eos_token_id": None,  # 将在运行时设置
    "pad_token_id": None,  # 将在运行时设置
}

# PostgreSQL配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "knowledge_base",
    "user": "root",
    "password": "admin",
    "table_name": "document_embeddings",
}

# 检索配置
RETRIEVER_CONFIG = {"top_k": 3}

# 文本分割配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
}

# 提示模板
PROMPT_TEMPLATE = """你是一个基于知识库的问答助手。请基于以下检索到的上下文信息，回答用户的问题。如果上下文中没有相关信息，请直接说"我没有找到相关信息"，不要编造答案。

上下文信息:
{context}

用户问题: {question}

思考过程：请先分析上下文信息中与问题相关的部分，然后组织一个清晰的回答。

回答: """
