"""
模型加载模块

该模块负责加载和配置大语言模型
"""

import logging
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from knowledgerag.config.settings import MODEL_CONFIG


class ModelLoader:
    """
    模型加载器类

    负责加载和配置大语言模型

    属性:
        model_name (str): 模型名称
        load_in_4bit (bool): 是否使用4bit量化
        max_memory (Dict): 最大内存配置
        device_map (str): 设备映射
        offload_folder (str): 卸载文件夹
        model: 加载的模型
        tokenizer: 分词器
    """

    def __init__(
        self,
        model_name: str = MODEL_CONFIG["model_name"],
        load_in_4bit: bool = MODEL_CONFIG["load_in_4bit"],
        max_memory: Dict = None,
        device_map: str = MODEL_CONFIG["device_map"],
        offload_folder: str = MODEL_CONFIG["offload_folder"],
    ):
        """
        初始化模型加载器

        参数:
            model_name (str): 模型名称
            load_in_4bit (bool): 是否使用4bit量化
            max_memory (Dict): 最大内存配置
            device_map (str): 设备映射
            offload_folder (str): 卸载文件夹
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.max_memory = (
            max_memory if max_memory is not None else MODEL_CONFIG["max_memory"].copy()
        )
        self.device_map = device_map
        self.offload_folder = offload_folder
        self.model = None
        self.tokenizer = None

    def load(self):
        """
        加载模型和分词器

        返回:
            tuple: (model, tokenizer) 模型和分词器
        """
        try:
            logging.info("开始加载模型: %s", self.model_name)
            logging.info("设备映射: %s", self.device_map)
            logging.info("内存配置: %s", self.max_memory)
            logging.info("是否使用4bit量化: %s", self.load_in_4bit)

            # 加载分词器
            logging.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logging.info("分词器加载完成")

            # 设置模型加载参数
            model_kwargs = {
                "device_map": self.device_map,
                "max_memory": self.max_memory,
                "offload_folder": self.offload_folder,
                "trust_remote_code": True,
            }

            # 如果使用4bit量化
            if self.load_in_4bit:
                logging.info("配置4bit量化参数...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                model_kwargs["quantization_config"] = quantization_config
                logging.info("4bit量化参数配置完成")

            # 加载模型
            logging.info("正在加载模型，这可能需要几分钟时间...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
            logging.info("模型加载完成")

            # 设置生成配置中的特殊token
            if not self.tokenizer.pad_token_id:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logging.info("设置pad_token_id为eos_token_id: %s", self.tokenizer.pad_token_id)

            logging.info("模型和分词器加载成功")
            return self.model, self.tokenizer

        except Exception as e:
            logging.error("加载模型时出错: %s", str(e))
            logging.exception("详细错误信息:")
            raise

    def create_pipeline(self):
        """
        创建文本生成管道

        返回:
            HuggingFacePipeline: Hugging Face管道
        """
        if not self.model or not self.tokenizer:
            self.load()

        # 创建文本生成管道
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        # 创建Hugging Face管道
        return HuggingFacePipeline(pipeline=text_generation_pipeline)
