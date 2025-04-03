"""
日志工具模块

该模块提供日志配置和工具函数
"""
import logging
import sys
import os
from datetime import datetime

def setup_logging(level=logging.INFO):
    """
    设置日志配置
    
    参数:
        level: 日志级别
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，使用日期时间
    log_file = os.path.join(log_dir, f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    
    # 添加控制台处理器（仅用于错误和警告）
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    logging.info(f"日志配置完成，日志文件: {log_file}")
    
    return logging.getLogger(__name__)
