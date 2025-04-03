"""
文档处理运行器模块

该模块负责批量处理文档
"""

import logging
import os
import glob

from knowledgerag.config.settings import DATA_DIR
from knowledgerag.document_processing.document_processor import DocumentProcessor


class DocumentProcessorRunner:
    """
    文档处理运行器

    负责批量处理文档
    """

    # 单例实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式
        """
        if cls._instance is None:
            cls._instance = super(DocumentProcessorRunner, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        初始化文档处理运行器
        """
        # 避免重复初始化
        if getattr(self, "_initialized", False):
            return

        self._initialized = True

    def _get_pdf_files(self):
        """
        获取PDF文件列表
        """
        return glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)

    def run(self, force_rebuild: bool = False) -> bool:
        """
        运行文档处理

        参数:
            force_rebuild (bool): 是否强制重建向量数据库

        返回:
            bool: 处理是否成功
        """
        # 获取PDF文件列表
        pdf_files = self._get_pdf_files()
        if not pdf_files:
            logging.warning("没有找到PDF文件")
            return False

        # 如果需要重建，先清空存储
        if force_rebuild:
            logging.info("强制重建向量数据库...")
            DocumentProcessor().clear_stores()

        # 处理文档
        try:
            for pdf_file in pdf_files:
                logging.info("处理文件: %s", pdf_file)
                DocumentProcessor().process_document(pdf_file)

            logging.info("所有文档处理完成")
            return True
            
        except FileNotFoundError as e:
            logging.error("文件不存在: %s", str(e))
            return False
        except ValueError as e:
            logging.error("文件格式错误: %s", str(e))
            return False
        except ImportError as e:
            logging.error("导入模块错误: %s", str(e))
            return False
        except ConnectionError as e:
            logging.error("数据库连接错误: %s", str(e))
            return False
        except PermissionError as e:
            logging.error("文件访问权限错误: %s", str(e))
            return False
        except Exception as e:
            logging.error("处理文档时出错: %s", str(e))
            logging.exception("详细错误信息:")
            return False
