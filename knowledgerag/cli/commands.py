"""
CLI命令模块

该模块提供命令行接口
"""

import argparse
import logging
import sys
from typing import NamedTuple

from knowledgerag.utils.logging_utils import setup_logging
from knowledgerag.document_processing.processor_runner import DocumentProcessorRunner
from knowledgerag.core.model_loader import ModelLoader
from knowledgerag.core.rag_chain import RAGChainBuilder
from knowledgerag.core.interactive_qa import InteractiveQA


class Args(NamedTuple):
    """命令行参数类型"""

    process: bool
    force_rebuild: bool
    query: str | None


def parse_args() -> Args:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="知识库问答系统")

    parser.add_argument(
        "--process", action="store_true", help="处理文档并创建向量数据库"
    )

    parser.add_argument(
        "--force-rebuild", action="store_true", help="强制重建向量数据库"
    )

    parser.add_argument("--query", type=str, help="直接查询，不进入交互模式")

    args = parser.parse_args()
    return Args(
        process=args.process, force_rebuild=args.force_rebuild, query=args.query
    )


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 解析参数
    args: Args = parse_args()

    # 处理文档
    if args.process:
        processor_runner = DocumentProcessorRunner()
        success = processor_runner.run(force_rebuild=args.force_rebuild)

        if not success:
            sys.exit(1)

    # 加载模型
    logging.info("加载大语言模型...")
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load()
    logging.info("大语言模型加载完成")

    # 创建RAG链
    logging.info("创建RAG问答链...")
    rag_chain_builder = RAGChainBuilder(
        model=model,
        tokenizer=tokenizer,
    )

    rag_chain = rag_chain_builder.build()
    logging.info("RAG问答链创建完成")

    # 如果有直接查询，则处理查询
    if args.query:
        qa = InteractiveQA(rag_chain, rag_chain_builder)
        thinking, answer = qa.query(args.query)

        # 使用分隔线和格式化输出，确保结果清晰可见
        print("\n" + "="*80)
        print("查询结果")
        print("="*80)
        
        if thinking:
            print(f"\n思考过程:\n{thinking}\n")

        print(f"回答:\n{answer}\n")
        print("="*80)
    else:
        # 启动交互式问答
        qa = InteractiveQA(rag_chain, rag_chain_builder)
        qa.start()
