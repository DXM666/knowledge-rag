"""
交互式问答模块

该模块负责与用户进行问答交互
"""
import logging
from typing import Tuple

class InteractiveQA:
    """
    交互式问答类
    
    负责与用户进行问答交互
    
    属性:
        rag_chain: RAG链
        rag_chain_builder: RAG链构建器
    """
    
    def __init__(self, rag_chain, rag_chain_builder):
        """
        初始化交互式问答
        
        参数:
            rag_chain: RAG链
            rag_chain_builder: RAG链构建器
        """
        self.rag_chain = rag_chain
        self.rag_chain_builder = rag_chain_builder
    
    def query(self, question: str) -> Tuple[str, str]:
        """
        查询单个问题
        
        参数:
            question (str): 问题
            
        返回:
            tuple: (thinking, answer) 思考过程和最终回答
        """
        try:
            logging.info(f"处理查询: {question}")
            
            # 使用RAG链生成回答
            logging.info("调用RAG链...")
            response = self.rag_chain.invoke(question)
            logging.info(f"RAG链返回响应，长度: {len(str(response))}")
            
            # 处理回答
            logging.info("处理响应...")
            thinking, answer = self.rag_chain_builder.process_response(response)
            logging.info(f"响应处理完成，思考过程长度: {len(thinking)}，回答长度: {len(answer)}")
            
            # 确保返回的是干净的字符串
            thinking = thinking.strip() if thinking else ""
            answer = answer.strip() if answer else ""
            
            return thinking, answer
            
        except Exception as e:
            logging.error(f"查询时出错: {str(e)}")
            logging.exception("详细错误信息:")
            return "", f"查询处理过程中出现错误，请稍后再试。错误信息: {str(e)}"
    
    def start(self):
        """
        启动交互式问答会话
        """
        print("\n欢迎使用知识库问答系统！")
        print("输入问题进行查询，输入'exit'或'quit'退出。\n")
        
        while True:
            # 获取用户输入
            question = input("\n请输入问题: ")
            
            # 检查是否退出
            if question.lower() in ["exit", "quit", "退出", "离开"]:
                print("\n感谢使用，再见！")
                break
            
            # 空输入
            if not question.strip():
                continue
            
            print("\n正在思考...")
            
            # 查询问题
            thinking, answer = self.query(question)
            
            # 输出思考过程（如果有）
            if thinking:
                print(f"\n思考过程: {thinking}")
            
            # 输出回答
            print(f"\n回答: {answer}")
