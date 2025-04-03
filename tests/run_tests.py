"""
测试运行脚本

该脚本用于运行项目的单元测试
"""
import unittest
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_tests():
    """运行所有测试"""
    logger.info("运行所有测试...")
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return result.wasSuccessful()

def run_db_tests():
    """运行数据库相关测试"""
    logger.info("运行数据库相关测试...")
    test_suite = unittest.defaultTestLoader.discover('./db', pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == 'db':
            success = run_db_tests()
        else:
            logger.error(f"未知的测试类型: {test_type}")
            sys.exit(1)
    else:
        # 默认运行所有测试
        success = run_all_tests()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)
