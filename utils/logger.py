# logger.py

import logging

class Logger:
    @staticmethod
    def get_logger(name):
        # 创建一个日志记录器
        logger = logging.getLogger(name)
        # 设置日志级别
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器并设置级别
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # 如果日志记录器没有处理器，则添加处理器
        if not logger.handlers:
            logger.addHandler(ch)
        
        return logger
