# src/uavlab/common/logger.py
''''''
import logging
import os

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    设置日志记录器，日志会输出到控制台和文件中。
    :param log_dir: 存储日志文件的目录
    :param log_level: 设置日志级别
    :return: logger 对象
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "experiment.log")

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台日志处理
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件日志处理
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger