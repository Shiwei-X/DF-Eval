import sys
import logging
from logging.handlers import RotatingFileHandler


# 重定向 stdout 和 stderr 到日志
class StdOutWrapper:
    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())
        self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, attr):
        return getattr(self.original_stream, attr)


def setup_logger(log_file_path):
    logger = logging.getLogger('stdout_stderr_logger')
    logger.setLevel(logging.DEBUG)

    # 创建 RotatingFileHandler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("[%(asctime)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 重定向 stdout 和 stderr
    sys.stdout = StdOutWrapper(logger, logging.INFO, sys.stdout)
    sys.stderr = StdOutWrapper(logger, logging.ERROR, sys.stderr)

    return logger



# def main():
#     #####
#     setup_logger('/home/lijuren/transfer/temp/test_logger.log')
#     #####
    
#     print('hello')
#     a = 1
#     if a == 1:
#         print('a is 1')
#         raise Exception('a is 1')

# if __name__ == '__main__':
#     main()