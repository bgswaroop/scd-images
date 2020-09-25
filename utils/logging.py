import logging
import sys


def singleton(cls):
    class_instances = {}

    def inner(*args, **kwargs):
        if cls not in class_instances:
            class_instances[cls] = cls(*args, **kwargs)
        return class_instances[cls]

    return inner


@singleton
class SetupLogger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.setup_logger()

    def setup_logger(self):
        # root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_file)
        sys.stdout = open(self.log_file, 'a')
        sys.stderr = open(self.log_file, 'a')

        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Log messages to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        logger.info('Logger file successfully initialized')


def log_running_time(func):
    """
    Decorator to log the running time of any function
    :param func: Any function
    :return: the output of the function

    """
    from functools import wraps
    import logging
    from time import perf_counter
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def inner(*args, **kwargs):
        logger.info(f'Started running {func.__name__}')
        start = perf_counter()
        output = func(*args, **kwargs)
        end = perf_counter()
        time_in_sec = end - start
        if time_in_sec < 60:
            logger.info(f"Finished running {func.__name__} in {format(time_in_sec, '.2f')} sec")
        else:
            logger.info(f"Finished running {func.__name__} in {format(time_in_sec / 60., '.2f')} min")
        return output

    return inner
