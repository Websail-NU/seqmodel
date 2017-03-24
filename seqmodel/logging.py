import logging


def get_logger(log_file_path=None, name="exp"):
    root_logger = logging.getLogger(name)
    if log_file_path is not None:
        log_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt='%Y/%m/%d %H:%M:%S')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    time_format = "\x1b[36m%(asctime)s\x1b[0m"
    level_format = "\x1b[94m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = logging.Formatter(
        "{} {} %(message)s".format(time_format, level_format),
        datefmt='%Y/%m/%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger
