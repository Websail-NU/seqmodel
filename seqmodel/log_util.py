import logging as py_logging


def _check_file_handler(logger, filepath):
    for handler in logger.handlers:
        if isinstance(handler, py_logging.FileHandler):
            handler.baseFilename
            return handler.baseFilename == ps.path.abspath(filepath)
    return False


def get_logger(log_file_path=None, name="exp"):
    root_logger = py_logging.getLogger(name)
    handlers = root_logger.handlers
    if (log_file_path is not None and not
            _check_file_handler(root_logger, log_file_path)):
        log_formatter = py_logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] %(message)s",
            datefmt='%Y/%m/%d %H:%M:%S')
        file_handler = py_logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    # time_format = "\x1b[36m%(asctime)s\x1b[0m"
    # level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    # log_formatter = py_logging.Formatter(
    #     "{} {} %(message)s".format(time_format, level_format),
    #     datefmt='%Y/%m/%d %H:%M:%S')
    if any([type(h) == py_logging.StreamHandler for h in handlers]):
        return root_logger
    level_format = "\x1b[36m[%(levelname)-5.5s]\x1b[0m"
    log_formatter = py_logging.Formatter(
        "{}%(message)s".format(level_format))
    console_handler = py_logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(py_logging.INFO)
    return root_logger
