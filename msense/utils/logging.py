import logging

DEFAULT_LOG_FORMAT = "%(name)s: - %(asctime)s - %(levelname)s - %(message)s"


def create_logger(name: str = "msense", level: int = logging.INFO, format: str = DEFAULT_LOG_FORMAT, filename: str = None, filemode: str = "a") -> logging.Logger:
    # Get logger and formatter
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)

    # Create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create file handler, if required
    if filename is not None:
        file_handler = logging.FileHandler(filename, filemode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
