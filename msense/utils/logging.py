import logging

DEFAULT_LOG_FORMAT = "%(name)s: - %(asctime)s - %(levelname)s - %(message)s"


class CustomFormatter(logging.Formatter):

    def __init__(self, fmt) -> None:
        super().__init__(fmt)

        blue = "\x1B[0;34;49m"
        cyan = "\x1B[0;36;49m"
        yellow = "\x1B[0;33;49m"
        red = "\x1B[0;31;49m"
        reset = "\x1B[0m"

        self.formats = {
            logging.DEBUG: blue + fmt + reset,
            logging.INFO: cyan + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: red + fmt + reset}

    def format(self, record):
        log_fmt = self.formats[record.levelno]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(name: str = "msense",
                  level: int = logging.INFO,
                  format: str = DEFAULT_LOG_FORMAT,
                  filename: str = None,
                  filemode: str = "a") -> logging.Logger:
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(CustomFormatter(format))
    logger.addHandler(stream_handler)

    # Create file handler, if required
    if filename is not None:
        file_handler = logging.FileHandler(filename, filemode)
        file_handler.setLevel(level)
        file_handler.setFormatter(CustomFormatter(format))
        logger.addHandler(file_handler)

    return logger
