from loguru import logger


def get_logger():
    """Return a loguru instance."""
    logger.level("ERROR")
    logger.add(
        "logs/debug.log",
        level="DEBUG",
        filter=lambda record: record["level"].no
        == logger.level("DEBUG").no,  # debug level의 파일만 저장함
    )
    logger.add("logs/error.log", level="ERROR")
    return logger
