from loguru import logger


class DuplicateFilter:
    def __init__(self):
        self.log_records = set()

    def __call__(self, record):
        msg = record["message"]
        if msg in self.log_records:
            return False
        self.log_records.add(msg)
        return True


def get_logger():
    """Return a loguru instance."""
    logger.level("ERROR")
    logger.add(
        "logs/debug.log",
        level="DEBUG",
        filter=lambda record: record["level"].no
        == logger.level("DEBUG").no,  # debug level의 파일만 저장함
    )
    logger.add("logs/error.log", level="ERROR", filter=DuplicateFilter())
    return logger
