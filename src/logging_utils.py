import logging
from pathlib import Path


def setup_logger(name: str, filename: str) -> logging.Logger:
    """
    Create a logger that writes to logs/<filename>.log.
    Avoids duplicate handlers if called multiple times.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(f"kmchat.{name}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logs_dir / f"{filename}.log", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
