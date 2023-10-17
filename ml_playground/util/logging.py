import logging
import sys
from typing import IO

from ml_playground.color import BOLD_BRIGHT_RED, CYAN, GREY, RED, RESET, YELLOW


class ColoredLogFormatter(logging.Formatter):
    DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(message)s  (%(name)s:%(lineno)d)"
    DEFAULT_DATE_FMT = "%Y%m%d-%H:%M:%S"

    COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_BRIGHT_RED,
    }

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt or self.DEFAULT_FMT, datefmt or self.DEFAULT_DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        return self.COLORS[record.levelno] + s + RESET


def create_colored_handler(
    stream: IO | None = None,
    *,
    fmt: str | None = None,
    datefmt: str | None = None,
) -> logging.Handler:
    """デフォルトでは sys.stderr に書き出す"""
    h = logging.StreamHandler(stream or sys.stderr)
    h.setFormatter(ColoredLogFormatter(fmt=fmt, datefmt=datefmt))
    return h
