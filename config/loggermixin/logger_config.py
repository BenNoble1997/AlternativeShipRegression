"""
This script defines the LoggerMixin class,
 and will store other Logging configurations as the project develops.

Usage:
from logger_mixin import LoggerMixin
Your_Class(LoggerMixin)
"""
import logging
import sys
from colorama import Fore, Style

"""
This script defines the LoggerMixin class,
 and will store other Logging configurations as the project develops.

Usage:
from logger_mixin import LoggerMixin
Your_Class(LoggerMixin)
"""
import logging
import copy
from colorama import Fore, Style

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        record = copy.copy(record)
        colour = LoggerMixin.LOG_COLOUR.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


class LoggerMixin:

    LOG_LEVEL = logging.INFO

    LOG_COLOUR = {
        "DEBUG": Fore.GREEN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT
    }

    def set_log_level(self, verbose=False, debug=False):
        if debug:
            self.logger.setLevel(logging.DEBUG)
        elif verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def logger(self):
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        logger = logging.getLogger(logger_name)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                ColoredFormatter(
                    fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)-15s | %(message)s',
                    datefmt='%H:%M:%S'
                )
            )
            logger.addHandler(handler)
            logger.propagate = False

        logger.setLevel(self.LOG_LEVEL)

        return logger