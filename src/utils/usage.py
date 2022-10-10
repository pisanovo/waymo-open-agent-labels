import logging
import os

import psutil

logger = logging.getLogger(__name__)


def print_mem_usage():
    size_name = "MB"

    mem_used = round(get_mem_usage_mb())
    if mem_used >= 1000:
        mem_used = round(mem_used / 1000, 3)
        size_name = "GB"

    logger.info(f"Memory usage: {mem_used} {size_name}")


def get_mem_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
