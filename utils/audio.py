"""
Audio processing utilities for the Conversify system.
"""

import time
import logging
from contextlib import contextmanager

@contextmanager
def find_time(label: str):
    """A context manager for timing code execution.
    
    Args:
        label: A descriptive name for the operation being timed.
        
    Example:
        with find_time('STT_inference'):
            result = model.transcribe(audio)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        logging.debug(f"{label} took {elapsed_ms:.4f} ms") 