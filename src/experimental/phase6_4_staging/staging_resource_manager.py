"""
Staging Resource Manager
========================
Provides resource protection, strict concurrency throttling, memory monitoring,
inference timeouts, and temporary file lifecycle management for CPU execution.
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple, Optional
import threading
import logging

from .staging_policy_config import MAX_CONCURRENT_INFERENCES, MAX_QUEUE_CAPACITY, DEFAULT_INFERENCE_TIMEOUT_SEC

logger = logging.getLogger("staging_resource_manager")

class QueueFullError(Exception):
    """Raised when the bounded waiting queue capacity is saturated."""
    pass

# Windows PSAPI structures for memory tracking
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class ResourceManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self._semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCES)
        self._queue_lock = threading.Lock()
        self._active_tasks = 0
        self._queued_tasks = 0

    @staticmethod
    def get_memory_usage_mb() -> Tuple[float, float]:
        """Returns (current_working_set_mb, peak_working_set_mb)."""
        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                current_mb = round(counters.WorkingSetSize / (1024 * 1024), 2)
                peak_mb = round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
                return current_mb, peak_mb
        except Exception as e:
            logger.warning(f"Failed to query Windows PSAPI memory: {e}")
        return 0.0, 0.0

    @contextmanager
    def execution_guard(self, timeout_sec: float = DEFAULT_INFERENCE_TIMEOUT_SEC) -> Generator[float, None, None]:
        """
        Guarantees bounded concurrency and records wait time.
        Raises QueueFullError if queue capacity is exceeded.
        Raises TimeoutError if execution exceeds timeout.
        """
        with self._queue_lock:
            if self._queued_tasks >= MAX_QUEUE_CAPACITY:
                raise QueueFullError(
                    f"Inference queue capacity saturated: current queue length ({self._queued_tasks}) >= limit ({MAX_QUEUE_CAPACITY})"
                )
            self._queued_tasks += 1

        t_queue_start = time.perf_counter()
        acquired = self._semaphore.acquire(timeout=timeout_sec)
        t_queue_wait = time.perf_counter() - t_queue_start

        with self._queue_lock:
            self._queued_tasks -= 1
            if acquired:
                self._active_tasks += 1

        if not acquired:
            raise TimeoutError(f"Inference queue saturated: could not acquire worker within {timeout_sec}s")

        try:
            yield t_queue_wait
        finally:
            with self._queue_lock:
                self._active_tasks -= 1
            self._semaphore.release()

    @contextmanager
    def safe_temp_file(self, prefix: str = "staging_tmp_", suffix: str = ".mp4") -> Generator[Path, None, None]:
        """Creates a managed temporary file and guarantees its deletion upon exit."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            yield tmp_path
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

    def get_status(self) -> dict:
        cur_mb, peak_mb = self.get_memory_usage_mb()
        with self._queue_lock:
            return {
                "active_tasks": self._active_tasks,
                "queued_tasks": self._queued_tasks,
                "max_concurrency": MAX_CONCURRENT_INFERENCES,
                "max_queue_capacity": MAX_QUEUE_CAPACITY,
                "current_ram_mb": cur_mb,
                "peak_ram_mb": peak_mb
            }
