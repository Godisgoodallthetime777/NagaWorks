"""Crash and diagnostic logging for NagaWorks STEP File Editor."""

from __future__ import annotations

import atexit
import faulthandler
import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "step_editor.log"
_LOGGER_NAME = "step_editor"
_logger: logging.Logger | None = None
_log_file_handle = None
_original_excepthook = sys.excepthook
_original_threading_excepthook = getattr(threading, "excepthook", None)


def get_log_path() -> Path:
    return _LOG_FILE


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        setup_logging()
    assert _logger is not None
    return _logger


def setup_logging() -> Path:
    """Configure file + console logging and crash hooks."""
    global _logger, _log_file_handle

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _logger = logger

    _log_file_handle = open(_LOG_FILE, "a", encoding="utf-8")
    faulthandler.enable(file=_log_file_handle, all_threads=True)

    sys.excepthook = _global_excepthook
    threading.excepthook = _thread_excepthook

    atexit.register(_shutdown_logging)

    logger.info("=" * 72)
    try:
        from app_info import APP_DISPLAY_NAME

        logger.info("%s started", APP_DISPLAY_NAME)
    except ImportError:
        logger.info("NagaWorks STEP File Editor started")
    logger.info("Python %s", sys.version.replace("\n", " "))
    logger.info("Platform %s", sys.platform)
    logger.info("Log file: %s", _LOG_FILE)
    logger.info("=" * 72)

    return _LOG_FILE


def _shutdown_logging() -> None:
    global _log_file_handle
    if _logger is not None:
        _logger.info("Application exiting")
    try:
        faulthandler.disable()
    except Exception:
        pass
    if _log_file_handle is not None:
        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None


def _write_exception(exc_type, exc_value, exc_tb, *, context: str) -> None:
    lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    text = "".join(lines)
    if _logger is not None:
        _logger.critical("%s\n%s", context, text)
    else:
        sys.stderr.write(f"{context}\n{text}\n")
    if _log_file_handle is not None:
        try:
            _log_file_handle.write(f"\n--- {context} {datetime.now().isoformat()} ---\n")
            _log_file_handle.write(text)
            _log_file_handle.flush()
        except Exception:
            pass


def _global_excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        _original_excepthook(exc_type, exc_value, exc_tb)
        return
    _write_exception(exc_type, exc_value, exc_tb, context="Uncaught exception")
    _original_excepthook(exc_type, exc_value, exc_tb)


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    _write_exception(args.exc_type, args.exc_value, args.exc_traceback, context="Thread exception")
    if _original_threading_excepthook is not None:
        _original_threading_excepthook(args)


def log_exception(context: str) -> None:
    """Log current exception with context (call from except block)."""
    get_logger().exception(context)


def safe_call(context: str, func, *args, default=None, **kwargs):
    """Run func and log failures instead of crashing."""
    try:
        return func(*args, **kwargs)
    except Exception:
        get_logger().exception("Failed: %s", context)
        return default
