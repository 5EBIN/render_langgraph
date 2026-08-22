"""Structured terminal logging for the render-langgraph CLI process.

Quiet by default (INFO phase markers + OK/WARN/ERROR), --verbose reveals
DEBUG (paths, candidate lists, timings, full tracebacks), --quiet drops to
WARN+ERROR only. Failures are always loud regardless of verbosity -- that's
just standard level ordering (ERROR > WARNING > INFO > DEBUG), not special
casing: --quiet raises the threshold to WARNING, which still lets ERROR
through.

Built on the stdlib `logging` module (one process-wide logger named
"render-langgraph"), wrapped in a tiny `Log` facade so call sites read as
`log.ok(...)` / `log.warn(...)` instead of juggling `extra={"tag": ...}`
everywhere. Any module can call `get_logger()` and it transparently shares
whatever handler/level `setup_logging()` installed at startup, because
`logging.getLogger(name)` returns the same singleton by name.
"""
from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "render-langgraph"

_RESET = "\033[0m"
_COLORS = {
    "OK": "\033[32m",  # green
    "WARN": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
}


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class _PrefixFormatter(logging.Formatter):
    """Prefixes every line (not just the first, for multi-line banners) with
    "render-langgraph: " and colors OK/WARN/ERROR when the stream is a TTY."""

    def __init__(self, use_color: bool):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None) or record.levelname
        message = record.getMessage()
        color = _COLORS.get(tag) if self.use_color else None

        lines = message.split("\n")
        prefixed = "\n".join(f"render-langgraph: {line}" for line in lines)

        if record.exc_info:
            prefixed += "\n" + self.formatException(record.exc_info)

        if color:
            return f"{color}{prefixed}{_RESET}"
        return prefixed


class Log:
    """Thin facade over `logging.Logger` giving named severities that map
    onto stdlib levels: DEBUG->DEBUG, INFO/OK->INFO, WARN->WARNING,
    ERROR->ERROR. OK is visually distinct (green) but not a separate
    logging level -- it's still filtered by the INFO threshold."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, extra={"tag": "DEBUG"}, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, extra={"tag": "INFO"}, **kwargs)

    def ok(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, extra={"tag": "OK"}, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, extra={"tag": "WARN"}, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, extra={"tag": "ERROR"}, **kwargs)

    @property
    def raw(self) -> logging.Logger:
        return self._logger


def setup_logging(verbose: bool = False, quiet: bool = False, stream=None) -> Log:
    """Configure the process-wide "render-langgraph" logger. Call once, at the CLI
    entrypoint. Every other module gets the same configuration for free via
    get_logger() (logging.getLogger caches by name)."""
    if verbose and quiet:
        # Caller's responsibility to prevent this via argparse mutual
        # exclusion; if it happens anyway, verbose wins -- showing more on
        # request is safer than silently hiding it.
        quiet = False

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    stream = stream or sys.stderr
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(_PrefixFormatter(use_color=_supports_color(stream)))
    logger.addHandler(handler)

    return Log(logger)


def get_logger() -> Log:
    """Get the shared logger from any module without reconfiguring it.
    Safe to call before setup_logging() too (default level is WARNING,
    same as the stdlib default) -- useful for library-style callers/tests
    that never call setup_logging() themselves."""
    return Log(logging.getLogger(LOGGER_NAME))
