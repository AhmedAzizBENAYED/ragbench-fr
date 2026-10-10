"""Structured logging: structlog and stdlib logs share one renderer and go to stderr.

Named ``log`` rather than ``logging`` to avoid shadowing the standard library module.
"""

import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "console"]

REDACTED = "***"

# A key is sensitive when one of its segments (split on separators and camelCase) is listed here.
# Segments, not substrings: `tokens_in`, `max_tokens` or `keyword` are usage data, not secrets.
_SENSITIVE_SEGMENTS = frozenset({"key", "token", "secret", "password", "authorization"})
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_SEPARATORS = re.compile(r"[\W_]+")

_HANDLER_NAME = "ragbench"

# Third-party loggers that log every HTTP request, lock or model-loading step at INFO/DEBUG.
# They are raised to WARNING (or to the configured level if it is higher).
# `transformers` also installs its own handler: the embedder re-routes it when loading the model.
NOISY_LOGGERS = (
    "filelock",
    "httpcore",
    "httpcore2",
    "httpx",
    "httpx2",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "urllib3",
)


def is_sensitive_key(key: str) -> bool:
    """Return whether a log key looks like it holds a secret (e.g. `api_key`, `Authorization`)."""
    words = _SEPARATORS.split(_CAMEL_BOUNDARY.sub(r"\1_\2", key).lower())
    return any(word in _SENSITIVE_SEGMENTS for word in words)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            k: REDACTED if isinstance(k, str) and is_sensitive_key(k) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(item) for item in value]
    return value


def redact_secrets(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Mask values of sensitive keys, recursively. Defence in depth on top of `SecretStr`."""
    for key, value in event_dict.items():
        event_dict[key] = REDACTED if is_sensitive_key(key) else _redact(value)
    return event_dict


def _shared_processors() -> list[Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        redact_secrets,
    ]


def _renderer_chain(fmt: LogFormat) -> list[Processor]:
    if fmt == "json":
        # json.dumps escapes non-ASCII characters by default: safe in any Windows pipe encoding.
        return [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    return [structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())]


def configure_logging(level: LogLevel = "INFO", fmt: LogFormat = "json") -> None:
    """Route structlog and stdlib logs through one renderer on stderr.

    Idempotent: calling it again replaces the previous ragbench handler.
    """
    shared = _shared_processors()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Not cached, so that reconfiguring (CLI callback, tests) applies to existing loggers.
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                *_renderer_chain(fmt),
            ],
        )
    )

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    # Propagated records are not filtered by the root level, hence max(WARNING, level).
    noisy_level = max(logging.WARNING, logging.getLevelNamesMapping()[level])
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger named after its module (pass `__name__`)."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger


__all__ = [
    "REDACTED",
    "LogFormat",
    "LogLevel",
    "configure_logging",
    "get_logger",
    "is_sensitive_key",
    "redact_secrets",
]
