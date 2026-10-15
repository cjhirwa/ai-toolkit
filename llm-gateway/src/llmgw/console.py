import logging
import sys

# Libraries that log one INFO line per HTTP request. Matched by prefix, because the
# names change between versions (the OpenAI SDK has logged as both "httpx" and "httpx2").
_CHATTY = ("httpx", "httpcore", "google_genai", "openai")


class QuietHTTPFilter(logging.Filter):
    """Drops records below WARNING from HTTP client libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not record.name.startswith(_CHATTY)


def utf8_stdio() -> None:
    """Make CLI output safe on Windows consoles (cp1252 can't print →, ≥, or most document text)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)  # absent when output is redirected to some objects
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:
            pass


def cli_logging(level: int = logging.INFO) -> None:
    """Log the toolkit's own messages at ``level`` but keep HTTP client libraries to
    warnings, so a run with hundreds of model calls does not print a line per request."""
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, QuietHTTPFilter) for f in handler.filters):
            handler.addFilter(QuietHTTPFilter())
