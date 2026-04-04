from __future__ import annotations

"""Logging helpers for structured console and file output.

This module centralises lightweight logging utilities used across the project.
It provides:

* logger setup with optional file output
* formatting helpers for key value log blocks
* compatibility handling for loggers that use brace style formatting
* manifest summarisation for exported plot artefacts
"""

import logging
from pathlib import Path
from typing import Any


# Third party libraries used for static image export can be very noisy at
# runtime, so keep them at WARNING unless explicitly debugging them.
NOISY_LOGGERS = [
    "kaleido",
    "kaleido.kaleido",
    "kaleido._kaleido_tab",
    "choreographer",
    "choreographer.browser_async",
    "choreographer.browsers.chromium",
    "choreographer.utils._tmpfile",
]


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configures the root logger for console and optional file logging.

    Args:
        level: Logging level name, such as ``"INFO"`` or ``"DEBUG"``.
        log_file: Optional file path for persisted logs. When provided, the
            parent directory is created automatically.
    """

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    # Reduce noise from known verbose dependency loggers.
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def format_kv_block(title: str, values: dict[str, Any]) -> list[str]:
    """Formats a title and mapping into a list of log friendly lines.

    Args:
        title: Heading line for the block.
        values: Mapping of keys and values to render as bullet style lines.

    Returns:
        A list of formatted strings, with the title first and one ``- key:
        value`` line per mapping entry after that.
    """

    lines = [title]
    for key, value in values.items():
        lines.append(f"- {key}: {value}")
    return lines


def _uses_brace_formatting(logger: Any) -> bool:
    """Returns whether the logger expects brace escaped message text.

    Some custom logger implementations interpret braces specially during
    formatting. This helper lets the module escape braces only when needed.

    Args:
        logger: Logger like object.

    Returns:
        ``True`` when the logger appears to use brace style formatting.
    """

    return logger.__class__.__name__ == "CustomLogger"


def _escape_braces(text: str) -> str:
    """Escapes braces so they are safe for brace style loggers.

    Args:
        text: Raw message text.

    Returns:
        Text with ``{`` and ``}`` doubled.
    """

    return text.replace("{", "{{").replace("}", "}}")


def log_kv_block(
    logger: Any,
    title: str,
    values: dict[str, Any],
    level: int = logging.INFO,
) -> None:
    """Logs a structured key value block line by line.

    This function is compatible with both standard loggers and the project's
    custom brace formatting logger implementation.

    Args:
        logger: Logger like object that exposes ``log(level, message)``.
        title: Heading line for the block.
        values: Mapping of values to log beneath the title.
        level: Logging level to use for each emitted line.
    """

    for line in format_kv_block(title, values):
        safe_line = _escape_braces(line) if _uses_brace_formatting(logger) else line
        logger.log(level, safe_line)


def summarise_plot_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    """Counts exported plot files recorded inside a nested manifest.

    The manifest may contain dictionaries and lists nested to arbitrary depth.
    This helper walks the full structure and counts string valued ``html``,
    ``png``, and ``eps`` entries.

    Args:
        manifest: Nested plot manifest dictionary.

    Returns:
        A dictionary containing counts for HTML, PNG, and EPS outputs.
    """

    summary = {"html_files": 0, "png_files": 0, "eps_files": 0}

    def walk(node: Any) -> None:
        """Recursively visits nested manifest nodes and updates counts.

        Args:
            node: Current manifest node, which may be a dictionary, list, or
                scalar value.
        """

        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"html", "png", "eps"} and isinstance(value, str):
                    summary[f"{key}_files"] += 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(manifest)
    return summary
