from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


NOISY_LOGGERS = [
    'kaleido',
    'kaleido.kaleido',
    'kaleido._kaleido_tab',
    'choreographer',
    'choreographer.browser_async',
    'choreographer.browsers.chromium',
    'choreographer.utils._tmpfile',
]


def setup_logging(level: str = 'INFO', log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=handlers,
        force=True,
    )

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def format_kv_block(title: str, values: dict[str, Any]) -> list[str]:
    lines = [title]
    for key, value in values.items():
        lines.append(f'- {key}: {value}')
    return lines


def _uses_brace_formatting(logger: Any) -> bool:
    return logger.__class__.__name__ == 'CustomLogger'


def _escape_braces(text: str) -> str:
    return text.replace('{', '{{').replace('}', '}}')


def log_kv_block(logger: Any, title: str, values: dict[str, Any], level: int = logging.INFO) -> None:
    for line in format_kv_block(title, values):
        safe_line = _escape_braces(line) if _uses_brace_formatting(logger) else line
        logger.log(level, safe_line)


def summarise_plot_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    summary = {'html_files': 0, 'png_files': 0, 'eps_files': 0}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {'html', 'png', 'eps'} and isinstance(value, str):
                    summary[f'{key}_files'] += 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(manifest)
    return summary
