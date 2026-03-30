from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PLOT_FIELDS = [
    'v2_id',
    'v1_av',
    'move_v1',
    'collision_type',
    'av_guilty',
    'main_factor',
]

DEFAULT_HISTOGRAM_FIELDS = [
    'av_manufacturer', 'av_make', 'av_model', 'vehicle_was', 'damage',
    'v2_id', 'v2_mov', 'v1_av', 'v1_move', 'v2_move', 'direction',
    'weather_v1', 'weather_v2', 'light_v1', 'light_v2',
    'surface_v1', 'surface_v2', 'condition_v1', 'condition_v2',
    'collision_type', 'av_guilty', 'main_factor', 'other_factor',
]

VALID_ROW_KEEP_POLICIES = {'output_only', 'best_available'}


@dataclass(slots=True)
class RuntimeConfig:
    config_path: Path
    input_csv: Path
    output_dir: Path
    figures_dir: Path
    text_column: str | None
    log_level: str
    auto_open_html: bool
    save_final: bool
    filter_rows_with_na: bool
    na_filter_fields: list[str]
    include_plot_fields: list[str]
    exclude_plot_fields: list[str]
    histogram_fields: list[str]
    min_count: int
    max_categories: int
    row_keep_policy: str


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _coerce_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return list(default)
        if value.startswith('['):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(',') if item.strip()]
    return list(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_row_keep_policy(value: Any, default: str = 'output_only') -> str:
    text = str(value or default).strip().lower()
    return text if text in VALID_ROW_KEEP_POLICIES else default


def _search_config_file(start: Path) -> Path:
    candidates = []
    for base in [start, *start.parents]:
        candidates.extend([base / 'config', base / 'default.config'])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not find config or default.config in current directory or parents.')


def _load_jsonish_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import ast
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, dict):
            raise ValueError(f'Config at {path} does not contain a dictionary.')
        return parsed


def _looks_like_container_path(path: Path) -> bool:
    return str(path).startswith('/mnt/') or str(path).startswith('/tmp/')


def _resolve_input_csv(raw_value: Any, project_root: Path) -> Path:
    if raw_value is None:
        raise ValueError('Config must define "data" or pass --input.')

    raw_text = str(raw_value).strip()
    raw_path = Path(raw_text).expanduser()

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((project_root / raw_path).resolve())
        candidates.append(raw_path.resolve())

    filename = raw_path.name or 'Output.csv'
    candidates.extend([
        (project_root / '_output' / filename).resolve(),
        (project_root / '_output' / 'Output.csv').resolve(),
        (project_root / filename).resolve(),
    ])

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    resolved_raw = raw_path.resolve() if not raw_path.is_absolute() else raw_path
    if _looks_like_container_path(resolved_raw):
        return (project_root / '_output' / 'Output.csv').resolve()
    return resolved_raw.resolve() if not resolved_raw.is_absolute() else resolved_raw


def _resolve_output_dir(raw_value: Any, project_root: Path) -> Path:
    default_output = (project_root / '_output').resolve()
    if raw_value is None:
        return default_output

    raw_path = Path(str(raw_value).strip()).expanduser()
    if not raw_path.is_absolute():
        return (project_root / raw_path).resolve()
    if _looks_like_container_path(raw_path):
        return default_output
    return raw_path.resolve()


def _resolve_figures_dir(raw_value: Any, project_root: Path) -> Path:
    default_figures = (project_root / 'figures').resolve()
    if raw_value is None:
        return default_figures

    raw_path = Path(str(raw_value).strip()).expanduser()
    if not raw_path.is_absolute():
        return (project_root / raw_path).resolve()
    if _looks_like_container_path(raw_path):
        return default_figures
    return raw_path.resolve()


def load_runtime_config() -> RuntimeConfig:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--input', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--text-column', default=None)
    parser.add_argument('--min-count', default=None)
    parser.add_argument('--max-categories', default=None)
    parser.add_argument('--row-keep-policy', default=None, choices=sorted(VALID_ROW_KEEP_POLICIES))
    parser.add_argument('--open-html', action='store_true', default=None)
    parser.add_argument('--no-open-html', action='store_true', default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else _search_config_file(Path.cwd())
    config_data = _load_jsonish_config(config_path)
    project_root = config_path.parent.resolve()

    input_csv_raw = args.input or config_data.get('data') or config_data.get('input_csv')
    input_csv = _resolve_input_csv(input_csv_raw, project_root)

    output_dir_raw = args.output_dir or config_data.get('output_dir')
    output_dir = _resolve_output_dir(output_dir_raw, project_root)
    figures_dir = _resolve_figures_dir(config_data.get('figures_dir'), project_root)

    if args.open_html:
        auto_open_html = True
    elif args.no_open_html:
        auto_open_html = False
    else:
        auto_open_html = _coerce_bool(config_data.get('auto_open_html'), True)

    min_count = _coerce_int(args.min_count if args.min_count is not None else config_data.get('min_count'), 1)
    max_categories = _coerce_int(args.max_categories if args.max_categories is not None else config_data.get('max_categories'), 20)
    row_keep_policy = _coerce_row_keep_policy(args.row_keep_policy or config_data.get('row_keep_policy'), 'output_only')

    return RuntimeConfig(
        config_path=config_path,
        input_csv=input_csv,
        output_dir=output_dir,
        figures_dir=figures_dir,
        text_column=args.text_column or config_data.get('text_column') or config_data.get('preferred_text_column'),
        log_level=str(config_data.get('logger_level', 'INFO')).upper(),
        auto_open_html=auto_open_html,
        save_final=_coerce_bool(config_data.get('save_final'), False),
        filter_rows_with_na=_coerce_bool(config_data.get('filter_rows_with_na'), False),
        na_filter_fields=_coerce_list(config_data.get('na_filter_fields'), DEFAULT_PLOT_FIELDS),
        include_plot_fields=_coerce_list(config_data.get('include_plot_fields'), DEFAULT_PLOT_FIELDS),
        exclude_plot_fields=_coerce_list(config_data.get('exclude_plot_fields'), []),
        histogram_fields=_coerce_list(config_data.get('histogram_fields'), DEFAULT_HISTOGRAM_FIELDS),
        min_count=min_count,
        max_categories=max_categories,
        row_keep_policy=row_keep_policy,
    )
