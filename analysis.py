from __future__ import annotations

"""Main analysis entrypoint for the accident reporting pipeline.

This module orchestrates the full workflow:

* loading runtime configuration from the shared config source
* loading and parsing input event records
* deriving research columns and filtered empirical subsets
* exporting tabular outputs and Markdown summaries
* generating all plots and writing a plot manifest
* logging key descriptive and interpretive results
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import common
from custom_logger import CustomLogger
from logmod import logs

from utils.io import (
    ensure_output_dirs,
    load_input_events,
    save_dataframe,
    save_json,
    save_markdown,
)
from utils.logging_utils import log_kv_block, summarise_plot_manifest
from utils.parsing import parse_events_dataframe
from utils.research import (
    build_research_summary,
    create_validation_sample,
    derive_research_columns,
    export_research_tables,
    format_research_markdown,
)
from utils.sankey import apply_plot_filters, build_overview_summary, resolve_plot_fields
from utils.summary_plots import create_all_plots

DEFAULT_PLOT_FIELDS = [
    'road_user_type',
    'av_mode_group',
    'av_movement_group',
    'collision_group',
    'blame_group',
    'scenario_class',
]

DEFAULT_5W1H_PLOT_FIELDS = [
    'who_group',
    'where_group',
    'what_group',
    'when_group',
    'why_group',
    # 'how_group',
]

DEFAULT_HISTOGRAM_FIELDS = [
    'road_user_type',
    'av_mode_group',
    'av_movement_group',
    'other_party_movement_group',
    'collision_group',
    'blame_group',
    'scenario_class',
    'report_completeness_band',
    'weather_v1',
    'light_v1',
    'surface_v1',
    'condition_v1',
]

DEFAULT_BLIND_SPOT_FIELDS = [
    'v1_lane',
    'v2_lane',
    'v1_speed',
    'v2_speed',
    'v1_intersection',
    'v2_intersection',
    'direction',
    'lane_number',
    'street_type',
    'street_busy',
    'q0_confidence',
    'v1_damage_desc',
    'v2_damage_desc',
]

VALID_ROW_KEEP_POLICIES = {'output_only', 'best_available', 'best_per_row'}

logger = CustomLogger(__name__)


@dataclass(slots=True)
class RuntimeConfig:
    """Resolved runtime configuration for a pipeline run.

    Attributes:
        config_source: High level source label for the configuration.
        config_path: Path to the config file or config root.
        input_csv: Input CSV file containing model outputs.
        output_dir: Directory for generated tables and figures.
        figures_dir: Directory for final copied figure exports.
        text_column: Optional preferred text column to parse.
        log_level: Logging level name.
        auto_open_html: Whether generated HTML plots should open automatically.
        save_final: Whether figures should also be copied to final directories.
        filter_rows_with_na: Whether rows missing critical plot fields are
            excluded from plot specific analyses.
        na_filter_fields: Fields used when filtering missing values for plots.
        include_plot_fields: Candidate plot fields in preferred order.
        exclude_plot_fields: Plot fields to exclude after inclusion.
        histogram_fields: Fields for which categorical histograms are built.
        blind_spot_fields: Fields used in blind spot analyses.
        min_count: Minimum Sankey edge count.
        max_categories: Maximum Sankey categories per stage.
        row_keep_policy: Policy controlling which text rows are retained.
        validation_sample_size: Validation sample size.
        validation_seed: Random seed for validation sampling.
        validation_include_text: Whether validation exports include raw text.
        paper_plot_top_n: Top N parameter for selected research figures.
        image_export_timeout_seconds: Timeout for static figure export.
    """

    config_source: str
    config_path: str
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
    blind_spot_fields: list[str]
    min_count: int
    max_categories: int
    row_keep_policy: str
    validation_sample_size: int
    validation_seed: int
    validation_include_text: bool
    paper_plot_top_n: int
    image_export_timeout_seconds: int


def _get_common_config(*keys: str, default: Any = None) -> Any:
    """Returns the first non null config value found for the given keys.

    Args:
        *keys: Candidate config keys to try in order.
        default: Fallback value when no key resolves successfully.

    Returns:
        The first resolved config value, or ``default`` when unavailable.
    """

    for key in keys:
        try:
            value = common.get_configs(key)
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerces a loosely typed config value to ``bool``.

    Args:
        value: Raw config value.
        default: Fallback value when parsing fails.

    Returns:
        A boolean value.
    """

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


def _coerce_int(value: Any, default: int) -> int:
    """Coerces a config value to ``int`` with a safe fallback.

    Args:
        value: Raw config value.
        default: Fallback integer when conversion fails.

    Returns:
        The converted integer or ``default``.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_list(value: Any, default: list[str]) -> list[str]:
    """Coerces a config value into a list of non empty strings.

    Supported inputs include lists, tuples, JSON list strings, and comma
    separated strings.

    Args:
        value: Raw config value.
        default: Fallback list when ``value`` is missing or unusable.

    Returns:
        A list of cleaned strings.
    """

    if value is None:
        return list(default)

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default)

        if text.startswith('['):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]

        return [item.strip() for item in text.split(',') if item.strip()]

    return list(default)


def _coerce_row_keep_policy(
    value: Any,
    default: str = 'best_per_row',
) -> str:
    """Normalises the configured row keep policy.

    Args:
        value: Raw config value.
        default: Fallback policy.

    Returns:
        A valid row keep policy.
    """

    text = str(value or default).strip().lower()
    return text if text in VALID_ROW_KEEP_POLICIES else default


def _resolve_project_root() -> Path:
    """Resolves the project root directory.

    The resolution order prefers explicit config values, then ``common.root_dir``,
    and finally the current working directory.

    Returns:
        The resolved project root path.
    """

    explicit_root = _get_common_config('project_root', 'repo_root', default=None)
    if explicit_root:
        try:
            return Path(str(explicit_root)).expanduser().resolve()
        except Exception:
            pass

    root_dir = getattr(common, 'root_dir', None)
    if root_dir:
        try:
            return Path(str(root_dir)).expanduser().resolve()
        except Exception:
            pass

    return Path.cwd().resolve()


def _looks_like_container_path(path: Path) -> bool:
    """Returns whether a path appears to point into a transient container.

    Args:
        path: Candidate path.

    Returns:
        ``True`` when the path starts with common container mount prefixes.
    """

    text = str(path)
    return text.startswith('/mnt/') or text.startswith('/tmp/')


def _resolve_input_csv(project_root: Path) -> Path:
    """Resolves the input CSV path from config and sensible fallbacks.

    Args:
        project_root: Resolved project root directory.

    Returns:
        The input CSV path to use.
    """

    raw_value = _get_common_config('data', 'input_csv', default=None)
    default_path = (project_root / '_output' / 'Output.csv').resolve()

    if raw_value is None:
        return default_path

    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_path

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

    # Preserve candidate order while removing duplicates.
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
        return default_path
    return resolved_raw.resolve() if not resolved_raw.is_absolute() else resolved_raw


def _resolve_output_dir(project_root: Path) -> Path:
    """Resolves the primary output directory.

    Args:
        project_root: Resolved project root directory.

    Returns:
        The output directory path.
    """

    default_output = (project_root / '_output').resolve()
    raw_value = _get_common_config('output_dir', default=None)
    if raw_value is None:
        return default_output

    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_output

    raw_path = Path(raw_text).expanduser()
    if not raw_path.is_absolute():
        return (project_root / raw_path).resolve()
    if _looks_like_container_path(raw_path):
        return default_output
    return raw_path.resolve()


def _resolve_figures_dir(project_root: Path) -> Path:
    """Resolves the final figures directory.

    Args:
        project_root: Resolved project root directory.

    Returns:
        The figures directory path.
    """

    default_figures = (project_root / 'figures').resolve()
    raw_value = _get_common_config('figures_dir', default=None)
    if raw_value is None:
        return default_figures

    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_figures

    raw_path = Path(raw_text).expanduser()
    if not raw_path.is_absolute():
        return (project_root / raw_path).resolve()
    if _looks_like_container_path(raw_path):
        return default_figures
    return raw_path.resolve()


def load_runtime_config_from_common() -> RuntimeConfig:
    """Loads and resolves runtime configuration from the shared config source.

    Returns:
        A fully resolved ``RuntimeConfig`` instance.
    """

    project_root = _resolve_project_root()
    input_csv = _resolve_input_csv(project_root)
    output_dir = _resolve_output_dir(project_root)
    figures_dir = _resolve_figures_dir(project_root)
    config_path = Path(project_root) / 'config'

    return RuntimeConfig(
        config_source='common',
        config_path=str(config_path),
        input_csv=input_csv,
        output_dir=output_dir,
        figures_dir=figures_dir,
        text_column=_get_common_config(
            'text_column',
            'preferred_text_column',
            default=None,
        ),
        log_level=str(_get_common_config('logger_level', default='INFO')).upper(),
        auto_open_html=_coerce_bool(
            _get_common_config('auto_open_html', default=False),
            False,
        ),
        save_final=_coerce_bool(
            _get_common_config('save_final', default=True),
            True,
        ),
        filter_rows_with_na=_coerce_bool(
            _get_common_config('filter_rows_with_na', default=True),
            True,
        ),
        na_filter_fields=_coerce_list(
            _get_common_config('na_filter_fields', default=None),
            DEFAULT_PLOT_FIELDS,
        ),
        include_plot_fields=_coerce_list(
            _get_common_config('include_plot_fields', default=None),
            DEFAULT_PLOT_FIELDS,
        ),
        exclude_plot_fields=_coerce_list(
            _get_common_config('exclude_plot_fields', default=None),
            [],
        ),
        histogram_fields=_coerce_list(
            _get_common_config('histogram_fields', default=None),
            DEFAULT_HISTOGRAM_FIELDS,
        ),
        blind_spot_fields=_coerce_list(
            _get_common_config('blind_spot_fields', default=None),
            DEFAULT_BLIND_SPOT_FIELDS,
        ),
        min_count=_coerce_int(_get_common_config('min_count', default=1), 1),
        max_categories=_coerce_int(
            _get_common_config('max_categories', default=20),
            20,
        ),
        row_keep_policy=_coerce_row_keep_policy(
            _get_common_config('row_keep_policy', default='best_per_row'),
            'best_per_row',
        ),
        validation_sample_size=_coerce_int(
            _get_common_config('validation_sample_size', default=100),
            100,
        ),
        validation_seed=_coerce_int(
            _get_common_config('validation_seed', default=42),
            42,
        ),
        validation_include_text=_coerce_bool(
            _get_common_config('validation_include_text', default=True),
            True,
        ),
        paper_plot_top_n=_coerce_int(
            _get_common_config('paper_plot_top_n', default=10),
            10,
        ),
        image_export_timeout_seconds=_coerce_int(
            _get_common_config('image_export_timeout_seconds', default=60),
            60,
        ),
    )


def _safe_total(mapping: dict[str, int]) -> int:
    """Sums integer like values from a mapping safely.

    Args:
        mapping: Mapping of string keys to integer like values.

    Returns:
        The summed integer total, or ``0`` when the mapping is empty.
    """

    return int(sum(int(v) for v in mapping.values())) if mapping else 0


def _is_missing_text(value: Any) -> bool:
    """Returns whether a loosely typed value should be treated as missing."""

    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return True

    return text.lower() in {
        'na',
        'n/a',
        'none',
        'null',
        'nan',
        'unknown',
        'not specified',
    }



def _first_non_missing_value(row: pd.Series, columns: list[str]) -> str | None:
    """Returns the first present value across the provided columns for one row."""

    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if not _is_missing_text(value):
            return str(value).strip()
    return None



def _infer_company_from_model(value: Any) -> str | None:
    """Infers a probable company or make from a free text vehicle model string."""

    if _is_missing_text(value):
        return None

    text = str(value).strip()
    text = re.sub(r'^\d{4}\s+', '', text)
    tokens = re.findall(r'[A-Za-z][A-Za-z0-9&.-]*', text)
    if not tokens:
        return None
    return tokens[0]



def _build_report_date_log(df: pd.DataFrame) -> dict[str, Any]:
    """Builds a logger friendly summary of report dates from oldest to newest."""

    required = {'accident_year', 'accident_month', 'accident_day'}
    if not required.issubset(df.columns):
        return {
            'dated_reports': 0,
            'undated_reports': int(len(df)),
            'oldest_report_date': 'NA',
            'newest_report_date': 'NA',
            'report_dates_oldest_to_newest': [],
        }

    working = df.copy()
    working['accident_year_num'] = pd.to_numeric(
        working['accident_year'],
        errors='coerce',
    )
    working['accident_month_num'] = pd.to_numeric(
        working['accident_month'],
        errors='coerce',
    )
    working['accident_day_num'] = pd.to_numeric(
        working['accident_day'],
        errors='coerce',
    )

    working['report_date'] = pd.to_datetime(
        {
            'year': working['accident_year_num'],
            'month': working['accident_month_num'],
            'day': working['accident_day_num'],
        },
        errors='coerce',
    )

    dated = (
        working.loc[working['report_date'].notna()]
        .sort_values('report_date')
        .reset_index(drop=True)
    )

    if dated.empty:
        return {
            'dated_reports': 0,
            'undated_reports': int(len(df)),
            'oldest_report_date': 'NA',
            'newest_report_date': 'NA',
            'report_dates_oldest_to_newest': [],
        }

    ordered_unique_dates = [
        ts.strftime('%Y-%m-%d')
        for ts in dated['report_date'].drop_duplicates().tolist()
    ]

    return {
        'dated_reports': int(len(dated)),
        'undated_reports': int(len(df) - len(dated)),
        'oldest_report_date': ordered_unique_dates[0],
        'newest_report_date': ordered_unique_dates[-1],
        'report_dates_oldest_to_newest': ordered_unique_dates,
    }



def _summarise_company_values(values: pd.Series, top_n: int = 10) -> dict[str, int]:
    """Builds a compact count summary for a company or make series."""

    cleaned = values.map(
        lambda value: None if _is_missing_text(value) else str(value).strip()
    ).dropna()

    if cleaned.empty:
        return {}

    return {
        str(key): int(value)
        for key, value in cleaned.value_counts().head(top_n).items()
    }



def _build_company_log(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Builds a logger friendly summary of AV and other vehicle companies."""

    working = df.copy()

    av_candidates = [
        'av_company',
        'v1_company',
        'av_manufacturer',
        'av_make',
    ]
    v2_candidates = [
        'v2_company',
        'v2_make',
    ]

    working['av_company_log'] = working.apply(
        lambda row: _first_non_missing_value(row, av_candidates),
        axis=1,
    )
    working['v2_company_log'] = working.apply(
        lambda row: _first_non_missing_value(row, v2_candidates),
        axis=1,
    )

    if 'v2_model' in working.columns:
        inferred_v2_company = working['v2_model'].map(_infer_company_from_model)
        working['v2_company_log'] = working['v2_company_log'].where(
            working['v2_company_log'].map(lambda value: not _is_missing_text(value)),
            inferred_v2_company,
        )

    av_available = int(
        working['av_company_log'].map(lambda value: not _is_missing_text(value)).sum()
    )
    v2_available = int(
        working['v2_company_log'].map(lambda value: not _is_missing_text(value)).sum()
    )

    return {
        'av_company_available': av_available,
        'av_company_missing': int(len(working) - av_available),
        'v2_company_available': v2_available,
        'v2_company_missing': int(len(working) - v2_available),
        'av_company_top_counts': _summarise_company_values(
            working['av_company_log'],
            top_n=top_n,
        ),
        'v2_company_top_counts': _summarise_company_values(
            working['v2_company_log'],
            top_n=top_n,
        ),
    }


def _build_interpretation_log(summary: dict[str, object]) -> dict[str, object]:
    """Builds a compact interpretation oriented summary for logging.

    Args:
        summary: Research summary dictionary returned by the pipeline.

    Returns:
        A dictionary containing derived high level findings for logging.
    """

    taxonomy = {
        str(k): int(v)
        for k, v in (summary.get('taxonomy_top_counts', {}) or {}).items()
    }
    blind_spots = {
        str(k): float(v)
        for k, v in (summary.get('blind_spot_top_missingness', {}) or {}).items()
    }
    blame = {
        str(k): int(v)
        for k, v in (summary.get('blame_distribution', {}) or {}).items()
    }
    determinability = {
        str(k): int(v)
        for k, v in (
            summary.get('scenario_determinability_distribution', {}) or {}
        ).items()
    }
    movement = {
        str(k): int(v)
        for k, v in (
            summary.get('movement_consistency_distribution', {}) or {}
        ).items()
    }
    availability = {
        str(k): int(v)
        for k, v in (summary.get('data_availability_summary', {}) or {}).items()
    }
    disagreements = {
        str(k): int(v)
        for k, v in (summary.get('source_disagreement_summary', {}) or {}).items()
    }

    empirical_rows = int(summary.get('rows_used_for_empirical_analysis', 0) or 0)
    parsed_rows = int(summary.get('rows_total', 0) or 0)
    rows_with_any_output = int(
        availability.get('rows_with_any_model_output', parsed_rows) or 0
    )
    ambiguity_count = int(taxonomy.get('other_or_ambiguous', 0) or 0)
    blame_total = _safe_total(blame)
    movement_total = _safe_total(movement)
    determinability_total = _safe_total(determinability)

    top_taxonomy_items = sorted(
        taxonomy.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    top_blind_spots = sorted(
        blind_spots.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    top_disagreement_fields = sorted(
        disagreements.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]

    return {
        'empirical_subset_retention_from_rows_with_output': (
            round(empirical_rows / rows_with_any_output, 3)
            if rows_with_any_output else 0.0
        ),
        'empirical_subset_retention_from_parsed_rows': (
            round(empirical_rows / parsed_rows, 3)
            if parsed_rows else 0.0
        ),
        'dominant_scenarios': (
            '; '.join(f'{k} ({v})' for k, v in top_taxonomy_items)
            if top_taxonomy_items else 'NA'
        ),
        'top_two_scenario_share_of_empirical_subset': (
            round(sum(v for _, v in top_taxonomy_items[:2]) / empirical_rows, 3)
            if empirical_rows and len(top_taxonomy_items) >= 2 else 0.0
        ),
        'ambiguity_rate_in_empirical_subset': (
            round(ambiguity_count / empirical_rows, 3)
            if empirical_rows else 0.0
        ),
        'reported_other_road_user_blame_share': (
            round(blame.get('other_road_user', 0) / blame_total, 3)
            if blame_total else 0.0
        ),
        'high_determinability_share': (
            round(determinability.get('high', 0) / determinability_total, 3)
            if determinability_total else 0.0
        ),
        'movement_inconsistency_share': (
            round(movement.get('inconsistent', 0) / movement_total, 3)
            if movement_total else 0.0
        ),
        'strongest_blind_spots': (
            '; '.join(f'{k}={round(v, 3)}' for k, v in top_blind_spots)
            if top_blind_spots else 'NA'
        ),
        'largest_cross_source_disagreements': (
            '; '.join(f'{k} ({v})' for k, v in top_disagreement_fields)
            if top_disagreement_fields else 'NA'
        ),
        'mean_context_gap': summary.get('average_context_gap', 0.0),
        'mean_explicitness_score': summary.get('average_explicitness_score', 0.0),
        'paper_takeaway': (
            'Recurring conflicts dominate and fine interaction context remains under specified'
            if empirical_rows else 'No empirical subset available'
        ),
    }


def main() -> int:
    """Runs the full analysis pipeline.

    Returns:
        Process style exit code. ``0`` indicates success.
    """

    config = load_runtime_config_from_common()
    output_dirs = ensure_output_dirs(config.output_dir, config.figures_dir)

    # Initialise logging before any substantial processing begins.
    logs(
        show_level=config.log_level,
        save_level=config.log_level,
        program_name='analysis',
        path=str(output_dirs.base),
        show_color=True,
    )

    log_kv_block(logger, 'Configuration summary', {
        'config_source': config.config_source,
        'config_path': config.config_path,
        'input_csv': config.input_csv,
        'output_dir': config.output_dir,
        'figures_dir': config.figures_dir,
        'row_keep_policy': config.row_keep_policy,
        'auto_open_html': config.auto_open_html,
        'save_final': config.save_final,
        'image_export_timeout_seconds': config.image_export_timeout_seconds,
    })

    # Load the raw CSV and pick the text column used for parsing.
    raw_df, selected_text_column = load_input_events(
        input_csv=config.input_csv,
        preferred_text_column=config.text_column,
        row_keep_policy=config.row_keep_policy,
    )
    logger.info('Using text column: {}', selected_text_column)

    parsed_df = parse_events_dataframe(raw_df, text_column=selected_text_column)
    if parsed_df.empty:
        logger.warning('No rows remained after loading and parsing. Nothing to analyse.')
        return 0

    # Derive research columns, resolve plot fields, and build the empirical
    # subset used for plot based analysis.
    research_df = derive_research_columns(
        parsed_df,
        blind_spot_fields=config.blind_spot_fields,
    )

    log_kv_block(
        logger,
        'Report date summary',
        _build_report_date_log(research_df),
    )
    log_kv_block(
        logger,
        'Vehicle company summary',
        _build_company_log(research_df),
    )

    plot_fields = resolve_plot_fields(
        research_df,
        config.include_plot_fields,
        config.exclude_plot_fields,
    )
    try:
        story_plot_fields = resolve_plot_fields(
            research_df,
            DEFAULT_5W1H_PLOT_FIELDS,
            [],
        )
    except ValueError:
        story_plot_fields = []

    filtered_df, filter_report = apply_plot_filters(
        research_df,
        plot_fields=plot_fields,
        filter_rows_with_na=config.filter_rows_with_na,
        na_filter_fields=config.na_filter_fields,
    )

    log_kv_block(logger, 'Row summary', {
        'loaded': research_df.attrs.get(
            'total_rows_original',
            len(raw_df) + filter_report.get('dropped_empty_output', 0),
        ),
        'dropped_by_row_policy': filter_report.get('dropped_empty_output', 0),
        'parsed': len(research_df),
        'dropped_for_plot_na': filter_report.get('dropped_for_plot_na', 0),
        'used_for_plots': len(filtered_df),
    })

    # Write core cleaned and plot input datasets.
    save_dataframe(research_df, output_dirs.base / 'cleaned_events.csv')
    save_dataframe(filtered_df, output_dirs.base / 'accident_overview.csv')

    plot_input_columns = [
        'row_id',
        'report_pdf',
        'source_report',
        'scenario_class',
    ] + [field for field in plot_fields if field in filtered_df.columns] \
      + [field for field in story_plot_fields if field in filtered_df.columns]
    plot_input_columns = [
        column for column in plot_input_columns if column in filtered_df.columns
    ]
    save_dataframe(
        filtered_df[plot_input_columns].copy(),
        output_dirs.base / 'plot_input_filtered.csv',
    )

    overview_summary = build_overview_summary(
        filtered_df=filtered_df,
        plot_fields=plot_fields,
        filter_report=filter_report,
    )
    save_json(overview_summary, output_dirs.base / 'accident_overview_summary.json')

    # Build and export the richer research summary tables.
    research_summary, research_tables = build_research_summary(
        research_df=research_df,
        filtered_research_df=filtered_df,
        filter_report=filter_report,
        blind_spot_fields=config.blind_spot_fields,
        taxonomy_top_n=config.paper_plot_top_n,
    )
    save_json(research_summary, output_dirs.base / 'research_summary.json')
    export_research_tables(research_tables, output_dirs.base)

    validation_df = create_validation_sample(
        filtered_df,
        sample_size=config.validation_sample_size,
        seed=config.validation_seed,
        include_text=config.validation_include_text,
    )
    save_dataframe(validation_df, output_dirs.base / 'validation_sample.csv')

    markdown_report = format_research_markdown(research_summary, config=config)
    save_markdown(markdown_report, output_dirs.base / 'run_report.md')

    log_kv_block(logger, 'Core outputs written', {
        'cleaned_events': output_dirs.base / 'cleaned_events.csv',
        'accident_overview': output_dirs.base / 'accident_overview.csv',
        'plot_input_filtered': output_dirs.base / 'plot_input_filtered.csv',
        'research_summary': output_dirs.base / 'research_summary.json',
        'validation_sample': output_dirs.base / 'validation_sample.csv',
        'run_report': output_dirs.base / 'run_report.md',
        'drop_reason_summary': output_dirs.base / 'drop_reason_summary.csv',
        'source_disagreement_summary': output_dirs.base / 'source_disagreement_summary.csv',
        'movement_inconsistency_audit': output_dirs.base / 'movement_inconsistency_audit.csv',
        'other_or_ambiguous_review': output_dirs.base / 'other_or_ambiguous_review.csv',
    })

    # Build and export all overview, histogram, and paper figures.
    manifest = create_all_plots(
        parsed_df=research_df,
        filtered_df=filtered_df,
        plot_fields=plot_fields,
        story_plot_fields=story_plot_fields,
        output_dirs=output_dirs,
        auto_open_html=config.auto_open_html,
        histogram_fields=config.histogram_fields,
        min_count=config.min_count,
        max_categories=config.max_categories,
        save_final=config.save_final,
        paper_plot_top_n=config.paper_plot_top_n,
        blind_spot_fields=config.blind_spot_fields,
        image_export_timeout_seconds=config.image_export_timeout_seconds,
    )
    save_json(manifest, output_dirs.base / 'plot_manifest.json')

    log_kv_block(logger, 'Plot export summary', summarise_plot_manifest(manifest))
    log_kv_block(logger, 'Key empirical results', {
        'top_taxonomy_classes': research_summary.get('taxonomy_top_counts', {}),
        'top_blind_spots': research_summary.get('blind_spot_top_missingness', {}),
        'blame_distribution': research_summary.get('blame_distribution', {}),
        'provenance_mean_availability': research_summary.get('provenance_mean_availability', {}),
        'movement_consistency_distribution': research_summary.get('movement_consistency_distribution', {}),
        'scenario_determinability_distribution': research_summary.get('scenario_determinability_distribution', {}),
        'data_availability_summary': research_summary.get('data_availability_summary', {}),
        'source_disagreement_summary': research_summary.get('source_disagreement_summary', {}),
        'movement_inconsistency_diagnosis': research_summary.get('movement_inconsistency_diagnosis', {}),
        'blame_evidence_strength_distribution': research_summary.get('blame_evidence_strength_distribution', {}),
        'text_source_selection': (
            {
                str(key): int(value)
                for key, value in research_df['selected_text_column'].value_counts().items()
            }
            if 'selected_text_column' in research_df.columns else {}
        ),
    })
    log_kv_block(
        logger,
        'Interpretation ready findings',
        _build_interpretation_log(research_summary),
    )

    logger.info('Finished successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
