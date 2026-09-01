from __future__ import annotations
#
# Documentation and reproducibility guide
#
# This file intentionally contains extensive comments because the analysis is
# designed for academic audit, handover, and research reproducibility.
# The comments explain why transformations exist, which outputs are descriptive,
# and where interpretation must remain cautious.
#
# Documentation principles used throughout this module:
#
# 1. Keep source extraction separate from derived analytical categories.
# 2. Preserve missing values instead of inventing values for incomplete reports.
# 3. Keep figure only relabelling separate from the analytical dataframe.
# 4. Treat automated stability checks as diagnostics rather than accuracy estimates.
# 5. Preserve Reviewer 1 and Reviewer 2 coding as independent observations.
# 6. Report exact agreement together with Cohen's kappa for categorical validation.
# 7. Keep Q0 responsibility attribution separate from bounded field extraction.
# 8. Treat responsibility labels as descriptive rather than legal fault findings.
# 9. Preserve row level rule traces so every scenario assignment can be audited.
# 10. Keep contradictory movement evidence visible and test sensitivity to it.
# 11. Distinguish report context from online enrichment fields.
# 12. Keep reporting precision formatting separate from full precision values.
# 13. Do not regenerate historical reviewer order records after coding is complete.
# 14. Store any future adjudicated reference separately from original human coding.
# 15. Keep configuration defaults aligned with the README and project config files.
#
# The source uses British English in research facing prose and comments where
# practical. Identifiers retain their existing spelling to avoid breaking APIs.
#


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
import plotly.express as px
import plotly.graph_objects as go

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
import utils.summary_plots as summary_plots

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


# ==========================================================================
# Developer documentation for `_get_common_config`
# ==========================================================================
# Purpose:
#   Returns the first non null config value found for the given keys.
#
# Inputs:
#   Parameters in this helper: *keys, default.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_coerce_bool`
# ==========================================================================
# Purpose:
#   Coerces a loosely typed config value to ``bool``.
#
# Inputs:
#   Parameters in this helper: value, default.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_coerce_int`
# ==========================================================================
# Purpose:
#   Coerces a config value to ``int`` with a safe fallback.
#
# Inputs:
#   Parameters in this helper: value, default.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_coerce_list`
# ==========================================================================
# Purpose:
#   Coerces a config value into a list of non empty strings.
#
# Inputs:
#   Parameters in this helper: value, default.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_coerce_row_keep_policy`
# ==========================================================================
# Purpose:
#   Normalises the configured row keep policy.
#
# Inputs:
#   Parameters in this helper: value, default.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_resolve_project_root`
# ==========================================================================
# Purpose:
#   Resolves the project root directory.
#
# Inputs:
#   Parameters in this helper: no explicit parameters.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_looks_like_container_path`
# ==========================================================================
# Purpose:
#   Returns whether a path appears to point into a transient container.
#
# Inputs:
#   Parameters in this helper: path.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _looks_like_container_path(path: Path) -> bool:
    """Returns whether a path appears to point into a transient container.

    Args:
        path: Candidate path.

    Returns:
        ``True`` when the path starts with common container mount prefixes.
    """

    text = str(path)
    return text.startswith('/mnt/') or text.startswith('/tmp/')


# ==========================================================================
# Developer documentation for `_resolve_input_csv`
# ==========================================================================
# Purpose:
#   Resolves the input CSV path from config and sensible fallbacks.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_resolve_output_dir`
# ==========================================================================
# Purpose:
#   Resolves the primary output directory.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_resolve_figures_dir`
# ==========================================================================
# Purpose:
#   Resolves the final figures directory.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `load_runtime_config_from_common`
# ==========================================================================
# Purpose:
#   Loads and resolves runtime configuration from the shared config source.
#
# Inputs:
#   Parameters in this helper: no explicit parameters.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_safe_total`
# ==========================================================================
# Purpose:
#   Sums integer like values from a mapping safely.
#
# Inputs:
#   Parameters in this helper: mapping.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _safe_total(mapping: dict[str, int]) -> int:
    """Sums integer like values from a mapping safely.

    Args:
        mapping: Mapping of string keys to integer like values.

    Returns:
        The summed integer total, or ``0`` when the mapping is empty.
    """

    return int(sum(int(v) for v in mapping.values())) if mapping else 0


# ==========================================================================
# Developer documentation for `_is_missing_text`
# ==========================================================================
# Purpose:
#   Returns whether a loosely typed value should be treated as missing.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_first_non_missing_value`
# ==========================================================================
# Purpose:
#   Returns the first present value across the provided columns for one row.
#
# Inputs:
#   Parameters in this helper: row, columns.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _first_non_missing_value(row: pd.Series, columns: list[str]) -> str | None:
    """Returns the first present value across the provided columns for one row."""

    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if not _is_missing_text(value):
            return str(value).strip()
    return None


# ==========================================================================
# Developer documentation for `_figure2_where_label`
# ==========================================================================
# Purpose:
#   Returns a transportation friendly location label for Figure 2.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Figure helpers change presentation while keeping the underlying analytical values unchanged.
#   Reader facing labels should be clear enough to stand alone in exported figures.
#   Display transformations must be applied to plotting copies rather than the analytical dataframe.
#   Scenario order should remain stable when a figure is regenerated for the same input data.
#   Percentages must use the same denominator stated in the axis label and figure documentation.
#   Legend and axis changes should not alter category membership or the underlying count table.
#   The current reporting precision convention is one decimal place for displayed percentages.
#   Static export behaviour should be checked because Plotly HTML and PNG layout can differ.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _figure2_where_label(value: Any) -> str:
    """Returns a transportation friendly location label for Figure 2.

    The analytical ``where_group`` field can encode both intersection status
    and a roadway subtype in one token, for example ``intersection_roadway``.
    That is useful internally but reads ambiguously in the 5W1H Sankey. For
    display, intersection locations are therefore shown simply as
    ``intersection`` and non intersection roadway locations as
    ``road_segment``. Other specific location labels are retained.
    """

    if _is_missing_text(value):
        return 'unknown'

    text = str(value).strip().lower().replace(' ', '_')

    if text == 'intersection' or text.startswith('intersection_'):
        return 'intersection'

    if (
        text == 'roadway'
        or text == 'segment'
        or text == 'road_segment'
        or text.startswith('non_intersection_')
    ):
        return 'road_segment'

    return text


# ==========================================================================
# Developer documentation for `_prepare_figure_display_data`
# ==========================================================================
# Purpose:
#   Returns a plotting copy with reader facing Figure 2 location labels.
#
# Inputs:
#   Parameters in this helper: df.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Figure helpers change presentation while keeping the underlying analytical values unchanged.
#   Reader facing labels should be clear enough to stand alone in exported figures.
#   Display transformations must be applied to plotting copies rather than the analytical dataframe.
#   Scenario order should remain stable when a figure is regenerated for the same input data.
#   Percentages must use the same denominator stated in the axis label and figure documentation.
#   Legend and axis changes should not alter category membership or the underlying count table.
#   The current reporting precision convention is one decimal place for displayed percentages.
#   Static export behaviour should be checked because Plotly HTML and PNG layout can differ.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _prepare_figure_display_data(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a plotting copy with reader facing Figure 2 location labels."""

    display_df = df.copy()
    if 'where_group' in display_df.columns:
        display_df['where_group'] = display_df['where_group'].map(
            _figure2_where_label
        )
    return display_df


FIGURE4_ATTRIBUTION_ORDER = [
    'other_road_user',
    'av_primary',
    'environment_or_conditions',
    'unclear',
]


# ==========================================================================
# Developer documentation for `_figure4_display_label`
# ==========================================================================
# Purpose:
#   Returns a reader-facing label while preserving the AV abbreviation.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Figure helpers change presentation while keeping the underlying analytical values unchanged.
#   Reader facing labels should be clear enough to stand alone in exported figures.
#   Display transformations must be applied to plotting copies rather than the analytical dataframe.
#   Scenario order should remain stable when a figure is regenerated for the same input data.
#   Percentages must use the same denominator stated in the axis label and figure documentation.
#   Legend and axis changes should not alter category membership or the underlying count table.
#   The current reporting precision convention is one decimal place for displayed percentages.
#   Static export behaviour should be checked because Plotly HTML and PNG layout can differ.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _figure4_display_label(value: Any) -> str:
    """Returns a reader-facing label while preserving the AV abbreviation."""

    text = str(value).strip().replace('_', ' ')
    if not text:
        return 'Unknown'

    words = text.split()
    formatted: list[str] = []
    for index, word in enumerate(words):
        lower = word.lower()
        if lower == 'av':
            formatted.append('AV')
        elif index == 0:
            formatted.append(word.capitalize())
        else:
            formatted.append(lower)
    return ' '.join(formatted)


# ==========================================================================
# Developer documentation for `_create_accountability_by_taxonomy_figure`
# ==========================================================================
# Purpose:
#   Creates Figure 4 as grouped within-scenario attribution percentages.
#
# Inputs:
#   Parameters in this helper: df, top_n.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Figure helpers change presentation while keeping the underlying analytical values unchanged.
#   Reader facing labels should be clear enough to stand alone in exported figures.
#   Display transformations must be applied to plotting copies rather than the analytical dataframe.
#   Scenario order should remain stable when a figure is regenerated for the same input data.
#   Percentages must use the same denominator stated in the axis label and figure documentation.
#   Legend and axis changes should not alter category membership or the underlying count table.
#   The current reporting precision convention is one decimal place for displayed percentages.
#   Static export behaviour should be checked because Plotly HTML and PNG layout can differ.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _create_accountability_by_taxonomy_figure(
    df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """Creates Figure 4 as grouped within-scenario attribution percentages.

    Figure 3 answers how frequent each scenario is in the corpus. Figure 4
    deliberately answers a different question: within each scenario, what
    percentage of reports was attributed to each responsibility group?

    The four principal attribution groups are shown side by side for every
    scenario. Zero-count combinations are retained so the grouping is
    consistent across scenario classes. Any additional attribution categories
    present in the data are retained after the four principal categories.
    """

    required = {'scenario_class', 'blame_group'}
    if not required.issubset(df.columns) or df.empty:
        return go.Figure()

    working = df.loc[
        df['scenario_class'].map(lambda value: not _is_missing_text(value))
        & df['blame_group'].map(lambda value: not _is_missing_text(value))
    ].copy()
    if working.empty:
        return go.Figure()

    working['scenario_class'] = working['scenario_class'].astype(str).str.strip()
    working['blame_group'] = working['blame_group'].astype(str).str.strip()

    # Use the same top scenario definition as Figure 3.
    scenario_counts = working['scenario_class'].value_counts()
    top_scenarios = scenario_counts.head(top_n).index.tolist()
    working = working.loc[working['scenario_class'].isin(top_scenarios)].copy()

    observed_attributions = working['blame_group'].drop_duplicates().tolist()
    attribution_order = [
        value for value in FIGURE4_ATTRIBUTION_ORDER
        if value in observed_attributions
    ]
    attribution_order.extend(
        sorted(
            value for value in observed_attributions
            if value not in attribution_order
        )
    )

    if not top_scenarios or not attribution_order:
        return go.Figure()

    # Complete the scenario × attribution grid so each scenario has the same
    # grouped-bar structure, including explicit zero-count combinations.
    complete_index = pd.MultiIndex.from_product(
        [top_scenarios, attribution_order],
        names=['scenario_class', 'blame_group'],
    )
    counts = (
        working.groupby(['scenario_class', 'blame_group'])
        .size()
        .reindex(complete_index, fill_value=0)
        .rename('count')
        .reset_index()
    )

    scenario_totals = counts.groupby('scenario_class')['count'].transform('sum')
    counts['percentage'] = (
        counts['count'] / scenario_totals.where(scenario_totals.gt(0), 1)
    ) * 100.0

    counts['scenario_label'] = counts['scenario_class'].map(
        _figure4_display_label
    )
    counts['attribution_label'] = counts['blame_group'].map(
        _figure4_display_label
    )

    # Figure 3 places the largest scenario at the top of the horizontal chart.
    # Plotly categorical arrays are ordered bottom to top, so reverse the
    # descending frequency order here to match Figure 3 visually.
    scenario_label_order = [
        _figure4_display_label(value)
        for value in reversed(top_scenarios)
    ]
    attribution_label_order = [
        _figure4_display_label(value)
        for value in attribution_order
    ]

    palette = px.colors.qualitative.Plotly
    attribution_colour_map = {
        label: palette[index % len(palette)]
        for index, label in enumerate(attribution_label_order)
    }

    try:
        font_size = int(common.get_configs('font_size'))
    except Exception:
        font_size = 14

    fig = px.bar(
        counts,
        x='percentage',
        y='scenario_label',
        color='attribution_label',
        orientation='h',
        barmode='group',
        title='',
        category_orders={
            'scenario_label': scenario_label_order,
            'attribution_label': attribution_label_order,
        },
        color_discrete_map=attribution_colour_map,
        custom_data=['count'],
    )

    # Percentage labels make Figure 4 interpretable independently of Figure 3.
    for trace in fig.data:
        percentages = pd.to_numeric(
            pd.Series(list(trace.x)),  # type: ignore
            errors='coerce',
        ).fillna(0.0)
        trace.text = [  # type: ignore
            f'{value:.1f}%' if value > 0 else ''
            for value in percentages
        ]
        trace.texttemplate = '%{text}'  # type: ignore
        trace.textposition = 'outside'  # type: ignore
        trace.cliponaxis = False  # type: ignore
        trace.hovertemplate = (  # type: ignore
            'Scenario=%{y}<br>'
            'Attribution=' + str(trace.name) + '<br>'  # type: ignore
            'Percentage=%{x:.1f}%<br>'
            'Count=%{customdata[0]:.0f}<extra></extra>'
        )

    figure_height = max(650, 88 * len(top_scenarios) + 150)

    fig.update_layout(
        # The native horizontal Plotly legend can be auto-repositioned during
        # static export when zero margins and long axis labels are combined.
        # Disable it and draw a small fixed legend in the reserved top strip
        # below. This makes its position deterministic in both HTML and PNG.
        showlegend=False,
        bargap=0.24,
        bargroupgap=0.06,
        height=figure_height,
        font=dict(size=font_size),
        margin=dict(t=0, r=0, b=0, l=0),
        uniformtext_minsize=max(font_size - 2, 8),
        uniformtext_mode='hide',
    )
    fig.update_xaxes(
        title_text='Percentage of reports within scenario',
        range=[0, 110],
        ticksuffix='%',
        showgrid=True,
        zeroline=True,
        automargin=True,
    )
    fig.update_yaxes(
        title_text='Scenario class',
        categoryorder='array',
        categoryarray=scenario_label_order,
        # Reserve the top strip of the existing canvas for the manual legend.
        # This is internal plot-domain space, not an external margin.
        domain=[0.0, 0.87],
        automargin=True,
    )

    # Draw a deterministic horizontal legend. The native Plotly legend was
    # visibly ignoring/clamping x/y placement during static export in this
    # zero-margin layout. Paper coordinates keep this legend tied to the
    # plotting canvas and prevent it from being clipped or moved automatically.
    legend_y = 0.87
    legend_x = 0.23
    square_width = 0.012
    square_half_height = 0.010

    for label in attribution_label_order:
        colour = attribution_colour_map[label]

        fig.add_shape(
            type='rect',
            xref='paper',
            yref='paper',
            x0=legend_x,
            x1=legend_x + square_width,
            y0=legend_y - square_half_height,
            y1=legend_y + square_half_height,
            fillcolor=colour,
            line=dict(width=0),
            layer='above',
        )
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=legend_x + square_width + 0.007,
            y=legend_y,
            text=label,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            align='left',
            font=dict(size=font_size),
        )

        # Approximate the horizontal space required by the rendered label.
        # This keeps the four entries on one line without a large right gap.
        legend_x += 0.052 + (0.0076 * len(label))

    return fig


# ==========================================================================
# Developer documentation for `_infer_company_from_model`
# ==========================================================================
# Purpose:
#   Infers a probable company or make from a free text vehicle model string.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Logging helpers create compact diagnostics without changing the dataframe used for analysis.
#   Counts and date ranges are intended for reproducibility checks and analysis verification.
#   Missing values should stay explicit instead of being converted into plausible looking labels.
#   Ordering is deterministic so repeated runs are easy to compare in logs and exported summaries.
#   Derived text in logs is descriptive and must not be interpreted as additional source evidence.
#   Where a field is inferred for logging convenience, the original extracted field remains untouched.
#   Keep log summaries compact enough to inspect while retaining the values needed for audit.
#   Changes here should not alter the scientific tables unless the corresponding analysis also changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_build_report_date_log`
# ==========================================================================
# Purpose:
#   Builds a logger friendly summary of report dates from oldest to newest.
#
# Inputs:
#   Parameters in this helper: df.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Logging helpers create compact diagnostics without changing the dataframe used for analysis.
#   Counts and date ranges are intended for reproducibility checks and analysis verification.
#   Missing values should stay explicit instead of being converted into plausible looking labels.
#   Ordering is deterministic so repeated runs are easy to compare in logs and exported summaries.
#   Derived text in logs is descriptive and must not be interpreted as additional source evidence.
#   Where a field is inferred for logging convenience, the original extracted field remains untouched.
#   Keep log summaries compact enough to inspect while retaining the values needed for audit.
#   Changes here should not alter the scientific tables unless the corresponding analysis also changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

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


# ==========================================================================
# Developer documentation for `_summarise_company_values`
# ==========================================================================
# Purpose:
#   Builds a compact count summary for a company, manufacturer, or make series.
#
# Inputs:
#   Parameters in this helper: values, top_n.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Logging helpers create compact diagnostics without changing the dataframe used for analysis.
#   Counts and date ranges are intended for reproducibility checks and analysis verification.
#   Missing values should stay explicit instead of being converted into plausible looking labels.
#   Ordering is deterministic so repeated runs are easy to compare in logs and exported summaries.
#   Derived text in logs is descriptive and must not be interpreted as additional source evidence.
#   Where a field is inferred for logging convenience, the original extracted field remains untouched.
#   Keep log summaries compact enough to inspect while retaining the values needed for audit.
#   Changes here should not alter the scientific tables unless the corresponding analysis also changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _summarise_company_values(
    values: pd.Series,
    top_n: int | None = 10,
) -> dict[str, int]:
    """Builds a compact count summary for a company, manufacturer, or make series."""

    cleaned = values.map(
        lambda value: None if _is_missing_text(value) else str(value).strip()
    ).dropna()

    if cleaned.empty:
        return {}

    counts = cleaned.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


# ==========================================================================
# Developer documentation for `_build_company_log`
# ==========================================================================
# Purpose:
#   Builds a logger friendly summary of AV and other vehicle company fields.
#
# Inputs:
#   Parameters in this helper: df, top_n.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Logging helpers create compact diagnostics without changing the dataframe used for analysis.
#   Counts and date ranges are intended for reproducibility checks and analysis verification.
#   Missing values should stay explicit instead of being converted into plausible looking labels.
#   Ordering is deterministic so repeated runs are easy to compare in logs and exported summaries.
#   Derived text in logs is descriptive and must not be interpreted as additional source evidence.
#   Where a field is inferred for logging convenience, the original extracted field remains untouched.
#   Keep log summaries compact enough to inspect while retaining the values needed for audit.
#   Changes here should not alter the scientific tables unless the corresponding analysis also changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _build_company_log(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Builds a logger friendly summary of AV and other vehicle company fields."""

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

    if 'av_manufacturer' in working.columns:
        working['av_manufacturer_log'] = working['av_manufacturer']
    else:
        working['av_manufacturer_log'] = None

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

    av_manufacturer_available = int(
        working['av_manufacturer_log'].map(
            lambda value: not _is_missing_text(value)
        ).sum()
    )
    av_available = int(
        working['av_company_log'].map(lambda value: not _is_missing_text(value)).sum()
    )
    v2_available = int(
        working['v2_company_log'].map(lambda value: not _is_missing_text(value)).sum()
    )

    return {
        'av_manufacturer_available': av_manufacturer_available,
        'av_manufacturer_missing': int(len(working) - av_manufacturer_available),
        'av_manufacturer_distinct_count': len(
            _summarise_company_values(working['av_manufacturer_log'], top_n=None)
        ),
        'av_manufacturer_counts': _summarise_company_values(
            working['av_manufacturer_log'],
            top_n=None,
        ),
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


# ==========================================================================
# Developer documentation for `_build_interpretation_log`
# ==========================================================================
# Purpose:
#   Builds a compact interpretation oriented summary for logging.
#
# Inputs:
#   Parameters in this helper: summary.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Logging helpers create compact diagnostics without changing the dataframe used for analysis.
#   Counts and date ranges are intended for reproducibility checks and analysis verification.
#   Missing values should stay explicit instead of being converted into plausible looking labels.
#   Ordering is deterministic so repeated runs are easy to compare in logs and exported summaries.
#   Derived text in logs is descriptive and must not be interpreted as additional source evidence.
#   Where a field is inferred for logging convenience, the original extracted field remains untouched.
#   Keep log summaries compact enough to inspect while retaining the values needed for audit.
#   Changes here should not alter the scientific tables unless the corresponding analysis also changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _build_interpretation_log(summary: dict[str, object]) -> dict[str, object]:
    """Builds a compact interpretation oriented summary for logging.

    Args:
        summary: Research summary dictionary returned by the pipeline.

    Returns:
        A dictionary containing derived high level findings for logging.
    """

    taxonomy = {
        str(k): int(v)
        for k, v in (summary.get('taxonomy_top_counts', {}) or {}).items()  # type: ignore
    }
    post_extraction_unavailability = {
        str(k): float(v)
        for k, v in (
            summary.get(
                'post_extraction_top_unavailability',
                summary.get('blind_spot_top_missingness', {}),
            )
            or {}
        ).items()  # type: ignore
    }
    blame = {
        str(k): int(v)
        for k, v in (summary.get('blame_distribution', {}) or {}).items()  # type: ignore
    }
    rule_support = {
        str(k): int(v)
        for k, v in (
            summary.get('scenario_rule_support_distribution', {}) or {}
        ).items()  # type: ignore
    }
    movement_agreement = {
        str(k): int(v)
        for k, v in (
            summary.get('movement_field_agreement_distribution', {}) or {}
        ).items()  # type: ignore
    }
    availability = {
        str(k): int(v)
        for k, v in (summary.get('data_availability_summary', {}) or {}).items()  # type: ignore
    }
    disagreements = {
        str(k): int(v)
        for k, v in (summary.get('source_disagreement_summary', {}) or {}).items()  # type: ignore
    }

    empirical_rows = int(summary.get('rows_used_for_empirical_analysis', 0) or 0)  # type: ignore
    parsed_rows = int(summary.get('rows_total', 0) or 0)  # type: ignore
    rows_with_any_output = int(
        availability.get('rows_with_any_model_output', parsed_rows) or 0
    )
    ambiguity_count = int(taxonomy.get('other_or_ambiguous', 0) or 0)
    blame_total = _safe_total(blame)
    movement_total = _safe_total(movement_agreement)
    rule_support_total = _safe_total(rule_support)

    top_taxonomy_items = sorted(
        taxonomy.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    top_unavailable_fields = sorted(
        post_extraction_unavailability.items(),
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
        'high_rule_support_share': (
            round(rule_support.get('high', 0) / rule_support_total, 3)
            if rule_support_total else 0.0
        ),
        'contradictory_movement_field_share': (
            round(
                movement_agreement.get('contradictory', 0) / movement_total,
                3,
            )
            if movement_total else 0.0
        ),
        'highest_post_extraction_unavailability': (
            '; '.join(
                f'{k}={round(v, 3)}'
                for k, v in top_unavailable_fields
            )
            if top_unavailable_fields else 'NA'
        ),
        'largest_cross_source_disagreements': (
            '; '.join(f'{k} ({v})' for k, v in top_disagreement_fields)
            if top_disagreement_fields else 'NA'
        ),
        'mean_context_gap': summary.get('average_context_gap', 0.0),
        'mean_explicitness_score': summary.get('average_explicitness_score', 0.0),
        'interpretation_boundary': (
            'Availability and automated stability only; source presence and '
            'extraction accuracy require reference coding'
        ),
    }


# Human validation analysis -------------------------------------------------

VALIDATION_REVIEWERS = ('reviewer1', 'reviewer2')

VALIDATION_HUMAN_FIELDS = [
    'road_user_type_human',
    'av_mode_human',
    'v1_move_narrative_human',
    'v2_move_narrative_human',
    'move_v1_checkbox_human',
    'move_v2_checkbox_human',
    'v1_intersection_human',
    'v2_intersection_human',
    'collision_v1_human',
    'collision_v2_human',
    'parked_or_curbside_cue_human',
    'obstruction_yield_blocked_cue_human',
    'v1_injury_human',
    'v2_injury_human',
    'av_responsibility_human',
    'v1_lane_presence',
    'v2_lane_presence',
    'v1_speed_presence',
    'v2_speed_presence',
    'direction_presence',
]

VALIDATION_LLM_FIELD_MAP = {
    'road_user_type_human': 'road_user_type_llm',
    'av_mode_human': 'av_mode_llm',
    'v1_move_narrative_human': 'v1_move_narrative_llm',
    'v2_move_narrative_human': 'v2_move_narrative_llm',
    'move_v1_checkbox_human': 'move_v1_checkbox_llm',
    'move_v2_checkbox_human': 'move_v2_checkbox_llm',
    'v1_intersection_human': 'v1_intersection_llm',
    'v2_intersection_human': 'v2_intersection_llm',
    'collision_v1_human': 'collision_v1_llm',
    'collision_v2_human': 'collision_v2_llm',
    'v1_injury_human': 'v1_injury_llm',
    'v2_injury_human': 'v2_injury_llm',
    'av_responsibility_human': 'blame_group_llm',
}

VALIDATION_FIELD_LABELS = {
    'road_user_type_human': 'Road user type',
    'av_mode_human': 'AV operating mode',
    'v1_move_narrative_human': 'AV narrative movement',
    'v2_move_narrative_human': 'Other party narrative movement',
    'move_v1_checkbox_human': 'AV checkbox movement',
    'move_v2_checkbox_human': 'Other party checkbox movement',
    'v1_intersection_human': 'AV intersection status',
    'v2_intersection_human': 'Other party intersection status',
    'collision_v1_human': 'AV collision type',
    'collision_v2_human': 'Other party collision type',
    'parked_or_curbside_cue_human': 'Parked or curbside cue',
    'obstruction_yield_blocked_cue_human': 'Obstruction, yield, or blocked cue',
    'v1_injury_human': 'AV occupant injury',
    'v2_injury_human': 'Other party injury',
    'av_responsibility_human': 'AV responsibility attribution',
    'v1_lane_presence': 'AV lane source presence',
    'v2_lane_presence': 'Other party lane source presence',
    'v1_speed_presence': 'AV speed source presence',
    'v2_speed_presence': 'Other party speed source presence',
    'direction_presence': 'Direction source presence',
}

VALIDATION_SCENARIO_INPUT_FIELDS = [
    'road_user_type_human',
    'v1_move_narrative_human',
    'v2_move_narrative_human',
    'move_v1_checkbox_human',
    'move_v2_checkbox_human',
    'v1_intersection_human',
    'v2_intersection_human',
    'collision_v1_human',
    'collision_v2_human',
    'parked_or_curbside_cue_human',
    'obstruction_yield_blocked_cue_human',
]

VALIDATION_MISSING_VALUES = {
    '',
    'na',
    'n/a',
    'nan',
    'none',
    'null',
    'unknown',
    'not_stated',
    'not stated',
    'not specified',
    'ambiguous',
}

VALIDATION_VULNERABLE_ROAD_USERS = {
    'pedestrian',
    'cyclist',
    'scooter',
    'motorcycle',
}

VALIDATION_TURN_MOVEMENTS = {
    'turn_left',
    'turn_right',
    'turn_other',
    'turn_u',
}


# ==========================================================================
# Developer documentation for `_validation_text`
# ==========================================================================
# Purpose:
#   Returns a stable string representation for validation comparisons.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_text(value: Any) -> str:
    """Returns a stable string representation for validation comparisons."""

    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return str(value).strip()


# ==========================================================================
# Developer documentation for `_validation_is_missing`
# ==========================================================================
# Purpose:
#   Returns whether a human validation value is unavailable or ambiguous.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_is_missing(value: Any) -> bool:
    """Returns whether a human validation value is unavailable or ambiguous."""

    return _validation_text(value).lower() in VALIDATION_MISSING_VALUES


# ==========================================================================
# Developer documentation for `_validation_first_present`
# ==========================================================================
# Purpose:
#   Returns the first human coded value that is not unavailable.
#
# Inputs:
#   Parameters in this helper: *values.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_first_present(*values: Any) -> str:
    """Returns the first human coded value that is not unavailable."""

    for value in values:
        text = _validation_text(value)
        if text and not _validation_is_missing(text):
            return text
    return 'not_stated'


# ==========================================================================
# Developer documentation for `_validation_cohen_kappa`
# ==========================================================================
# Purpose:
#   Calculates chance adjusted categorical agreement.
#
# Inputs:
#   Parameters in this helper: left, right, sample_weights.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_cohen_kappa(
    left: pd.Series,
    right: pd.Series,
    sample_weights: pd.Series | None = None,
) -> float | None:
    """Calculates chance adjusted categorical agreement.

    When ``sample_weights`` is supplied, the observed and expected category
    probabilities are estimated with the validation sampling weights. This
    provides a corpus oriented estimate for the deliberately stratified sample.
    """

    pair = pd.DataFrame({
        'left': left.map(_validation_text),
        'right': right.map(_validation_text),
    })

    if sample_weights is None:
        pair['weight'] = 1.0
    else:
        pair['weight'] = pd.to_numeric(
            sample_weights.reindex(pair.index),
            errors='coerce',
        ).fillna(0.0)

    pair = pair.loc[
        pair['left'].ne('')
        & pair['right'].ne('')
        & pair['weight'].gt(0)
    ].copy()

    if pair.empty:
        return None

    total_weight = float(pair['weight'].sum())
    if total_weight <= 0:
        return None

    observed = float(
        pair.loc[pair['left'].eq(pair['right']), 'weight'].sum()
        / total_weight
    )

    labels = sorted(set(pair['left']) | set(pair['right']))
    expected = 0.0
    for label in labels:
        left_share = float(
            pair.loc[pair['left'].eq(label), 'weight'].sum() / total_weight
        )
        right_share = float(
            pair.loc[pair['right'].eq(label), 'weight'].sum() / total_weight
        )
        expected += left_share * right_share

    denominator = 1.0 - expected
    if abs(denominator) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else None
    return float((observed - expected) / denominator)


# ==========================================================================
# Developer documentation for `_validation_agreement_metrics`
# ==========================================================================
# Purpose:
#   Calculates exact agreement and Cohen's kappa for two coding series.
#
# Inputs:
#   Parameters in this helper: left, right, sample_weights.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_agreement_metrics(
    left: pd.Series,
    right: pd.Series,
    sample_weights: pd.Series | None = None,
) -> dict[str, float | int | None]:
    """Calculates exact agreement and Cohen's kappa for two coding series."""

    pair = pd.DataFrame({
        'left': left.map(_validation_text),
        'right': right.map(_validation_text),
    })
    if sample_weights is None:
        pair['weight'] = 1.0
    else:
        pair['weight'] = pd.to_numeric(
            sample_weights.reindex(pair.index),
            errors='coerce',
        ).fillna(0.0)

    pair = pair.loc[
        pair['left'].ne('')
        & pair['right'].ne('')
        & pair['weight'].gt(0)
    ].copy()

    if pair.empty:
        return {
            'n': 0,
            'exact_agreement': None,
            'cohen_kappa': None,
            'weighted_exact_agreement': None,
            'weighted_cohen_kappa': None,
        }

    matches = pair['left'].eq(pair['right'])
    exact = float(matches.mean())
    kappa = _validation_cohen_kappa(pair['left'], pair['right'])

    total_weight = float(pair['weight'].sum())
    weighted_exact = float(
        pair.loc[matches, 'weight'].sum() / total_weight
    ) if total_weight > 0 else None
    weighted_kappa = _validation_cohen_kappa(
        pair['left'],
        pair['right'],
        pair['weight'],
    )

    return {
        'n': int(len(pair)),
        'exact_agreement': exact,
        'cohen_kappa': kappa,
        'weighted_exact_agreement': weighted_exact,
        'weighted_cohen_kappa': weighted_kappa,
    }


# ==========================================================================
# Developer documentation for `_validation_find_file`
# ==========================================================================
# Purpose:
#   Finds one validation result file under the configured results directory.
#
# Inputs:
#   Parameters in this helper: root, filename.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_find_file(root: Path, filename: str) -> Path | None:
    """Finds one validation result file under the configured results directory."""

    direct = root / filename
    if direct.exists() and direct.is_file():
        return direct

    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            'Multiple validation files named {} found under {}. Using {}.',
            filename,
            root,
            matches[0],
        )
        return matches[0]
    return None


# ==========================================================================
# Developer documentation for `_resolve_validation_results_dir`
# ==========================================================================
# Purpose:
#   Resolves the frozen completed human validation result directory.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _resolve_validation_results_dir(project_root: Path) -> Path:
    """Resolves the frozen completed human validation result directory."""

    raw_value = _get_common_config(
        'validation_results_dir',
        default='validation_results/complete',
    )
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


# ==========================================================================
# Developer documentation for `_resolve_validation_analysis_dir`
# ==========================================================================
# Purpose:
#   Resolves the output directory for derived validation analyses.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Configuration helpers keep runtime behaviour reproducible across local and repository runs.
#   Fallback order is part of the public behaviour because different machines may expose different paths.
#   Values are normalised before use so later analysis code can operate on stable Python types.
#   A missing optional setting should normally fall back rather than stop an otherwise valid analysis.
#   Path handling must avoid accidentally binding a local run to a transient container location.
#   When changing configuration keys, retain backwards compatibility with existing project config files.
#   Do not silently reinterpret a valid user supplied value when a direct conversion is possible.
#   Keep default values aligned with `default.config` and the README whenever configuration changes.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _resolve_validation_analysis_dir(project_root: Path) -> Path:
    """Resolves the output directory for derived validation analyses."""

    raw_value = _get_common_config(
        'validation_analysis_dir',
        default='validation_results/analysis',
    )
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


# ==========================================================================
# Developer documentation for `_validation_load_completed_data`
# ==========================================================================
# Purpose:
#   Loads and validates the completed two reviewer human coding export.
#
# Inputs:
#   Parameters in this helper: project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_load_completed_data(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]] | None:
    """Loads and validates the completed two reviewer human coding export."""

    results_dir = _resolve_validation_results_dir(project_root)
    annotations_path = _validation_find_file(results_dir, 'human_annotations.csv')
    manifest_path = _validation_find_file(
        results_dir,
        'validation_sample_manifest.csv',
    )

    if annotations_path is None or manifest_path is None:
        logger.info(
            'Completed human validation results not found under {}. '
            'Skipping human validation analysis.',
            results_dir,
        )
        return None

    annotations = pd.read_csv(annotations_path)
    manifest = pd.read_csv(manifest_path)

    required_annotation_columns = {
        'reviewer_id',
        'validation_id',
        'submitted_at',
        *VALIDATION_HUMAN_FIELDS,
    }
    missing_annotation_columns = sorted(
        required_annotation_columns.difference(annotations.columns)
    )
    if missing_annotation_columns:
        raise ValueError(
            'Human validation annotations are missing required columns: '
            + ', '.join(missing_annotation_columns)
        )

    required_manifest_columns = {
        'validation_id',
        'source_report',
        'sampling_support_group',
        'sampling_scenario_class',
        'sampling_weight',
        'scenario_class_llm',
        *VALIDATION_LLM_FIELD_MAP.values(),
    }
    missing_manifest_columns = sorted(
        required_manifest_columns.difference(manifest.columns)
    )
    if missing_manifest_columns:
        raise ValueError(
            'Human validation manifest is missing required columns: '
            + ', '.join(missing_manifest_columns)
        )

    submitted = annotations.loc[
        annotations['submitted_at'].fillna('').astype(str).str.strip().ne('')
    ].copy()

    reviewer_counts = {
        reviewer: int(submitted['reviewer_id'].eq(reviewer).sum())
        for reviewer in VALIDATION_REVIEWERS
    }

    manifest_ids = set(manifest['validation_id'].astype(str))
    reviewer_id_sets = {
        reviewer: set(
            submitted.loc[
                submitted['reviewer_id'].eq(reviewer),
                'validation_id',
            ].astype(str)
        )
        for reviewer in VALIDATION_REVIEWERS
    }

    same_validation_set = all(
        reviewer_id_sets[reviewer] == manifest_ids
        for reviewer in VALIDATION_REVIEWERS
    )

    orders = None
    orders_path = _validation_find_file(results_dir, 'reviewer_orders.csv')
    same_presentation_order = None
    if orders_path is not None:
        orders = pd.read_csv(orders_path)
        if {
            'reviewer_id',
            'position',
            'validation_id',
        }.issubset(orders.columns):
            sequences: dict[str, list[str]] = {}
            for reviewer in VALIDATION_REVIEWERS:
                reviewer_order = (
                    orders.loc[orders['reviewer_id'].eq(reviewer)]
                    .sort_values('position')
                )
                sequences[reviewer] = reviewer_order[
                    'validation_id'
                ].astype(str).tolist()
            same_presentation_order = (
                sequences[VALIDATION_REVIEWERS[0]]
                == sequences[VALIDATION_REVIEWERS[1]]
            )

    validation_set_id = 'NA'
    set_id_path = _validation_find_file(results_dir, 'validation_set_id.txt')
    if set_id_path is not None:
        validation_set_id = set_id_path.read_text(encoding='utf-8').strip()

    status = {
        'results_dir': str(results_dir),
        'validation_set_id': validation_set_id,
        'manifest_n': int(len(manifest)),
        'reviewer_counts': reviewer_counts,
        'same_validation_set': bool(same_validation_set),
        'same_presentation_order': same_presentation_order,
        'complete': bool(
            len(manifest) > 0
            and all(
                reviewer_counts[reviewer] == len(manifest)
                for reviewer in VALIDATION_REVIEWERS
            )
            and same_validation_set
        ),
    }

    return submitted, manifest, orders, status


# ==========================================================================
# Developer documentation for `_validation_paired_annotations`
# ==========================================================================
# Purpose:
#   Returns one indexed annotation table for each independent reviewer.
#
# Inputs:
#   Parameters in this helper: annotations.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_paired_annotations(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one indexed annotation table for each independent reviewer."""

    reviewer_tables: dict[str, pd.DataFrame] = {}
    for reviewer in VALIDATION_REVIEWERS:
        reviewer_df = annotations.loc[
            annotations['reviewer_id'].eq(reviewer)
        ].copy()
        reviewer_df['validation_id'] = reviewer_df['validation_id'].astype(str)
        reviewer_df = reviewer_df.drop_duplicates(
            subset=['validation_id'],
            keep='last',
        )
        reviewer_tables[reviewer] = reviewer_df.set_index('validation_id')

    return (
        reviewer_tables[VALIDATION_REVIEWERS[0]],
        reviewer_tables[VALIDATION_REVIEWERS[1]],
    )


# ==========================================================================
# Developer documentation for `_validation_interrater_table`
# ==========================================================================
# Purpose:
#   Builds human versus human agreement and the disagreement audit.
#
# Inputs:
#   Parameters in this helper: annotations, manifest.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_interrater_table(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds human versus human agreement and the disagreement audit."""

    reviewer1, reviewer2 = _validation_paired_annotations(annotations)
    manifest_indexed = manifest.copy()
    manifest_indexed['validation_id'] = manifest_indexed[
        'validation_id'
    ].astype(str)
    manifest_indexed = manifest_indexed.set_index('validation_id')

    shared_ids = reviewer1.index.intersection(reviewer2.index)
    weights = manifest_indexed.reindex(shared_ids)['sampling_weight']

    agreement_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []

    for field in VALIDATION_HUMAN_FIELDS:
        left = reviewer1.reindex(shared_ids)[field]
        right = reviewer2.reindex(shared_ids)[field]
        metrics = _validation_agreement_metrics(left, right, weights)

        agreement_rows.append({
            'field': field,
            'field_label': VALIDATION_FIELD_LABELS.get(field, field),
            'n_both_coded': metrics['n'],
            'exact_agreement': metrics['exact_agreement'],
            'cohen_kappa': metrics['cohen_kappa'],
            'weighted_exact_agreement': metrics['weighted_exact_agreement'],
            'weighted_cohen_kappa': metrics['weighted_cohen_kappa'],
        })

        comparable = pd.DataFrame({
            'reviewer1': left.map(_validation_text),
            'reviewer2': right.map(_validation_text),
        })
        comparable = comparable.loc[
            comparable['reviewer1'].ne('')
            & comparable['reviewer2'].ne('')
            & comparable['reviewer1'].ne(comparable['reviewer2'])
        ]
        for validation_id, row in comparable.iterrows():
            manifest_row = manifest_indexed.loc[validation_id]  # type: ignore
            disagreement_rows.append({
                'validation_id': validation_id,
                'source_report': manifest_row.get('source_report', 'NA'),
                'sampling_support_group': manifest_row.get(
                    'sampling_support_group',
                    'NA',
                ),
                'sampling_scenario_class': manifest_row.get(
                    'sampling_scenario_class',
                    'NA',
                ),
                'field': field,
                'field_label': VALIDATION_FIELD_LABELS.get(field, field),
                'reviewer1': row['reviewer1'],
                'reviewer2': row['reviewer2'],
                'adjudicated_value': '',
                'adjudication_notes': '',
            })

    return pd.DataFrame(agreement_rows), pd.DataFrame(disagreement_rows)


# ==========================================================================
# Developer documentation for `_validation_llm_vs_reviewer_table`
# ==========================================================================
# Purpose:
#   Compares the LLM extraction independently against each human reviewer.
#
# Inputs:
#   Parameters in this helper: annotations, manifest.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_llm_vs_reviewer_table(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compares the LLM extraction independently against each human reviewer."""

    manifest_indexed = manifest.copy()
    manifest_indexed['validation_id'] = manifest_indexed[
        'validation_id'
    ].astype(str)
    manifest_indexed = manifest_indexed.set_index('validation_id')

    rows: list[dict[str, Any]] = []
    for reviewer in VALIDATION_REVIEWERS:
        reviewer_df = annotations.loc[
            annotations['reviewer_id'].eq(reviewer)
        ].copy()
        reviewer_df['validation_id'] = reviewer_df[
            'validation_id'
        ].astype(str)
        reviewer_df = reviewer_df.drop_duplicates(
            subset=['validation_id'],
            keep='last',
        ).set_index('validation_id')

        shared_ids = reviewer_df.index.intersection(manifest_indexed.index)
        weights = manifest_indexed.reindex(shared_ids)['sampling_weight']

        for human_field, llm_field in VALIDATION_LLM_FIELD_MAP.items():
            metrics = _validation_agreement_metrics(
                reviewer_df.reindex(shared_ids)[human_field],
                manifest_indexed.reindex(shared_ids)[llm_field],
                weights,
            )
            rows.append({
                'reviewer_id': reviewer,
                'field': human_field,
                'field_label': VALIDATION_FIELD_LABELS.get(
                    human_field,
                    human_field,
                ),
                'llm_field': llm_field,
                'n': metrics['n'],
                'exact_agreement': metrics['exact_agreement'],
                'cohen_kappa': metrics['cohen_kappa'],
                'weighted_exact_agreement': metrics[
                    'weighted_exact_agreement'
                ],
                'weighted_cohen_kappa': metrics['weighted_cohen_kappa'],
            })

    return pd.DataFrame(rows)


# ==========================================================================
# Developer documentation for `_validation_consensus_reference`
# ==========================================================================
# Purpose:
#   Creates a conservative human reference and an adjudication template.
#
# Inputs:
#   Parameters in this helper: annotations, manifest.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_consensus_reference(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a conservative human reference and an adjudication template.

    Values are accepted as consensus only when the two independent reviewers
    coded the same category. Disagreements remain unresolved until an explicit
    adjudication file is provided.
    """

    reviewer1, reviewer2 = _validation_paired_annotations(annotations)
    manifest_indexed = manifest.copy()
    manifest_indexed['validation_id'] = manifest_indexed[
        'validation_id'
    ].astype(str)
    manifest_indexed = manifest_indexed.set_index('validation_id')

    shared_ids = reviewer1.index.intersection(
        reviewer2.index
    ).intersection(manifest_indexed.index)

    consensus = pd.DataFrame(index=shared_ids)
    consensus.index.name = 'validation_id'
    consensus['source_report'] = manifest_indexed.reindex(
        shared_ids
    )['source_report']
    consensus['sampling_support_group'] = manifest_indexed.reindex(
        shared_ids
    )['sampling_support_group']
    consensus['sampling_scenario_class'] = manifest_indexed.reindex(
        shared_ids
    )['sampling_scenario_class']
    consensus['sampling_weight'] = manifest_indexed.reindex(
        shared_ids
    )['sampling_weight']

    adjudication_rows: list[dict[str, Any]] = []

    for field in VALIDATION_HUMAN_FIELDS:
        left = reviewer1.reindex(shared_ids)[field].map(_validation_text)
        right = reviewer2.reindex(shared_ids)[field].map(_validation_text)
        same = left.eq(right) & left.ne('')

        consensus[field] = left.where(same, '')
        consensus[f'{field}__reference_status'] = same.map(
            {True: 'human_consensus', False: 'needs_adjudication'}
        )

        disagreement_ids = shared_ids[~same]
        for validation_id in disagreement_ids:
            adjudication_rows.append({
                'validation_id': validation_id,
                'source_report': manifest_indexed.loc[
                    validation_id,
                    'source_report',
                ],
                'sampling_support_group': manifest_indexed.loc[
                    validation_id,
                    'sampling_support_group',
                ],
                'sampling_scenario_class': manifest_indexed.loc[
                    validation_id,
                    'sampling_scenario_class',
                ],
                'field': field,
                'field_label': VALIDATION_FIELD_LABELS.get(field, field),
                'reviewer1': left.loc[validation_id],
                'reviewer2': right.loc[validation_id],
                'adjudicated_value': '',
                'adjudication_notes': '',
            })

    return consensus.reset_index(), pd.DataFrame(adjudication_rows)


# ==========================================================================
# Developer documentation for `_validation_apply_adjudication`
# ==========================================================================
# Purpose:
#   Applies explicit adjudications when a completed adjudication file exists.
#
# Inputs:
#   Parameters in this helper: consensus_reference, project_root.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_apply_adjudication(
    consensus_reference: pd.DataFrame,
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Applies explicit adjudications when a completed adjudication file exists."""

    reference = consensus_reference.copy()
    adjudicated_dir = project_root / 'validation_results' / 'adjudicated'
    adjudication_path = _validation_find_file(
        adjudicated_dir,
        'adjudicated_reference.csv',
    )

    status = {
        'adjudication_file': (
            str(adjudication_path) if adjudication_path is not None else None
        ),
        'adjudication_rows_supplied': 0,
        'adjudication_rows_applied': 0,
        'remaining_unresolved_field_values': 0,
        'reference_type': 'human_consensus_only',
    }

    reference = reference.set_index('validation_id')

    if adjudication_path is not None:
        adjudication = pd.read_csv(adjudication_path)
        required = {'validation_id', 'field', 'adjudicated_value'}
        missing = sorted(required.difference(adjudication.columns))
        if missing:
            raise ValueError(
                'adjudicated_reference.csv is missing required columns: '
                + ', '.join(missing)
            )

        status['adjudication_rows_supplied'] = int(len(adjudication))
        applied = 0
        for _, row in adjudication.iterrows():
            validation_id = _validation_text(row.get('validation_id'))
            field = _validation_text(row.get('field'))
            value = _validation_text(row.get('adjudicated_value'))
            if (
                not validation_id
                or field not in VALIDATION_HUMAN_FIELDS
                or not value
                or validation_id not in reference.index
            ):
                continue

            reference.at[validation_id, field] = value
            reference.at[
                validation_id,
                f'{field}__reference_status',
            ] = 'adjudicated'
            applied += 1

        status['adjudication_rows_applied'] = applied

    unresolved = 0
    for field in VALIDATION_HUMAN_FIELDS:
        unresolved += int(
            reference[field].map(_validation_text).eq('').sum()
        )
    status['remaining_unresolved_field_values'] = unresolved

    if unresolved == 0:
        status['reference_type'] = 'human_consensus_plus_adjudication'
    elif status['adjudication_rows_applied'] > 0:
        status['reference_type'] = 'partially_adjudicated_reference'

    return reference.reset_index(), status


# ==========================================================================
# Developer documentation for `_validation_llm_vs_reference_table`
# ==========================================================================
# Purpose:
#   Compares the LLM with the available consensus or adjudicated reference.
#
# Inputs:
#   Parameters in this helper: human_reference, manifest.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_llm_vs_reference_table(
    human_reference: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compares the LLM with the available consensus or adjudicated reference."""

    reference = human_reference.copy()
    reference['validation_id'] = reference['validation_id'].astype(str)
    reference = reference.set_index('validation_id')

    manifest_indexed = manifest.copy()
    manifest_indexed['validation_id'] = manifest_indexed[
        'validation_id'
    ].astype(str)
    manifest_indexed = manifest_indexed.set_index('validation_id')

    shared_ids = reference.index.intersection(manifest_indexed.index)
    weights = manifest_indexed.reindex(shared_ids)['sampling_weight']

    rows: list[dict[str, Any]] = []
    for human_field, llm_field in VALIDATION_LLM_FIELD_MAP.items():
        metrics = _validation_agreement_metrics(
            reference.reindex(shared_ids)[human_field],
            manifest_indexed.reindex(shared_ids)[llm_field],
            weights,
        )
        reference_status = reference.reindex(shared_ids)[
            f'{human_field}__reference_status'
        ].fillna('')
        rows.append({
            'field': human_field,
            'field_label': VALIDATION_FIELD_LABELS.get(
                human_field,
                human_field,
            ),
            'llm_field': llm_field,
            'n_reference_available': metrics['n'],
            'n_human_consensus': int(
                reference_status.eq('human_consensus').sum()
            ),
            'n_adjudicated': int(
                reference_status.eq('adjudicated').sum()
            ),
            'exact_agreement': metrics['exact_agreement'],
            'cohen_kappa': metrics['cohen_kappa'],
            'weighted_exact_agreement': metrics[
                'weighted_exact_agreement'
            ],
            'weighted_cohen_kappa': metrics['weighted_cohen_kappa'],
        })

    return pd.DataFrame(rows)


# ==========================================================================
# Developer documentation for `_validation_llm_reference_by_support`
# ==========================================================================
# Purpose:
#   Reports LLM versus human reference agreement within rule support strata.
#
# Inputs:
#   Parameters in this helper: human_reference, manifest.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_llm_reference_by_support(
    human_reference: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Reports LLM versus human reference agreement within rule support strata."""

    reference = human_reference.copy()
    reference['validation_id'] = reference['validation_id'].astype(str)
    manifest_copy = manifest.copy()
    manifest_copy['validation_id'] = manifest_copy[
        'validation_id'
    ].astype(str)

    merged = reference.merge(
        manifest_copy,
        on='validation_id',
        how='inner',
        suffixes=('', '_manifest'),
    )

    rows: list[dict[str, Any]] = []
    support_order = ['high', 'medium', 'low']

    for support_group in support_order:
        subset = merged.loc[
            merged['sampling_support_group'].astype(str).eq(support_group)
        ].copy()
        if subset.empty:
            continue

        for human_field, llm_field in VALIDATION_LLM_FIELD_MAP.items():
            metrics = _validation_agreement_metrics(
                subset[human_field],
                subset[llm_field],
                subset['sampling_weight'],
            )
            rows.append({
                'sampling_support_group': support_group,
                'field': human_field,
                'field_label': VALIDATION_FIELD_LABELS.get(
                    human_field,
                    human_field,
                ),
                'n_reference_available': metrics['n'],
                'exact_agreement': metrics['exact_agreement'],
                'cohen_kappa': metrics['cohen_kappa'],
                'weighted_exact_agreement': metrics[
                    'weighted_exact_agreement'
                ],
                'weighted_cohen_kappa': metrics[
                    'weighted_cohen_kappa'
                ],
            })

    return pd.DataFrame(rows)


# ==========================================================================
# Developer documentation for `_validation_as_bool`
# ==========================================================================
# Purpose:
#   Converts stored availability flags into booleans.
#
# Inputs:
#   Parameters in this helper: value.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_as_bool(value: Any) -> bool:
    """Converts stored availability flags into booleans."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = _validation_text(value).lower()
    return text in {'true', '1', 'yes', 'y'}


# ==========================================================================
# Developer documentation for `_validation_source_presence_vs_llm`
# ==========================================================================
# Purpose:
#   Decomposes source absence and extraction recovery for fine context fields.
#
# Inputs:
#   Parameters in this helper: annotations, manifest, human_reference.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_source_presence_vs_llm(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    human_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Decomposes source absence and extraction recovery for fine context fields."""

    mappings = {
        'v1_lane_presence': 'v1_lane_llm_available',
        'v2_lane_presence': 'v2_lane_llm_available',
        'v1_speed_presence': 'v1_speed_llm_available',
        'v2_speed_presence': 'v2_speed_llm_available',
        'direction_presence': 'direction_llm_available',
    }

    manifest_copy = manifest.copy()
    manifest_copy['validation_id'] = manifest_copy[
        'validation_id'
    ].astype(str)

    source_tables: list[tuple[str, pd.DataFrame]] = []
    for reviewer in VALIDATION_REVIEWERS:
        reviewer_df = annotations.loc[
            annotations['reviewer_id'].eq(reviewer)
        ].copy()
        reviewer_df['validation_id'] = reviewer_df[
            'validation_id'
        ].astype(str)
        source_tables.append((reviewer, reviewer_df))

    reference_df = human_reference.copy()
    reference_df['validation_id'] = reference_df[
        'validation_id'
    ].astype(str)
    source_tables.append(('human_reference', reference_df))

    rows: list[dict[str, Any]] = []

    for source_name, source_df in source_tables:
        merged = source_df.merge(
            manifest_copy,
            on='validation_id',
            how='inner',
            suffixes=('', '_manifest'),
        )

        for presence_field, llm_available_field in mappings.items():
            outcome_rows: list[dict[str, Any]] = []

            for _, row in merged.iterrows():
                source_state = _validation_text(
                    row.get(presence_field)
                ).lower()
                if not source_state:
                    continue

                llm_available = _validation_as_bool(
                    row.get(llm_available_field)
                )

                if source_state == 'present' and llm_available:
                    outcome = 'source_present_llm_recovered'
                elif source_state == 'present' and not llm_available:
                    outcome = 'source_present_llm_missed'
                elif source_state == 'not_stated' and not llm_available:
                    outcome = 'source_absent_llm_abstained'
                elif source_state == 'not_stated' and llm_available:
                    outcome = 'source_absent_llm_returned_value'
                else:
                    outcome = 'source_ambiguous'

                outcome_rows.append({
                    'outcome': outcome,
                    'sampling_weight': float(
                        row.get('sampling_weight', 1.0) or 1.0
                    ),
                })

            if not outcome_rows:
                continue

            outcomes = pd.DataFrame(outcome_rows)
            total = int(len(outcomes))
            total_weight = float(outcomes['sampling_weight'].sum())

            for outcome, outcome_df in outcomes.groupby('outcome'):
                count = int(len(outcome_df))
                weighted_count = float(
                    outcome_df['sampling_weight'].sum()
                )
                rows.append({
                    'reference_source': source_name,
                    'field': presence_field,
                    'field_label': VALIDATION_FIELD_LABELS.get(
                        presence_field,
                        presence_field,
                    ),
                    'outcome': outcome,
                    'count': count,
                    'share': float(count / total) if total else None,
                    'weighted_share': (
                        weighted_count / total_weight
                        if total_weight > 0
                        else None
                    ),
                })

    return pd.DataFrame(rows)


# ==========================================================================
# Developer documentation for `_validation_derive_scenario`
# ==========================================================================
# Purpose:
#   Derives the deterministic taxonomy from one human coding record.
#
# Inputs:
#   Parameters in this helper: row.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_derive_scenario(row: pd.Series) -> str:
    """Derives the deterministic taxonomy from one human coding record.

    The implementation mirrors the ordering of the main research taxonomy while
    using the human coded source variables directly.
    """

    road_user = _validation_text(
        row.get('road_user_type_human')
    ).lower()

    av_move = _validation_first_present(
        row.get('move_v1_checkbox_human'),
        row.get('v1_move_narrative_human'),
    ).lower()
    other_move = _validation_first_present(
        row.get('move_v2_checkbox_human'),
        row.get('v2_move_narrative_human'),
    ).lower()
    collision = _validation_first_present(
        row.get('collision_v1_human'),
        row.get('collision_v2_human'),
    ).lower()

    v1_intersection = _validation_text(
        row.get('v1_intersection_human')
    ).lower()
    v2_intersection = _validation_text(
        row.get('v2_intersection_human')
    ).lower()

    intersection = (
        'true'
        if 'true' in {v1_intersection, v2_intersection}
        else 'false'
        if 'false' in {v1_intersection, v2_intersection}
        else 'not_stated'
    )

    parked_cue = _validation_text(
        row.get('parked_or_curbside_cue_human')
    ).lower() == 'yes'
    obstruction_cue = _validation_text(
        row.get('obstruction_yield_blocked_cue_human')
    ).lower() == 'yes'

    if road_user in VALIDATION_VULNERABLE_ROAD_USERS:
        return 'vulnerable_road_user_interaction'

    if av_move == 'stop' and collision == 'rear_end':
        return 'AV_stopped_rear_end'

    if (
        intersection == 'true'
        and collision in {'broadside', 'side_swipe', 'head_on'}
    ):
        return 'intersection_lateral_conflict'

    if av_move == 'straight' and other_move in VALIDATION_TURN_MOVEMENTS:
        return 'turn_across_path_conflict'

    if (
        av_move in {'change_lane', 'merging'}
        or other_move in {'change_lane', 'merging'}
        or collision == 'side_swipe'
    ):
        return 'lane_change_or_merge_conflict'

    if parked_cue or collision == 'object':
        return 'curbside_or_parked_vehicle_conflict'

    if av_move == 'stop' and obstruction_cue:
        return 'low_speed_stop_or_obstruction_case'

    return 'other_or_ambiguous'


# ==========================================================================
# Developer documentation for `_validation_scenario_analysis`
# ==========================================================================
# Purpose:
#   Derives scenario classes from both human coders and summarises agreement.
#
# Inputs:
#   Parameters in this helper: annotations, manifest, human_reference.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Validation helpers preserve the two original reviewer coding sets as separate observations.
#   Reviewer specific LLM agreement is reported separately and must not be averaged into one accuracy score.
#   Cohen's kappa is reported together with exact agreement because prevalence can affect kappa.
#   The validation sample is stratified, so optional sampling weights are retained for corpus oriented estimates.
#   Missing, ambiguous, and not stated values are handled explicitly rather than silently imputed.
#   Disagreement records are preserved even when a later adjudication value becomes available.
#   No reviewer should be treated as a sole gold standard for a field that remains disputed.
#   Any adjudicated reference must remain separate from the original independent annotations.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _validation_scenario_analysis(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    human_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derives scenario classes from both human coders and summarises agreement."""

    reviewer1, reviewer2 = _validation_paired_annotations(annotations)

    manifest_indexed = manifest.copy()
    manifest_indexed['validation_id'] = manifest_indexed[
        'validation_id'
    ].astype(str)
    manifest_indexed = manifest_indexed.set_index('validation_id')

    shared_ids = reviewer1.index.intersection(
        reviewer2.index
    ).intersection(manifest_indexed.index)

    detail = pd.DataFrame(index=shared_ids)
    detail.index.name = 'validation_id'
    detail['source_report'] = manifest_indexed.reindex(
        shared_ids
    )['source_report']
    detail['sampling_support_group'] = manifest_indexed.reindex(
        shared_ids
    )['sampling_support_group']
    detail['sampling_weight'] = manifest_indexed.reindex(
        shared_ids
    )['sampling_weight']
    detail['scenario_class_llm'] = manifest_indexed.reindex(
        shared_ids
    )['scenario_class_llm']
    detail['scenario_class_reviewer1'] = reviewer1.reindex(
        shared_ids
    ).apply(_validation_derive_scenario, axis=1)
    detail['scenario_class_reviewer2'] = reviewer2.reindex(
        shared_ids
    ).apply(_validation_derive_scenario, axis=1)

    reference = human_reference.copy()
    reference['validation_id'] = reference['validation_id'].astype(str)
    reference = reference.set_index('validation_id').reindex(shared_ids)

    reference_complete = pd.Series(True, index=shared_ids)
    for field in VALIDATION_SCENARIO_INPUT_FIELDS:
        reference_complete &= reference[field].map(
            _validation_text
        ).ne('')

    detail['scenario_class_reference'] = ''
    if reference_complete.any():
        derived_reference = reference.loc[
            reference_complete,
            VALIDATION_SCENARIO_INPUT_FIELDS,
        ].apply(_validation_derive_scenario, axis=1)
        detail.loc[
            reference_complete,
            'scenario_class_reference',
        ] = derived_reference

    weights = detail['sampling_weight']

    comparison_specs = [
        (
            'reviewer1_vs_reviewer2',
            detail['scenario_class_reviewer1'],
            detail['scenario_class_reviewer2'],
        ),
        (
            'llm_vs_reviewer1',
            detail['scenario_class_llm'],
            detail['scenario_class_reviewer1'],
        ),
        (
            'llm_vs_reviewer2',
            detail['scenario_class_llm'],
            detail['scenario_class_reviewer2'],
        ),
    ]

    if detail['scenario_class_reference'].astype(str).str.strip().ne('').any():
        comparison_specs.append((
            'llm_vs_human_reference',
            detail['scenario_class_llm'],
            detail['scenario_class_reference'],
        ))

    summary_rows: list[dict[str, Any]] = []
    for comparison, left, right in comparison_specs:
        metrics = _validation_agreement_metrics(left, right, weights)
        summary_rows.append({
            'comparison': comparison,
            'n': metrics['n'],
            'exact_agreement': metrics['exact_agreement'],
            'cohen_kappa': metrics['cohen_kappa'],
            'weighted_exact_agreement': metrics[
                'weighted_exact_agreement'
            ],
            'weighted_cohen_kappa': metrics['weighted_cohen_kappa'],
        })

    return detail.reset_index(), pd.DataFrame(summary_rows)


# ==========================================================================
# Developer documentation for `_movement_contradiction_sensitivity_table`
# ==========================================================================
# Purpose:
#   Quantifies how the scenario distribution changes with movement evidence.
#
# Inputs:
#   Parameters in this helper: research_df.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   Sensitivity helpers test whether headline scenario patterns depend on evidence quality choices.
#   The baseline distribution remains the reference against which restricted subsets are compared.
#   Total variation distance summarises distributional change without implying causal importance.
#   Percentage point shifts are kept separate from relative percentage changes.
#   Contradictory movement evidence can reflect source disagreement, extraction error, or normalisation error.
#   Restricted subsets therefore support robustness interpretation rather than automatic data cleaning.
#   Counts and denominators must be retained so reported percentages can be independently reproduced.
#   The function should remain deterministic for the same frozen analytical dataset.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _movement_contradiction_sensitivity_table(
    research_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quantifies how the scenario distribution changes with movement evidence."""

    if (
        'movement_field_agreement' not in research_df.columns
        or 'scenario_class' not in research_df.columns
    ):
        return pd.DataFrame(), {}

    specifications = {
        'baseline_all_reports': pd.Series(True, index=research_df.index),
        'exclude_contradictory_movement': research_df[
            'movement_field_agreement'
        ].ne('contradictory'),
        'movement_exact_or_compatible_only': research_df[
            'movement_field_agreement'
        ].isin({'exact_agreement', 'compatible_agreement'}),
        'contradictory_movement_only': research_df[
            'movement_field_agreement'
        ].eq('contradictory'),
    }

    all_classes = sorted(
        research_df['scenario_class'].fillna('NA').astype(str).unique()
    )

    rows: list[dict[str, Any]] = []
    shares_by_spec: dict[str, dict[str, float]] = {}

    for specification, mask in specifications.items():
        subset = research_df.loc[mask].copy()
        counts = subset['scenario_class'].fillna('NA').astype(str).value_counts()
        n = int(len(subset))
        shares = {
            scenario_class: (
                float(counts.get(scenario_class, 0) / n)
                if n else 0.0
            )
            for scenario_class in all_classes
        }
        shares_by_spec[specification] = shares

        for scenario_class in all_classes:
            rows.append({
                'specification': specification,
                'specification_n': n,
                'scenario_class': scenario_class,
                'count': int(counts.get(scenario_class, 0)),
                'share': shares[scenario_class],
            })

    baseline = shares_by_spec.get('baseline_all_reports', {})
    non_contradictory = shares_by_spec.get(
        'exclude_contradictory_movement',
        {},
    )
    supported = shares_by_spec.get(
        'movement_exact_or_compatible_only',
        {},
    )

    # ==========================================================================
    # Developer documentation for `_total_variation`
    # ==========================================================================
    # Purpose:
    #   Implements the `_total_variation` step used by the analysis pipeline.
    #
    # Inputs:
    #   Parameters in this helper: left, right.
    #   Callers should pass already resolved project objects where possible so configuration and data provenance
    #   remain explicit.
    #
    # Output contract:
    #   The return value is the documented interface for downstream code. Temporary local variables are
    #   implementation details.
    #   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or
    #   negative result.
    #
    # Scientific and maintenance notes:
    #   Sensitivity helpers test whether headline scenario patterns depend on evidence quality choices.
    #   The baseline distribution remains the reference against which restricted subsets are compared.
    #   Total variation distance summarises distributional change without implying causal importance.
    #   Percentage point shifts are kept separate from relative percentage changes.
    #   Contradictory movement evidence can reflect source disagreement, extraction error, or normalisation error.
    #   Restricted subsets therefore support robustness interpretation rather than automatic data cleaning.
    #   Counts and denominators must be retained so reported percentages can be independently reproduced.
    #   The function should remain deterministic for the same frozen analytical dataset.
    #
    # Change control:
    #   When behaviour changes, update the corresponding README, manuscript description, and validation
    #   documentation where relevant.
    #   Keep generated categories and reported statistics traceable to the source fields used to create them.
    #   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
    #   explicit.
    # ==========================================================================

    def _total_variation(
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        labels = set(left) | set(right)
        return 0.5 * sum(
            abs(left.get(label, 0.0) - right.get(label, 0.0))
            for label in labels
        )

    # ==========================================================================
    # Developer documentation for `_largest_shift`
    # ==========================================================================
    # Purpose:
    #   Implements the `_largest_shift` step used by the analysis pipeline.
    #
    # Inputs:
    #   Parameters in this helper: left, right.
    #   Callers should pass already resolved project objects where possible so configuration and data provenance
    #   remain explicit.
    #
    # Output contract:
    #   The return value is the documented interface for downstream code. Temporary local variables are
    #   implementation details.
    #   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or
    #   negative result.
    #
    # Scientific and maintenance notes:
    #   Sensitivity helpers test whether headline scenario patterns depend on evidence quality choices.
    #   The baseline distribution remains the reference against which restricted subsets are compared.
    #   Total variation distance summarises distributional change without implying causal importance.
    #   Percentage point shifts are kept separate from relative percentage changes.
    #   Contradictory movement evidence can reflect source disagreement, extraction error, or normalisation error.
    #   Restricted subsets therefore support robustness interpretation rather than automatic data cleaning.
    #   Counts and denominators must be retained so reported percentages can be independently reproduced.
    #   The function should remain deterministic for the same frozen analytical dataset.
    #
    # Change control:
    #   When behaviour changes, update the corresponding README, manuscript description, and validation
    #   documentation where relevant.
    #   Keep generated categories and reported statistics traceable to the source fields used to create them.
    #   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
    #   explicit.
    # ==========================================================================

    def _largest_shift(
        left: dict[str, float],
        right: dict[str, float],
    ) -> tuple[str, float]:
        labels = set(left) | set(right)
        if not labels:
            return 'NA', 0.0
        scenario = max(
            labels,
            key=lambda label: abs(
                left.get(label, 0.0) - right.get(label, 0.0)
            ),
        )
        return (
            scenario,
            float(right.get(scenario, 0.0) - left.get(scenario, 0.0)),
        )

    largest_non_contradictory = _largest_shift(
        baseline,
        non_contradictory,
    )
    largest_supported = _largest_shift(baseline, supported)

    baseline_top = (
        max(baseline, key=baseline.get) if baseline else 'NA'  # type: ignore
    )
    non_contradictory_top = (
        max(non_contradictory, key=non_contradictory.get)  # type: ignore
        if non_contradictory
        else 'NA'
    )
    supported_top = (
        max(supported, key=supported.get) if supported else 'NA'  # type: ignore
    )

    summary = {
        'baseline_n': int(len(research_df)),
        'contradictory_n': int(
            research_df['movement_field_agreement'].eq(
                'contradictory'
            ).sum()
        ),
        'contradictory_share': float(
            research_df['movement_field_agreement'].eq(
                'contradictory'
            ).mean()
        ),
        'non_contradictory_n': int(
            specifications['exclude_contradictory_movement'].sum()
        ),
        'movement_supported_n': int(
            specifications['movement_exact_or_compatible_only'].sum()
        ),
        'baseline_top_scenario': baseline_top,
        'baseline_top_share': baseline.get(baseline_top, 0.0),
        'non_contradictory_top_scenario': non_contradictory_top,
        'non_contradictory_top_share': non_contradictory.get(
            non_contradictory_top,
            0.0,
        ),
        'movement_supported_top_scenario': supported_top,
        'movement_supported_top_share': supported.get(
            supported_top,
            0.0,
        ),
        'tvd_baseline_vs_non_contradictory': _total_variation(
            baseline,
            non_contradictory,
        ),
        'tvd_baseline_vs_movement_supported': _total_variation(
            baseline,
            supported,
        ),
        'largest_share_shift_non_contradictory_scenario': (
            largest_non_contradictory[0]
        ),
        'largest_share_shift_non_contradictory': (
            largest_non_contradictory[1]
        ),
        'largest_share_shift_supported_scenario': largest_supported[0],
        'largest_share_shift_supported': largest_supported[1],
    }

    return pd.DataFrame(rows), summary


# ==========================================================================
# Developer documentation for `_other_or_ambiguous_validation_summary`
# ==========================================================================
# Purpose:
#   Summarises the large other or ambiguous taxonomy category.
#
# Inputs:
#   Parameters in this helper: research_df.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This helper isolates one repeated operation so the main pipeline remains easier to audit.
#   The implementation favours explicit missing value handling over implicit coercion.
#   Return values should remain stable because downstream analysis may rely on their exact type.
#   The helper should avoid modifying caller owned data unless mutation is clearly documented.
#   Deterministic behaviour is preferred for reproducible research outputs.
#   Error handling should preserve useful context while avoiding silent data fabrication.
#   Keep transformation logic close to the field definition used elsewhere in the pipeline.
#   Any change to category semantics should be reflected in downstream documentation.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def _other_or_ambiguous_validation_summary(
    research_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarises the large other or ambiguous taxonomy category."""

    if 'scenario_class' not in research_df.columns:
        return pd.DataFrame(), {}

    ambiguous = research_df.loc[
        research_df['scenario_class'].eq('other_or_ambiguous')
    ].copy()

    if ambiguous.empty:
        return pd.DataFrame(), {
            'n': 0,
            'share': 0.0,
        }

    rows: list[dict[str, Any]] = []

    for column in [
        'scenario_rule_support_group',
        'movement_field_agreement',
        'scenario_candidate_count',
        'report_completeness_band',
    ]:
        if column not in ambiguous.columns:
            continue
        counts = (
            ambiguous[column]
            .fillna('NA')
            .astype(str)
            .value_counts(dropna=False)
        )
        for category, count in counts.items():
            rows.append({
                'dimension': column,
                'category': category,
                'count': int(count),
                'share_within_other_or_ambiguous': float(
                    count / len(ambiguous)
                ),
            })

    summary = {
        'n': int(len(ambiguous)),
        'share': float(len(ambiguous) / max(len(research_df), 1)),
        'high_support_n': int(
            ambiguous.get(
                'scenario_rule_support_group',
                pd.Series(index=ambiguous.index, dtype='object'),
            ).eq('high').sum()
        ),
        'medium_support_n': int(
            ambiguous.get(
                'scenario_rule_support_group',
                pd.Series(index=ambiguous.index, dtype='object'),
            ).eq('medium').sum()
        ),
        'low_support_n': int(
            ambiguous.get(
                'scenario_rule_support_group',
                pd.Series(index=ambiguous.index, dtype='object'),
            ).eq('low').sum()
        ),
        'contradictory_movement_n': int(
            ambiguous.get(
                'movement_field_agreement',
                pd.Series(index=ambiguous.index, dtype='object'),
            ).eq('contradictory').sum()
        ),
    }

    return pd.DataFrame(rows), summary


# Developer documentation for `analyse_human_validation_results`
# ==========================================================================
# Purpose:
#   Analyses completed human coding and writes derived validation outputs.
#
# Inputs:
#   Parameters in this helper: project_root, research_df.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   This orchestration step reads the frozen human validation artefacts and writes derived analysis outputs.
#   The original independent reviewer annotations are inputs and must never be overwritten.
#   Generated consensus or adjudicated files are derivative products and remain separate from raw coding.
#   Reviewer specific agreement tables are retained even when a later reference table is available.
#   All output paths should remain deterministic under the configured validation analysis directory.
#   A missing optional adjudication file should not invalidate the completed independent validation analysis.
#   Status metadata records whether the two reviewers covered the same frozen validation set.
#   The completed reviewer order record is historical evidence and must not be regenerated retrospectively.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================

def analyse_human_validation_results(
    *,
    project_root: Path,
    research_df: pd.DataFrame,
) -> dict[str, Any]:
    """Analyses completed human coding and writes derived validation outputs.

    The function is intentionally non destructive. The frozen reviewer exports
    under ``validation_results/complete`` are read only. All derived files are
    written to ``validation_results/analysis``.
    """

    loaded = _validation_load_completed_data(project_root)
    if loaded is None:
        return {}

    annotations, manifest, _orders, validation_status = loaded
    analysis_dir = _resolve_validation_analysis_dir(project_root)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    interrater, disagreement_template = _validation_interrater_table(
        annotations,
        manifest,
    )
    llm_vs_reviewer = _validation_llm_vs_reviewer_table(
        annotations,
        manifest,
    )

    consensus_reference, consensus_disagreements = (
        _validation_consensus_reference(annotations, manifest)
    )

    # Use the consensus generated disagreement table as the adjudication
    # template because it also includes disagreement in non headline fields.
    if not consensus_disagreements.empty:
        disagreement_template = consensus_disagreements

    human_reference, reference_status = _validation_apply_adjudication(
        consensus_reference,
        project_root,
    )

    llm_vs_reference = _validation_llm_vs_reference_table(
        human_reference,
        manifest,
    )
    by_support = _validation_llm_reference_by_support(
        human_reference,
        manifest,
    )
    source_presence_vs_llm = _validation_source_presence_vs_llm(
        annotations,
        manifest,
        human_reference,
    )
    scenario_detail, scenario_summary = _validation_scenario_analysis(
        annotations,
        manifest,
        human_reference,
    )

    movement_sensitivity, movement_summary = (
        _movement_contradiction_sensitivity_table(research_df)
    )
    ambiguous_detail, ambiguous_summary = (
        _other_or_ambiguous_validation_summary(research_df)
    )

    interrater.to_csv(
        analysis_dir / 'interrater_agreement.csv',
        index=False,
    )
    llm_vs_reviewer.to_csv(
        analysis_dir / 'llm_vs_each_reviewer.csv',
        index=False,
    )
    consensus_reference.to_csv(
        analysis_dir / 'human_consensus_reference.csv',
        index=False,
    )
    disagreement_template.to_csv(
        analysis_dir / 'adjudication_required.csv',
        index=False,
    )
    human_reference.to_csv(
        analysis_dir / 'human_reference_current.csv',
        index=False,
    )
    llm_vs_reference.to_csv(
        analysis_dir / 'llm_vs_human_reference.csv',
        index=False,
    )
    by_support.to_csv(
        analysis_dir / 'llm_vs_human_reference_by_rule_support.csv',
        index=False,
    )
    source_presence_vs_llm.to_csv(
        analysis_dir / 'source_presence_vs_llm.csv',
        index=False,
    )
    scenario_detail.to_csv(
        analysis_dir / 'human_derived_scenario_detail.csv',
        index=False,
    )
    scenario_summary.to_csv(
        analysis_dir / 'human_derived_scenario_agreement.csv',
        index=False,
    )
    movement_sensitivity.to_csv(
        analysis_dir / 'movement_contradiction_sensitivity.csv',
        index=False,
    )
    ambiguous_detail.to_csv(
        analysis_dir / 'other_or_ambiguous_breakdown.csv',
        index=False,
    )

    summary = {
        'validation_status': validation_status,
        'reference_status': reference_status,
        'sample_support_counts': {
            str(key): int(value)
            for key, value in (
                manifest['sampling_support_group']
                .fillna('NA')
                .astype(str)
                .value_counts()
                .items()
            )
        },
        'sample_scenario_counts': {
            str(key): int(value)
            for key, value in (
                manifest['sampling_scenario_class']
                .fillna('NA')
                .astype(str)
                .value_counts()
                .items()
            )
        },
        'movement_contradiction_sensitivity': movement_summary,
        'other_or_ambiguous': ambiguous_summary,
    }

    (analysis_dir / 'human_validation_summary.json').write_text(
        json.dumps(summary, indent=2, default=str),
        encoding='utf-8',
    )

    logger.info(
        'Human validation analysis written to {}.',
        analysis_dir,
    )
    log_kv_block(
        logger,
        'Human validation status',
        {
            'validation_set_id': validation_status.get(
                'validation_set_id',
                'NA',
            ),
            'manifest_n': validation_status.get('manifest_n', 0),
            'reviewer1_n': validation_status.get(
                'reviewer_counts',
                {},
            ).get('reviewer1', 0),
            'reviewer2_n': validation_status.get(
                'reviewer_counts',
                {},
            ).get('reviewer2', 0),
            'same_validation_set': validation_status.get(
                'same_validation_set',
                False,
            ),
            'same_presentation_order': validation_status.get(
                'same_presentation_order',
            ),
            'reference_type': reference_status.get('reference_type'),
            'unresolved_reference_values': reference_status.get(
                'remaining_unresolved_field_values',
                0,
            ),
        },
    )

    return {
        'analysis_dir': str(analysis_dir),
        'interrater_agreement': str(
            analysis_dir / 'interrater_agreement.csv'
        ),
        'llm_vs_each_reviewer': str(
            analysis_dir / 'llm_vs_each_reviewer.csv'
        ),
        'adjudication_required': str(
            analysis_dir / 'adjudication_required.csv'
        ),
        'llm_vs_human_reference': str(
            analysis_dir / 'llm_vs_human_reference.csv'
        ),
        'scenario_agreement': str(
            analysis_dir / 'human_derived_scenario_agreement.csv'
        ),
        'source_presence_vs_llm': str(
            analysis_dir / 'source_presence_vs_llm.csv'
        ),
        'movement_sensitivity': str(
            analysis_dir / 'movement_contradiction_sensitivity.csv'
        ),
        'reference_status': reference_status,
    }


# ==========================================================================
# Developer documentation for `main`
# ==========================================================================
# Purpose:
#   Runs the full analysis pipeline.
#
# Inputs:
#   Parameters in this helper: no explicit parameters.
#   Callers should pass already resolved project objects where possible so configuration and data provenance remain
#   explicit.
#
# Output contract:
#   The return value is the documented interface for downstream code. Temporary local variables are implementation
#   details.
#   Missing or unavailable information must remain distinguishable from a valid zero, empty category, or negative
#   result.
#
# Scientific and maintenance notes:
#   The main entry point coordinates configuration, parsing, research derivation, exports, plots, and validation.
#   Each stage should operate on an explicit dataframe so analytical provenance remains inspectable.
#   The full 971 report analytical corpus is retained for the primary descriptive analyses.
#   Plot specific filtering must not be confused with deletion from the analytical corpus.
#   Figure display relabelling is applied only to plotting copies.
#   Automated robustness checks measure internal stability and are distinct from human agreement.
#   Human validation outputs are loaded from the frozen completed validation directory when available.
#   All generated files should be reproducible from the same input data, configuration, and code version.
#
# Change control:
#   When behaviour changes, update the relevant configuration and validation documentation.
#   Keep generated categories and reported statistics traceable to the source fields used to create them.
#   Do not add hidden heuristics merely to force a desired result; analytical rules must remain
#   explicit.
# ==========================================================================


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

    # Figure 2 uses transportation friendly display labels. Keep the
    # analytical data unchanged and apply the relabelling only to plot copies.
    # In particular, values such as ``intersection_roadway`` are displayed as
    # ``intersection`` and ``non_intersection_roadway`` as ``road_segment``.
    plot_research_df = _prepare_figure_display_data(research_df)
    plot_filtered_df = _prepare_figure_display_data(filtered_df)

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
        'scenario_rule_support_group',
        'scenario_candidate_count',
        'movement_field_agreement',
        'reported_injury_status',
        'report_period',
        'manufacturer_group',
    ] + [field for field in plot_fields if field in filtered_df.columns] \
      + [field for field in story_plot_fields if field in filtered_df.columns]
    plot_input_columns = [
        column for column in plot_input_columns if column in filtered_df.columns
    ]
    save_dataframe(
        plot_filtered_df[plot_input_columns].copy(),
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

    # Analyse the completed two reviewer human validation when the frozen
    # exports are available under validation_results/complete. This step is
    # non destructive and writes all derived validation outputs to
    # validation_results/analysis.
    human_validation_outputs = analyse_human_validation_results(
        project_root=_resolve_project_root(),
        research_df=research_df,
    )

    markdown_report = format_research_markdown(research_summary, config=config)
    save_markdown(markdown_report, output_dirs.base / 'run_report.md')

    log_kv_block(logger, 'Core outputs written', {
        'cleaned_events': output_dirs.base / 'cleaned_events.csv',
        'accident_overview': output_dirs.base / 'accident_overview.csv',
        'plot_input_filtered': output_dirs.base / 'plot_input_filtered.csv',
        'research_summary': output_dirs.base / 'research_summary.json',
        'validation_sample': output_dirs.base / 'validation_sample.csv',
        'human_validation_analysis': (
            human_validation_outputs.get('analysis_dir', 'not_available')
            if human_validation_outputs
            else 'not_available'
        ),
        'run_report': output_dirs.base / 'run_report.md',
        'drop_reason_summary': output_dirs.base / 'drop_reason_summary.csv',
        'source_disagreement_summary': output_dirs.base / 'source_disagreement_summary.csv',
        'movement_inconsistency_audit': output_dirs.base / 'movement_inconsistency_audit.csv',
        'other_or_ambiguous_review': output_dirs.base / 'other_or_ambiguous_review.csv',
        'corpus_manifest': output_dirs.base / 'corpus_manifest.csv',
        'report_context_unavailability': output_dirs.base / 'report_context_unavailability.csv',
        'external_context_unavailability': output_dirs.base / 'external_context_unavailability.csv',
        'taxonomy_sensitivity': output_dirs.base / 'taxonomy_sensitivity.csv',
        'taxonomy_agreement': output_dirs.base / 'taxonomy_agreement.csv',
        'taxonomy_rule_overlap': output_dirs.base / 'taxonomy_rule_overlap.csv',
        'scenario_by_av_mode': output_dirs.base / 'scenario_by_av_mode.csv',
        'scenario_by_period': output_dirs.base / 'scenario_by_period.csv',
        'scenario_by_manufacturer': output_dirs.base / 'scenario_by_manufacturer.csv',
        'scenario_by_reported_injury': output_dirs.base / 'scenario_by_reported_injury.csv',
        'manufacturer_leave_one_out': output_dirs.base / 'manufacturer_leave_one_out.csv',
    })

    # Build and export all overview, histogram, and research figures.
    # Figure 4 uses grouped within-scenario attribution percentages rather
    # than another count-based view. This keeps it complementary to
    # Figure 3 while preserving a visible attribution legend.
    summary_plots.create_accountability_by_taxonomy_figure = (
        _create_accountability_by_taxonomy_figure
    )
    manifest = summary_plots.create_all_plots(
        parsed_df=plot_research_df,
        filtered_df=plot_filtered_df,
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
        'top_post_extraction_unavailability': research_summary.get(
            'post_extraction_top_unavailability',
            {},
        ),
        'blame_distribution': research_summary.get('blame_distribution', {}),
        'provenance_mean_availability': research_summary.get('provenance_mean_availability', {}),
        'movement_consistency_distribution': research_summary.get('movement_consistency_distribution', {}),
        'movement_field_agreement_distribution': research_summary.get('movement_field_agreement_distribution', {}),
        'scenario_determinability_distribution': research_summary.get('scenario_determinability_distribution', {}),
        'scenario_rule_support_distribution': research_summary.get('scenario_rule_support_distribution', {}),
        'scenario_rule_overlap_distribution': research_summary.get('scenario_rule_overlap_distribution', {}),
        'reported_injury_distribution': research_summary.get('reported_injury_distribution', {}),
        'blame_field_completeness_distribution': research_summary.get('blame_field_completeness_distribution', {}),
        'average_external_context_score': research_summary.get('average_external_context_score', 0.0),
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
