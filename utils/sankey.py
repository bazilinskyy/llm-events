from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from utils.labels import humanize_text
from utils.normalise import is_missing, normalise_category

logger = logging.getLogger(__name__)


def resolve_plot_fields(df: pd.DataFrame, include_plot_fields: list[str], exclude_plot_fields: list[str]) -> list[str]:
    fields = [field for field in include_plot_fields if field in df.columns]
    if exclude_plot_fields:
        excluded = set(exclude_plot_fields)
        fields = [field for field in fields if field not in excluded]
    if len(fields) < 2:
        raise ValueError('Need at least two plot fields after applying include and exclude settings.')
    return fields


def _build_non_missing_mask(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    if not fields:
        return pd.Series(True, index=df.index)
    non_missing_by_field = {field: df[field].map(lambda value: not is_missing(value)) for field in fields}
    mask_df = pd.DataFrame(non_missing_by_field, index=df.index)
    return mask_df.all(axis=1)


def apply_plot_filters(parsed_df: pd.DataFrame, plot_fields: list[str], filter_rows_with_na: bool, na_filter_fields: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = parsed_df.copy()
    dropped_empty_output = int(getattr(parsed_df, 'attrs', {}).get('dropped_empty_output', 0))
    relevant_filter_fields = [field for field in na_filter_fields if field in df.columns]

    dropped_for_plot_na = 0
    if filter_rows_with_na and relevant_filter_fields:
        mask = _build_non_missing_mask(df, relevant_filter_fields)
        dropped_for_plot_na = int((~mask).sum())
        df = df.loc[mask].copy()
        logger.info('Dropped %s rows due to missing values in critical fields: %s', dropped_for_plot_na, ', '.join(relevant_filter_fields))

    for field in plot_fields:
        if field in df.columns:
            df[field] = df[field].map(normalise_category)

    filter_report = {
        'dropped_empty_output': dropped_empty_output,
        'filter_rows_with_na': filter_rows_with_na,
        'na_filter_fields': relevant_filter_fields,
        'dropped_for_plot_na': dropped_for_plot_na,
        'rows_after_filtering': len(df),
        'plot_fields': plot_fields,
    }
    return df.reset_index(drop=True), filter_report


def build_sankey_figure(df: pd.DataFrame, plot_fields: list[str], min_count: int = 1, max_categories: int = 20) -> go.Figure:
    working = df.copy()
    for field in plot_fields:
        counts = working[field].value_counts(dropna=False)
        allowed = set(counts.head(max_categories).index.astype(str))
        working[field] = working[field].astype(str).where(working[field].astype(str).isin(allowed), 'Other')

    node_labels: list[str] = []
    node_map: dict[str, int] = {}
    link_source: list[int] = []
    link_target: list[int] = []
    link_value: list[int] = []

    def get_node_id(stage: str, value: str) -> int:
        key = f'{stage}|{value}'
        if key not in node_map:
            node_map[key] = len(node_labels)
            node_labels.append(humanize_text(value))
        return node_map[key]

    for left, right in zip(plot_fields[:-1], plot_fields[1:]):
        grouped = working.groupby([left, right], dropna=False).size().reset_index(name='count')
        if min_count > 1:
            grouped = grouped.loc[grouped['count'] >= min_count].copy()
        for _, row in grouped.iterrows():
            left_value = normalise_category(row[left])
            right_value = normalise_category(row[right])
            s = get_node_id(left, left_value)
            t = get_node_id(right, right_value)
            link_source.append(s)
            link_target.append(t)
            link_value.append(int(row['count']))

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(pad=18, thickness=18, label=node_labels),
        link=dict(source=link_source, target=link_target, value=link_value),
    ))
    fig.update_layout(title='')
    return fig


def build_overview_summary(filtered_df: pd.DataFrame, plot_fields: list[str], filter_report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'row_counts': {'rows_used_for_plots': len(filtered_df)},
        'filter_report': filter_report,
        'plot_fields': plot_fields,
        'field_value_counts': {},
    }
    for field in plot_fields:
        if field in filtered_df.columns:
            counts = Counter(filtered_df[field].astype(str))
            summary['field_value_counts'][field] = dict(counts)
    return summary
