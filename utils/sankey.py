from __future__ import annotations

"""Helpers for selecting plot fields, filtering rows, and building Sankey plots.

This module contains lightweight utilities used by the plotting pipeline to:

* validate and resolve plot fields
* filter rows with missing values in critical columns
* construct Sankey diagram inputs from categorical fields
* summarise filtered data for reporting
"""

import logging
from collections import Counter
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from utils.labels import humanize_field_name, humanize_text
from utils.normalise import is_missing, normalise_category

logger = logging.getLogger(__name__)


def resolve_plot_fields(
    df: pd.DataFrame,
    include_plot_fields: list[str],
    exclude_plot_fields: list[str],
) -> list[str]:
    """Resolves the ordered set of plot fields available in the dataframe.

    Args:
        df: Source dataframe.
        include_plot_fields: Candidate fields to include, in priority order.
        exclude_plot_fields: Fields to remove from the included set.

    Returns:
        The final ordered list of plot fields present in ``df``.

    Raises:
        ValueError: If fewer than two plot fields remain after filtering.
    """

    fields = [field for field in include_plot_fields if field in df.columns]

    if exclude_plot_fields:
        excluded = set(exclude_plot_fields)
        fields = [field for field in fields if field not in excluded]

    if len(fields) < 2:
        raise ValueError(
            'Need at least two plot fields after applying include and exclude settings.'
        )

    return fields


def _build_non_missing_mask(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Builds a boolean mask for rows with non missing values in all fields.

    Args:
        df: Source dataframe.
        fields: Fields that must all be non missing.

    Returns:
        A boolean series aligned to ``df.index``.
    """

    if not fields:
        return pd.Series(True, index=df.index)

    non_missing_by_field = {
        field: df[field].map(lambda value: not is_missing(value))
        for field in fields
    }
    mask_df = pd.DataFrame(non_missing_by_field, index=df.index)
    return mask_df.all(axis=1)


def apply_plot_filters(
    parsed_df: pd.DataFrame,
    plot_fields: list[str],
    filter_rows_with_na: bool,
    na_filter_fields: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filters rows used for plotting and returns a filter report.

    Args:
        parsed_df: Parsed dataframe before plot specific filtering.
        plot_fields: Plot fields that should be normalised for display.
        filter_rows_with_na: Whether to remove rows missing critical fields.
        na_filter_fields: Candidate fields used for missing value filtering.

    Returns:
        A tuple containing:
            * the filtered dataframe
            * a dictionary describing the filtering outcome
    """

    df = parsed_df.copy()
    dropped_empty_output = int(
        getattr(parsed_df, 'attrs', {}).get('dropped_empty_output', 0)
    )
    relevant_filter_fields = [
        field for field in na_filter_fields if field in df.columns
    ]

    dropped_for_plot_na = 0
    if filter_rows_with_na and relevant_filter_fields:
        mask = _build_non_missing_mask(df, relevant_filter_fields)
        dropped_for_plot_na = int((~mask).sum())
        df = df.loc[mask].copy()
        logger.info(
            'Dropped %s rows due to missing values in critical fields: %s',
            dropped_for_plot_na,
            ', '.join(relevant_filter_fields),
        )

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


def _build_stage_annotations(plot_fields: list[str]) -> list[dict[str, Any]]:
    """Builds top of plot annotations for Sankey stage labels.

    The first and last stage headers are slightly inset from the plot edges so
    they are not clipped in exported images.
    """

    side_padding = 0.03

    if len(plot_fields) < 2:
        x_positions = [0.5]
    else:
        usable_width = 1.0 - (2 * side_padding)
        x_positions = [
            side_padding + (usable_width * i / (len(plot_fields) - 1))
            for i in range(len(plot_fields))
        ]

    annotations: list[dict[str, Any]] = []
    for index, (x, field) in enumerate(zip(x_positions, plot_fields)):
        if index == 0:
            xanchor = 'left'
        elif index == len(plot_fields) - 1:
            xanchor = 'right'
        else:
            xanchor = 'center'

        annotations.append(
            dict(
                x=x,
                y=1.03,
                xref='paper',
                yref='paper',
                text=f'<b>{humanize_field_name(field)}</b>',
                showarrow=False,
                xanchor=xanchor,
                yanchor='bottom',
                font=dict(size=14),
            )
        )

    return annotations


def build_sankey_figure(
    df: pd.DataFrame,
    plot_fields: list[str],
    min_count: int = 1,
    max_categories: int = 20,
    show_stage_headers: bool = False,
) -> go.Figure:
    """Builds a Sankey figure across the ordered plot fields.

    Categories outside the top ``max_categories`` for each field are grouped
    into ``Other`` before links are constructed.

    Args:
        df: Source dataframe.
        plot_fields: Ordered categorical fields representing Sankey stages.
        min_count: Minimum edge count required to keep a link.
        max_categories: Maximum number of categories to keep per field.
        show_stage_headers: Whether to label each stage across the top.

    Returns:
        A Plotly Sankey figure.
    """

    working = df.copy()

    for field in plot_fields:
        counts = working[field].value_counts(dropna=False)
        allowed = set(counts.head(max_categories).index.astype(str))
        working[field] = working[field].astype(str).where(
            working[field].astype(str).isin(allowed),
            'Other',
        )

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
        grouped = (
            working.groupby([left, right], dropna=False)
            .size()
            .reset_index(name='count')
        )

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

    sankey_domain = dict(x=[0.02, 0.98], y=[0.05, 0.98])
    if show_stage_headers:
        sankey_domain = dict(x=[0.02, 0.98], y=[0.06, 0.95])

    fig = go.Figure(
        go.Sankey(
            arrangement='snap',
            domain=sankey_domain,
            node=dict(pad=16, thickness=18, label=node_labels),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
            ),
        )
    )

    layout_kwargs: dict[str, Any] = {
        'title': '',
        'margin': dict(l=0, r=0, b=0, t=0),
    }
    if show_stage_headers:
        layout_kwargs['annotations'] = _build_stage_annotations(plot_fields)
        layout_kwargs['margin'] = dict(l=0, r=0, b=0, t=0)

    fig.update_layout(**layout_kwargs)
    return fig


def build_overview_summary(
    filtered_df: pd.DataFrame,
    plot_fields: list[str],
    filter_report: dict[str, Any],
) -> dict[str, Any]:
    """Builds a compact summary of the filtered plotting dataset.

    Args:
        filtered_df: Dataframe after plot related filtering.
        plot_fields: Ordered plot fields included in the analysis.
        filter_report: Metadata returned by ``apply_plot_filters``.

    Returns:
        A summary dictionary containing row counts, filter metadata, and per
        field value counts.
    """

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
