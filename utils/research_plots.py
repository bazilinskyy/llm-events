from __future__ import annotations

"""Research figure helpers built on top of Plotly Express and Graph Objects.

This module contains small plotting utilities used to generate the paper style
figures for the analysis pipeline. The helpers standardise label formatting,
axis titles, and a few repeated aggregation patterns across figures.
"""

import common
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.labels import humanize_field_name, humanize_text
from utils.normalise import is_missing


def _format_series(series: pd.Series) -> pd.Series:
    """Formats a pandas series for display in figures.

    Args:
        series: Input series containing raw categorical values.

    Returns:
        A series with values converted to strings and humanised for display.
    """

    return series.astype(str).map(humanize_text)


def _top_counts(df: pd.DataFrame, field: str, top_n: int) -> pd.DataFrame:
    """Computes top category counts and shares for a field.

    Args:
        df: Source dataframe.
        field: Column name to aggregate.
        top_n: Maximum number of categories to keep.

    Returns:
        A dataframe containing the formatted category, raw count, and share.
    """

    counts = df[field].astype(str).value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = [field, 'count']
    counts['share'] = counts['count'] / max(len(df), 1)
    counts[field] = _format_series(counts[field])
    return counts


def _update_axis_labels(
    fig: go.Figure,
    *,
    x: str | None = None,
    y: str | None = None,
    legend: str | None = None,
) -> go.Figure:
    """Applies human readable axis and legend titles to a figure.

    Args:
        fig: Figure to update.
        x: Optional x axis field name.
        y: Optional y axis field name.
        legend: Optional legend field name.

    Returns:
        The updated figure.
    """

    if x is not None:
        fig.update_xaxes(title_text=humanize_field_name(x))
    if y is not None:
        fig.update_yaxes(title_text=humanize_field_name(y))
    if legend is not None:
        fig.update_layout(legend_title_text=humanize_field_name(legend))
    return fig


def _apply_uniform_horizontal_bar_density(
    fig: go.Figure,
    *,
    n_bars: int,
    bar_width: float = 0.55,
    bargap: float = 0.35,
    row_slot_px: int = 58,
    min_height: int = 650,
    margin_top: int = 0,
    margin_right: int = 0,
    margin_bottom: int = 0,
    margin_left: int = 0,
) -> go.Figure:
    """Applies a consistent visual bar density to horizontal bar charts.

    This keeps bar thickness and inter bar spacing visually aligned across
    figures even when the number of categories differs. The export helper can
    then reuse the figure's explicit height so static outputs match the HTML
    rendering.

    Args:
        fig: Figure to update.
        n_bars: Number of horizontal bars in the chart.
        bar_width: Relative bar thickness within each category slot.
        bargap: Gap between category slots.
        row_slot_px: Pixel height reserved per category slot.
        min_height: Minimum overall figure height.
        margin_top: Top margin in pixels.
        margin_right: Right margin in pixels.
        margin_bottom: Bottom margin in pixels.
        margin_left: Left margin in pixels.

    Returns:
        The updated figure.
    """

    bar_count = max(int(n_bars), 1)
    figure_height = max(
        int(min_height),
        int(margin_top + margin_bottom + (bar_count * row_slot_px)),
    )

    fig.update_traces(width=bar_width)
    fig.update_layout(
        bargap=bargap,
        height=figure_height,
        margin=dict(
            t=margin_top,
            r=margin_right,
            b=margin_bottom,
            l=margin_left,
        ),
    )
    return fig


def _to_list(values: object) -> list[object]:
    """Safely coerces scalar or array like values into a Python list."""

    if values is None:
        return []

    if hasattr(values, 'tolist'):
        try:
            return list(values.tolist())  # type: ignore
        except Exception:
            pass

    if isinstance(values, (list, tuple)):
        return list(values)

    try:
        return list(values)  # type: ignore
    except TypeError:
        return [values]


def _get_numeric_trace_max(fig: go.Figure, axis: str) -> float:
    """Returns the maximum numeric value found across bar traces on an axis."""

    max_value = 0.0
    for trace in fig.data:
        if getattr(trace, 'type', None) != 'bar':
            continue

        values = getattr(trace, axis, None)
        if values is None:
            continue

        numeric = pd.to_numeric(pd.Series(list(values)), errors='coerce').dropna()
        if not numeric.empty:
            max_value = max(max_value, float(numeric.max()))

    return max_value


def _add_bar_value_labels(
    fig: go.Figure,
    *,
    numeric_axis: str,
    value_format: str,
    textposition: str = 'outside',
    headroom_factor: float = 1.15,
) -> go.Figure:
    """Adds numeric labels to bar traces and expands the numeric axis if needed.

    Args:
        fig: Figure to annotate.
        numeric_axis: The numeric trace axis, either ``'x'`` or ``'y'``.
        value_format: Python format specifier such as ``'.0f'`` or ``'.0%'``.
        textposition: Plotly text position.
        headroom_factor: Axis expansion factor for outside labels.

    Returns:
        The updated figure.
    """

    if numeric_axis not in {'x', 'y'}:
        raise ValueError("numeric_axis must be either 'x' or 'y'.")

    for trace in fig.data:
        if getattr(trace, 'type', None) != 'bar':
            continue

        values = getattr(trace, numeric_axis, None)
        if values is None:
            continue

        labels: list[str] = []
        for value in values:
            numeric_value = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
            if pd.isna(numeric_value):
                labels.append('')
            else:
                labels.append(format(float(numeric_value), value_format))

        trace.text = labels  # type: ignore
        trace.texttemplate = '%{text}'  # type: ignore
        trace.textposition = textposition  # type: ignore
        trace.cliponaxis = False  # type: ignore

    if textposition == 'outside':
        max_value = _get_numeric_trace_max(fig, numeric_axis)
        if max_value > 0:
            upper_bound = max_value * headroom_factor
            if numeric_axis == 'x':
                fig.update_xaxes(range=[0, upper_bound], automargin=True)
            else:
                fig.update_yaxes(range=[0, upper_bound], automargin=True)

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=40, r=100, b=60, l=140),
    )
    return fig


def _add_stacked_bar_segment_labels(
    fig: go.Figure,
    *,
    min_inside_value: float = 10.0,
    headroom_factor: float = 1.18,
) -> go.Figure:
    """Adds readable labels to stacked bar charts.

    Large segments are labelled inside the bar. Small segments get callout
    annotations so their values remain visible even when the segment is too
    thin to hold text. A total is also shown above each stack.

    Args:
        fig: Figure to annotate.
        min_inside_value: Minimum segment height that can hold an inside label.
        headroom_factor: Axis expansion factor for the total labels.

    Returns:
        The updated figure.
    """

    totals: dict[str, float] = {}
    cumulative: dict[str, float] = {}
    small_label_count: dict[str, int] = {}

    for trace in fig.data:
        if getattr(trace, 'type', None) != 'bar':
            continue

        x_values = _to_list(getattr(trace, 'x', None))
        y_values = pd.to_numeric(pd.Series(_to_list(getattr(trace, 'y', None))), errors='coerce')

        labels: list[str] = []
        for x_value, raw_y in zip(x_values, y_values):
            if pd.isna(raw_y) or float(raw_y) <= 0:
                labels.append('')
                continue

            y_value = float(raw_y)
            category = str(x_value)
            base_value = cumulative.get(category, 0.0)

            totals[category] = totals.get(category, 0.0) + y_value

            if y_value >= min_inside_value:
                labels.append(f'{y_value:.0f}')
            else:
                labels.append('')
                callout_index = small_label_count.get(category, 0)
                xshift = 26 + (callout_index % 3) * 14
                yshift = (-16 if callout_index % 2 == 0 else 16) + (callout_index // 2) * 6
                fig.add_annotation(
                    x=x_value,
                    y=base_value + (y_value / 2.0),
                    text=f'{y_value:.0f}',
                    showarrow=True,
                    arrowhead=0,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=xshift,
                    ay=yshift,
                    bgcolor='rgba(255,255,255,0.92)',
                    bordercolor='rgba(90,90,90,0.55)',
                    borderpad=2,
                    font=dict(size=10),
                    align='center',
                )
                small_label_count[category] = callout_index + 1

            cumulative[category] = base_value + y_value

        trace.text = labels  # type: ignore
        trace.texttemplate = '%{text}'  # type: ignore
        trace.textposition = 'inside'  # type: ignore
        trace.insidetextanchor = 'middle'  # type: ignore
        trace.cliponaxis = False  # type: ignore

    if totals:
        max_total = max(totals.values())
        for category, total in totals.items():
            fig.add_annotation(
                x=category,
                y=total,
                text=f'{total:.0f}',
                showarrow=False,
                yshift=14,
                font=dict(size=11),
            )

        fig.update_yaxes(range=[0, max_total * headroom_factor], automargin=True)

    fig.update_xaxes(automargin=True)
    fig.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=50, r=170, b=80, l=80),
    )
    return fig


def create_taxonomy_bar_figure(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Creates a horizontal bar chart for the most common scenario classes.

    Args:
        df: Source dataframe.
        top_n: Maximum number of scenario classes to display.

    Returns:
        A Plotly figure.
    """

    def _sentence_case_preserve_abbreviations(value: str) -> str:
        text = humanize_text(value)
        if text == 'NA':
            return text

        words = text.split()
        if not words:
            return text

        formatted_words: list[str] = []
        for index, word in enumerate(words):
            if word.isupper():
                formatted_words.append(word)
            elif index == 0:
                formatted_words.append(word.capitalize())
            else:
                formatted_words.append(word.lower())

        return ' '.join(formatted_words)

    counts = _top_counts(df, 'scenario_class', top_n)
    counts['scenario_class'] = counts['scenario_class'].map(
        _sentence_case_preserve_abbreviations
    )

    fig = px.bar(
        counts,
        x='count',
        y='scenario_class',
        orientation='h',
        title='',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig = _apply_uniform_horizontal_bar_density(
        fig,
        n_bars=len(counts),
    )
    fig = _update_axis_labels(fig, x='count', y='scenario_class')
    return _add_bar_value_labels(fig, numeric_axis='x', value_format='.0f')


def create_blind_spot_figure(df: pd.DataFrame, fields: list[str]) -> go.Figure:
    """Creates a post extraction unavailability chart for selected fields.

    This figure deliberately describes what is unavailable after extraction.
    It does not assume that an unavailable value was absent from the source
    form or narrative.

    Args:
        df: Source dataframe.
        fields: Ordered list of fields to evaluate for missingness.

    Returns:
        A Plotly figure.
    """

    rows = []
    total = max(len(df), 1)

    for field in fields:
        if field in df.columns:
            missing = int(df[field].map(is_missing).sum())
        else:
            missing = total

        rows.append({
            'field': humanize_field_name(field),
            'missing_rate': missing / total,
        })

    frame = pd.DataFrame(rows).sort_values('missing_rate', ascending=True)
    fig = px.bar(frame, x='missing_rate', y='field', orientation='h', title='')
    fig = _apply_uniform_horizontal_bar_density(
        fig,
        n_bars=len(frame),
    )
    fig = _update_axis_labels(fig, x='missing_rate', y='field')
    return _add_bar_value_labels(fig, numeric_axis='x', value_format='.0%')


def create_accountability_by_taxonomy_figure(
    df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """Creates a grouped horizontal accountability chart by scenario class.

    This version avoids Plotly multicategory y axes because they can reorder
    bars within a section during rendering and static export. Instead, it uses
    explicit numeric row positions, scenario labels drawn as annotations inside
    a reserved left label area, and blame group labels drawn inside that same
    area. This guarantees two things for the exported figure:

    * bars stay sorted within each blame group by descending count
    * the left side labels stay inside the plot and do not get cut off

    Args:
        df: Source dataframe.
        top_n: Maximum number of scenario classes to display.

    Returns:
        A Plotly figure.
    """

    font_size = int(common.get_configs("font_size"))

    top_taxonomy = (
        df['scenario_class']
        .astype(str)
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )
    working = df.loc[df['scenario_class'].astype(str).isin(top_taxonomy)].copy()

    counts = (
        working.groupby(['blame_group', 'scenario_class'])
        .size()
        .reset_index(name='count')
    )
    if counts.empty:
        return go.Figure()

    counts['scenario_class'] = _format_series(counts['scenario_class'])
    counts['blame_group'] = _format_series(counts['blame_group'])

    blame_totals = counts.groupby('blame_group', as_index=False)['count'].sum()

    # Keep the overall section order visually similar to the existing figure:
    # smaller sections at the top and larger sections at the bottom.
    blame_order_top_to_bottom = (
        blame_totals.sort_values(['count', 'blame_group'],
                                 ascending=[True, True])  # type: ignore
        ['blame_group']
        .tolist()
    )

    palette = px.colors.qualitative.Plotly
    blame_color_map = {
        blame_group: palette[index % len(palette)]
        for index, blame_group in enumerate(blame_order_top_to_bottom)
    }

    plot_rows: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    separator_ys: list[float] = []

    y_cursor = 0.0
    group_gap = 1.0

    for blame_group in blame_order_top_to_bottom:
        group_rows = (
            counts.loc[counts['blame_group'] == blame_group]
            .sort_values(['count', 'scenario_class'], ascending=[False, True])
            .reset_index(drop=True)
        )
        if group_rows.empty:
            continue

        start_y = y_cursor
        for row in group_rows.itertuples(index=False):
            plot_rows.append({
                'y': y_cursor,
                'count': float(row.count),
                'scenario_class': str(row.scenario_class),
                'blame_group': str(blame_group),
                'bar_color': blame_color_map[str(blame_group)],
            })
            y_cursor += 1.0

        end_y = y_cursor - 1.0
        group_mid_y = (start_y + end_y) / 2.0
        annotations.append({
            'xref': 'x',
            'yref': 'y',
            'x': 0.0,  # placeholder, filled after label area width is known
            'y': group_mid_y,
            'text': f'<b>{blame_group}</b>',
            'showarrow': False,
            'xanchor': 'left',
            'yanchor': 'middle',
            'align': 'left',
            'font': {'size': font_size},
        })

        separator_ys.append(y_cursor - 0.5 + (group_gap / 2.0))
        y_cursor += group_gap

    if separator_ys:
        separator_ys = separator_ys[:-1]

    plot_frame = pd.DataFrame(plot_rows)
    if plot_frame.empty:
        return go.Figure()

    max_count = float(pd.to_numeric(plot_frame['count'], errors='coerce').max())
    if max_count <= 0:
        x_upper = 1.0
    elif max_count <= 300:
        x_upper = 300.0
    else:
        x_upper = float(((int(max_count) + 49) // 50) * 50)

    # Reserve a wider internal left label area inside the x axis range so long
    # labels are rendered within the plotting area instead of being clipped in
    # the export margin. Keep a larger gutter between the blame group label
    # column and the scenario label column so the two levels are clearly
    # separated in exports.
    label_zone_width = max(285.0, x_upper * 0.82)
    group_label_padding = max(20.0, font_size * 1.2)
    scenario_label_padding = max(8.0, font_size * 0.45)
    group_label_x = -label_zone_width + group_label_padding
    scenario_label_x = -scenario_label_padding
    label_column_separator_x = (group_label_x + scenario_label_x) / 2.0

    for annotation in annotations:
        annotation['x'] = group_label_x

    for row in plot_frame.itertuples(index=False):
        annotations.append({
            'xref': 'x',
            'yref': 'y',
            'x': scenario_label_x,
            'y': float(row.y),  # type: ignore
            'text': str(row.scenario_class),
            'showarrow': False,
            'xanchor': 'right',
            'yanchor': 'middle',
            'align': 'right',
            'font': {'size': font_size},
        })

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_frame['count'].tolist(),
            y=plot_frame['y'].tolist(),
            orientation='h',
            marker=dict(color=plot_frame['bar_color'].tolist()),
            text=[f'{value:.0f}' for value in plot_frame['count'].tolist()],
            textposition='outside',
            cliponaxis=False,
            hovertemplate=(
                'Blame group=%{customdata[0]}<br>'
                'Scenario class=%{customdata[1]}<br>'
                'Count=%{x:.0f}<extra></extra>'
            ),
            customdata=plot_frame[['blame_group', 'scenario_class']].to_numpy(),
        )
    )

    tick_step = 50 if x_upper >= 50 else 10
    x_tick_values = list(range(0, int(x_upper) + 1, tick_step))
    if not x_tick_values or x_tick_values[-1] != int(x_upper):
        x_tick_values.append(int(x_upper))

    shapes: list[dict[str, object]] = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': 0,
            'x1': 0,
            'y0': 0,
            'y1': 1,
            'line': {'color': 'rgba(140,140,140,0.55)', 'width': 1},
        },
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': label_column_separator_x,
            'x1': label_column_separator_x,
            'y0': 0,
            'y1': 1,
            'line': {
                'color': 'rgba(140,140,140,0.20)',
                'width': 1,
                'dash': 'dot',
            },
        },
    ]
    for separator_y in separator_ys:
        shapes.append({
            'type': 'line',
            'xref': 'x',
            'yref': 'y',
            'x0': -label_zone_width,
            'x1': x_upper,
            'y0': separator_y,
            'y1': separator_y,
            'line': {'color': 'rgba(140,140,140,0.35)', 'width': 1},
        })

    fig.update_layout(
        title='',
        showlegend=False,
        bargap=0.28,
        annotations=annotations,
        shapes=shapes,
        font=dict(size=font_size),
        uniformtext_minsize=font_size,
        uniformtext_mode='hide',
        # The labels live inside the plot area, so only a modest margin is
        # needed and the exported figure no longer clips the left side.
        margin=dict(t=0, r=0, b=0, l=0),
    )
    fig.update_xaxes(
        title_text='Count',
        range=[-label_zone_width, x_upper],
        tickmode='array',
        tickvals=x_tick_values,
        ticktext=[str(value) for value in x_tick_values],
        showgrid=True,
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        title_text='',
        showticklabels=False,
        range=[y_cursor - 0.5, -0.5],
        automargin=False,
    )
    return fig


def create_completeness_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a histogram of report completeness scores.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    series = pd.to_numeric(df['report_completeness_score'], errors='coerce').dropna()
    if series.empty:
        return go.Figure()

    if float(series.min()) == float(series.max()):
        bin_edges = np.array([float(series.min()) - 0.5, float(series.max()) + 0.5])
    else:
        bin_edges = np.histogram_bin_edges(series.to_numpy(), bins=20)

    counts, edges = np.histogram(series.to_numpy(), bins=bin_edges)
    frame = pd.DataFrame({
        'bin_left': edges[:-1],
        'bin_right': edges[1:],
        'count': counts,
    })
    frame['bin_center'] = (frame['bin_left'] + frame['bin_right']) / 2
    frame['bin_width'] = frame['bin_right'] - frame['bin_left']

    fig = px.bar(frame, x='bin_center', y='count', title='')
    fig.update_traces(
        width=frame['bin_width'],
        hovertemplate=(
            'Range=%{customdata[0]:.3f} to %{customdata[1]:.3f}<br>'
            'Count=%{y:.0f}<extra></extra>'
        ),
        customdata=frame[['bin_left', 'bin_right']].to_numpy(),
    )
    fig = _update_axis_labels(fig, x='report_completeness_score', y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_taxonomy_by_road_user_figure(
    df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """Creates a heatmap of road user type by scenario class.

    Args:
        df: Source dataframe.
        top_n: Maximum number of scenario classes to include.

    Returns:
        A Plotly figure. Returns an empty figure when no data is available.
    """

    top_taxonomy = df['scenario_class'].astype(str).value_counts().head(top_n).index.tolist()
    pivot = (
        df.loc[df['scenario_class'].astype(str).isin(top_taxonomy)]
        .groupby(['road_user_type', 'scenario_class'])
        .size()
        .reset_index(name='count')
        .pivot(index='road_user_type', columns='scenario_class', values='count')
        .fillna(0)
    )

    if pivot.empty:
        return go.Figure()

    pivot.index = [humanize_text(value) for value in pivot.index]
    pivot.columns = [humanize_text(value) for value in pivot.columns]

    fig = px.imshow(pivot, aspect='auto', title='')
    fig.update_traces(text=pivot.to_numpy(), texttemplate='%{text:.0f}')
    return _update_axis_labels(fig, x='scenario_class', y='road_user_type')


def create_provenance_availability_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a provenance availability rate bar chart.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    rows = []
    for field in [
        'form_field_rate',
        'checkbox_field_rate',
        'narrative_field_rate',
        'online_field_rate',
    ]:
        if field in df.columns:
            rows.append({
                'provenance': humanize_text(field.replace('_field_rate', '')),
                'availability_rate': float(
                    pd.to_numeric(df[field], errors='coerce').mean()
                ),
            })

    summary = pd.DataFrame(rows).sort_values('availability_rate', ascending=True)
    fig = px.bar(summary, x='availability_rate', y='provenance', orientation='h', title='')
    fig = _update_axis_labels(fig, x='availability_rate', y='provenance')
    return _add_bar_value_labels(fig, numeric_axis='x', value_format='.0%')


def create_context_gap_figure(df: pd.DataFrame) -> go.Figure:
    """Compares coarse, fine report, and external context availability.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    frame = pd.DataFrame([
        {
            'context_type': humanize_field_name('coarse_context_score'),
            'mean_score': float(
                pd.to_numeric(df['coarse_context_score'], errors='coerce').mean()
            ),
        },
        {
            'context_type': humanize_field_name('fine_context_score'),
            'mean_score': float(
                pd.to_numeric(df['fine_context_score'], errors='coerce').mean()
            ),
        },
        {
            'context_type': humanize_field_name('external_context_score'),
            'mean_score': float(
                pd.to_numeric(
                    df['external_context_score'],
                    errors='coerce',
                ).mean()
            ),
        },
    ])

    fig = px.bar(frame, x='context_type', y='mean_score', title='')
    fig = _update_axis_labels(fig, x='context_type', y='mean_score')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.2f')


def create_consistency_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for cross field movement agreement.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    field = (
        'movement_field_agreement'
        if 'movement_field_agreement' in df.columns
        else 'movement_consistency_overall'
    )
    counts = df[field].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [field, 'count']
    counts[field] = _format_series(counts[field])

    fig = px.bar(counts, x=field, y='count', title='')
    fig = _update_axis_labels(fig, x=field, y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_determinability_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for internal scenario rule support groups.

    Rule support measures how strongly the available extracted fields support
    the deterministic assignment. It is not a validated accuracy estimate.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    field = (
        'scenario_rule_support_group'
        if 'scenario_rule_support_group' in df.columns
        else 'scenario_determinability_group'
    )
    counts = df[field].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [field, 'count']
    counts[field] = _format_series(counts[field])

    fig = px.bar(counts, x=field, y='count', title='')
    fig = _update_axis_labels(fig, x=field, y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_environment_profile_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for environment friction profiles.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    counts = df['environment_friction_profile'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['environment_friction_profile', 'count']
    counts['environment_friction_profile'] = _format_series(
        counts['environment_friction_profile']
    )

    fig = px.bar(counts, x='environment_friction_profile', y='count', title='')
    fig = _update_axis_labels(fig, x='environment_friction_profile', y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_blame_alignment_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for blame confidence alignment.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    counts = df['blame_confidence_alignment'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['blame_confidence_alignment', 'count']
    counts['blame_confidence_alignment'] = _format_series(
        counts['blame_confidence_alignment']
    )

    fig = px.bar(counts, x='blame_confidence_alignment', y='count', title='')
    fig = _update_axis_labels(fig, x='blame_confidence_alignment', y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_stopped_av_subtype_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for non missing stopped AV subtypes.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure. Returns an empty figure when no rows are available.
    """

    working = df.loc[df['stopped_av_subtype'].astype(str) != 'NA'].copy()
    if working.empty:
        return go.Figure()

    counts = working['stopped_av_subtype'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['stopped_av_subtype', 'count']
    counts['stopped_av_subtype'] = _format_series(counts['stopped_av_subtype'])

    fig = px.bar(counts, x='stopped_av_subtype', y='count', title='')
    fig = _update_axis_labels(fig, x='stopped_av_subtype', y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')


def create_intersection_detail_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for intersection detail quality.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    counts = df['intersection_detail_quality'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['intersection_detail_quality', 'count']
    counts['intersection_detail_quality'] = _format_series(
        counts['intersection_detail_quality']
    )

    fig = px.bar(counts, x='intersection_detail_quality', y='count', title='')
    fig = _update_axis_labels(fig, x='intersection_detail_quality', y='count')
    return _add_bar_value_labels(fig, numeric_axis='y', value_format='.0f')
