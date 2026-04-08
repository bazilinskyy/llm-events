from __future__ import annotations

"""Research figure helpers built on top of Plotly Express and Graph Objects.

This module contains small plotting utilities used to generate the paper style
figures for the analysis pipeline. The helpers standardise label formatting,
axis titles, and a few repeated aggregation patterns across figures.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.labels import humanize_field_name, humanize_text


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
    return _update_axis_labels(fig, x='count', y='scenario_class')


def create_blind_spot_figure(df: pd.DataFrame, fields: list[str]) -> go.Figure:
    """Creates a missing rate chart for the requested blind spot fields.

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
            missing = int(
                df[field].astype(str).str.lower().isin(
                    {'na', 'none', 'nan', 'null', ''}
                ).sum()
            )
        else:
            missing = total

        rows.append({
            'field': humanize_field_name(field),
            'missing_rate': missing / total,
        })

    frame = pd.DataFrame(rows).sort_values('missing_rate', ascending=True)
    fig = px.bar(frame, x='missing_rate', y='field', orientation='h', title='')
    return _update_axis_labels(fig, x='missing_rate', y='field')


def create_accountability_by_taxonomy_figure(
    df: pd.DataFrame,
    top_n: int = 8,
) -> go.Figure:
    """Creates a stacked accountability chart by scenario class.

    Args:
        df: Source dataframe.
        top_n: Maximum number of scenario classes to display.

    Returns:
        A Plotly figure.
    """

    top_taxonomy = df['scenario_class'].astype(str).value_counts().head(top_n).index.tolist()
    working = df.loc[df['scenario_class'].astype(str).isin(top_taxonomy)].copy()

    counts = (
        working.groupby(['scenario_class', 'blame_group'])
        .size()
        .reset_index(name='count')
    )
    counts['scenario_class'] = _format_series(counts['scenario_class'])
    counts['blame_group'] = _format_series(counts['blame_group'])

    fig = px.bar(
        counts,
        x='scenario_class',
        y='count',
        color='blame_group',
        barmode='stack',
        title='',
    )
    fig = _update_axis_labels(fig, x='scenario_class', y='count', legend='')

    # The legend title is intentionally hidden for this plot.
    fig.update_layout(legend_title_text='')
    return fig


def create_completeness_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a histogram of report completeness scores.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    fig = px.histogram(df, x='report_completeness_score', nbins=20, title='')
    return _update_axis_labels(fig, x='report_completeness_score', y='count')


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
    return _update_axis_labels(fig, x='availability_rate', y='provenance')


def create_context_gap_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a mean score comparison across context related metrics.

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
            'context_type': humanize_field_name('report_explicitness_score'),
            'mean_score': float(
                pd.to_numeric(
                    df['report_explicitness_score'],
                    errors='coerce',
                ).mean()
            ),
        },
    ])

    fig = px.bar(frame, x='context_type', y='mean_score', title='')
    return _update_axis_labels(fig, x='context_type', y='mean_score')


def create_consistency_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for movement consistency status.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    counts = df['movement_consistency_overall'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['movement_consistency_overall', 'count']
    counts['movement_consistency_overall'] = _format_series(
        counts['movement_consistency_overall']
    )

    fig = px.bar(counts, x='movement_consistency_overall', y='count', title='')
    return _update_axis_labels(fig, x='movement_consistency_overall', y='count')


def create_determinability_figure(df: pd.DataFrame) -> go.Figure:
    """Creates a count chart for scenario determinability groups.

    Args:
        df: Source dataframe.

    Returns:
        A Plotly figure.
    """

    counts = df['scenario_determinability_group'].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['scenario_determinability_group', 'count']
    counts['scenario_determinability_group'] = _format_series(
        counts['scenario_determinability_group']
    )

    fig = px.bar(counts, x='scenario_determinability_group', y='count', title='')
    return _update_axis_labels(fig, x='scenario_determinability_group', y='count')


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
    return _update_axis_labels(fig, x='environment_friction_profile', y='count')


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
    return _update_axis_labels(fig, x='blame_confidence_alignment', y='count')


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
    return _update_axis_labels(fig, x='stopped_av_subtype', y='count')


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
    return _update_axis_labels(fig, x='intersection_detail_quality', y='count')
