from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.plots import save_plotly_figure
from utils.sankey import build_sankey_figure

logger = logging.getLogger(__name__)


def _safe_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    counts = df[field].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [field, 'count']
    return counts


def create_histogram_figure(df: pd.DataFrame, field: str) -> go.Figure:
    counts = _safe_counts(df, field)
    fig = px.bar(counts, x=field, y='count', title='')
    fig.update_layout(xaxis_title=field, yaxis_title='Count')
    return fig


def create_sunburst_figure(df: pd.DataFrame, plot_fields: list[str]) -> go.Figure:
    fig = px.sunburst(df, path=plot_fields, title='')
    return fig


def create_transition_graph_figure(df: pd.DataFrame, plot_fields: list[str]) -> go.Figure:
    graph = nx.DiGraph()
    for left, right in zip(plot_fields[:-1], plot_fields[1:]):
        grouped = df.groupby([left, right]).size().reset_index(name='count')
        for _, row in grouped.iterrows():
            source = f'{left}: {row[left]}'
            target = f'{right}: {row[right]}'
            graph.add_edge(source, target, weight=int(row['count']))

    if not graph.nodes:
        return go.Figure()

    positions = nx.spring_layout(graph, seed=42)
    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1), hoverinfo='none', mode='lines')
    node_x = [positions[node][0] for node in graph.nodes()]
    node_y = [positions[node][1] for node in graph.nodes()]
    node_text = list(graph.nodes())
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(size=14),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='', showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def create_all_plots(
    parsed_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    plot_fields: list[str],
    output_dirs,
    auto_open_html: bool,
    histogram_fields: list[str],
    min_count: int,
    max_categories: int,
    save_final: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {'plots': {}}

    if filtered_df.empty:
        logger.warning('Filtered dataframe is empty. Skipping plot creation.')
        return manifest

    manifest['plots']['accident_overview_sankey'] = save_plotly_figure(
        build_sankey_figure(filtered_df, plot_fields, min_count=min_count, max_categories=max_categories),
        'accident_overview_sankey',
        output_dir=output_dirs.plots,
        final_dir=output_dirs.figures,
        auto_open_html=auto_open_html,
        save_final=save_final,
        save_png=True,
        save_eps=True,
    )

    if len(plot_fields) >= 2:
        manifest['plots']['accident_overview_sunburst'] = save_plotly_figure(
            create_sunburst_figure(filtered_df, plot_fields),
            'accident_overview_sunburst',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            auto_open_html=auto_open_html,
            save_final=save_final,
            save_png=True,
            save_eps=True,
        )

        manifest['plots']['accident_transition_graph'] = save_plotly_figure(
            create_transition_graph_figure(filtered_df, plot_fields),
            'accident_transition_graph',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            auto_open_html=auto_open_html,
            save_final=save_final,
            save_png=True,
            save_eps=True,
        )

    histogram_manifest: dict[str, Any] = {}
    for field in histogram_fields:
        if field not in parsed_df.columns:
            continue
        histogram_manifest[field] = save_plotly_figure(
            create_histogram_figure(parsed_df, field),
            field,
            output_dir=output_dirs.histograms,
            final_dir=output_dirs.figures_histograms,
            auto_open_html=auto_open_html,
            save_final=save_final,
            save_png=True,
            save_eps=True,
        )
    manifest['plots']['histograms'] = histogram_manifest
    return manifest
