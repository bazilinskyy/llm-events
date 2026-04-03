from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.labels import humanize_field_name, humanize_text
from utils.plots import save_plotly_figure
from utils.research_plots import (
    create_accountability_by_taxonomy_figure,
    create_blame_alignment_figure,
    create_blind_spot_figure,
    create_completeness_figure,
    create_consistency_figure,
    create_context_gap_figure,
    create_determinability_figure,
    create_environment_profile_figure,
    create_intersection_detail_figure,
    create_provenance_availability_figure,
    create_stopped_av_subtype_figure,
    create_taxonomy_bar_figure,
    create_taxonomy_by_road_user_figure,
)
from utils.sankey import build_sankey_figure

logger = logging.getLogger(__name__)


def _safe_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    counts = df[field].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [field, 'count']
    counts[field] = counts[field].map(humanize_text)
    return counts


def create_histogram_figure(df: pd.DataFrame, field: str) -> go.Figure:
    counts = _safe_counts(df, field)
    fig = px.bar(counts, x=field, y='count', title='')
    fig.update_xaxes(title_text=humanize_field_name(field))
    fig.update_yaxes(title_text=humanize_field_name('count'))

    if field == 'road_user_type':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'av_mode_group':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'av_movement_group':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'other_party_movement_group':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'collision_group':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'blame_group':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'scenario_class':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'report_completeness_band':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'weather_v1':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'light_v1':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'surface_v1':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    elif field == 'condition_v1':
        fig.update_layout(font=dict(family='Times New Roman', size=16))
        fig.update_xaxes(
            title_font=dict(family='Times New Roman', size=24),
            tickfont=dict(family='Times New Roman', size=16),
            tickangle=0,
        )
        fig.update_yaxes(
            title_font=dict(family='Times New Roman', size=22),
            tickfont=dict(family='Times New Roman', size=14),
        )

    return fig


def create_sunburst_figure(df: pd.DataFrame, plot_fields: list[str]) -> go.Figure:
    working = df.copy()
    formatted_fields: list[str] = []
    for field in plot_fields:
        label = humanize_field_name(field)
        working[label] = working[field].astype(str).map(humanize_text)
        formatted_fields.append(label)
    fig = px.sunburst(working, path=formatted_fields, title='')
    return fig


def create_transition_graph_figure(df: pd.DataFrame, plot_fields: list[str]) -> go.Figure:
    graph = nx.DiGraph()
    for left, right in zip(plot_fields[:-1], plot_fields[1:]):
        grouped = df.groupby([left, right]).size().reset_index(name='count')
        for _, row in grouped.iterrows():
            source = f"{humanize_field_name(left)}: {humanize_text(row[left])}"
            target = f"{humanize_field_name(right)}: {humanize_text(row[right])}"
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
    paper_plot_top_n: int,
    blind_spot_fields: list[str],
    image_export_timeout_seconds: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {'plots': {}}

    if filtered_df.empty:
        logger.warning('Filtered dataframe is empty. Skipping plot creation.')
        return manifest

    export_kwargs = {
        'auto_open_html': auto_open_html,
        'save_final': save_final,
        'save_png': True,
        'save_eps': True,
        'export_timeout_seconds': image_export_timeout_seconds,
    }

    manifest['plots']['accident_overview_sankey'] = save_plotly_figure(
        build_sankey_figure(filtered_df, plot_fields, min_count=min_count, max_categories=max_categories),
        'accident_overview_sankey',
        output_dir=output_dirs.plots,
        final_dir=output_dirs.figures,
        **export_kwargs,
    )

    if len(plot_fields) >= 2:
        manifest['plots']['accident_overview_sunburst'] = save_plotly_figure(
            create_sunburst_figure(filtered_df, plot_fields),
            'accident_overview_sunburst',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            **export_kwargs,
        )

        manifest['plots']['accident_transition_graph'] = save_plotly_figure(
            create_transition_graph_figure(filtered_df, plot_fields),
            'accident_transition_graph',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            **export_kwargs,
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
            **export_kwargs,
        )
    manifest['plots']['histograms'] = histogram_manifest

    paper_manifest: dict[str, Any] = {}
    paper_manifest['taxonomy_overview'] = save_plotly_figure(
        create_taxonomy_bar_figure(filtered_df, top_n=paper_plot_top_n),
        'taxonomy_overview',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['blind_spots_missingness'] = save_plotly_figure(
        create_blind_spot_figure(parsed_df, fields=blind_spot_fields),
        'blind_spots_missingness',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['accountability_by_taxonomy'] = save_plotly_figure(
        create_accountability_by_taxonomy_figure(filtered_df, top_n=paper_plot_top_n),
        'accountability_by_taxonomy',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['report_completeness'] = save_plotly_figure(
        create_completeness_figure(parsed_df),
        'report_completeness',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['taxonomy_by_road_user'] = save_plotly_figure(
        create_taxonomy_by_road_user_figure(filtered_df, top_n=paper_plot_top_n),
        'taxonomy_by_road_user',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['provenance_availability'] = save_plotly_figure(
        create_provenance_availability_figure(parsed_df),
        'provenance_availability',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['context_gap'] = save_plotly_figure(
        create_context_gap_figure(parsed_df),
        'context_gap',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['movement_consistency'] = save_plotly_figure(
        create_consistency_figure(parsed_df),
        'movement_consistency',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['scenario_determinability'] = save_plotly_figure(
        create_determinability_figure(filtered_df),
        'scenario_determinability',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['environment_profile'] = save_plotly_figure(
        create_environment_profile_figure(filtered_df),
        'environment_profile',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['blame_confidence_alignment'] = save_plotly_figure(
        create_blame_alignment_figure(parsed_df),
        'blame_confidence_alignment',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['stopped_av_subtype'] = save_plotly_figure(
        create_stopped_av_subtype_figure(filtered_df),
        'stopped_av_subtype',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    paper_manifest['intersection_detail_quality'] = save_plotly_figure(
        create_intersection_detail_figure(parsed_df),
        'intersection_detail_quality',
        output_dir=output_dirs.paper,
        final_dir=output_dirs.figures_paper,
        **export_kwargs,
    )
    manifest['plots']['paper'] = paper_manifest
    return manifest
