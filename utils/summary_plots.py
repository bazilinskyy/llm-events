from __future__ import annotations

"""High level plotting orchestration for analysis and paper figures.

This module builds the full set of figures used by the pipeline, including:

* overview Sankey, sunburst, and transition graph figures
* per field histogram figures
* paper style research figures
* logging helpers that record the data underlying each figure
* export orchestration through the shared Plotly save helper
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.labels import humanize_field_name, humanize_text
from utils.normalise import is_missing
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


_LOCATION_MISSING_TOKENS = {
    '',
    'na',
    'n/a',
    'nan',
    'none',
    'null',
    'unknown',
    'not specified',
}


def _clean_location_component(value: Any) -> str:
    """Returns a cleaned location component or an empty string."""

    if is_missing(value):
        return ''

    text = str(value).strip(' ,;')
    if text.lower() in _LOCATION_MISSING_TOKENS:
        return ''
    return text


def _dedupe_location_parts(parts: list[str]) -> list[str]:
    """Deduplicates location parts while preserving order."""

    deduped: list[str] = []
    seen: set[str] = set()

    for part in parts:
        cleaned = _clean_location_component(part)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        deduped.append(cleaned)
        seen.add(key)

    return deduped


def _format_location_label(row: pd.Series) -> str:
    """Builds a readable location label from questionnaire fields."""

    county = _clean_location_component(row.get('county'))
    if county and 'county' not in county.casefold():
        county = f'{county} County'

    parts = _dedupe_location_parts([
        row.get('address'),
        row.get('city'),
        county,
        row.get('zipcode'),
    ])
    return ', '.join(parts)


def _build_location_candidates(row: pd.Series) -> list[str]:
    """Builds ordered geocoding candidates from the parsed questionnaire data."""

    county = _clean_location_component(row.get('county'))
    if county and 'county' not in county.casefold():
        county = f'{county} County'

    address = _clean_location_component(row.get('address'))
    city = _clean_location_component(row.get('city'))
    zipcode = _clean_location_component(row.get('zipcode'))

    candidates = [
        ', '.join(_dedupe_location_parts([address, city, county, zipcode, 'USA'])),
        ', '.join(_dedupe_location_parts([city, county, zipcode, 'USA'])),
        ', '.join(_dedupe_location_parts([zipcode, 'USA'])),
    ]

    ordered_unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue

        key = candidate.casefold()
        if key in seen:
            continue

        ordered_unique.append(candidate)
        seen.add(key)

    return ordered_unique


def _load_geocode_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Loads a simple JSON geocode cache from disk when available."""

    if not cache_path.exists():
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    except Exception:
        logger.exception('Failed to load geocode cache from %s', cache_path)
        return {}

    return payload if isinstance(payload, dict) else {}


def _save_geocode_cache(cache: dict[str, dict[str, Any]], cache_path: Path) -> None:
    """Persists the geocode cache to disk."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def _query_nominatim(query: str) -> dict[str, Any] | None:
    """Geocodes a free text location with the public Nominatim API."""

    url = (
        'https://nominatim.openstreetmap.org/search?'
        + urllib.parse.urlencode({
            'q': query,
            'format': 'jsonv2',
            'limit': 1,
        })
    )
    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'llm-events-accident-map/1.0',
            'Accept-Language': 'en',
        },
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode('utf-8'))

    if not payload:
        return None

    match = payload[0]
    return {
        'status': 'ok',
        'lat': float(match['lat']),
        'lon': float(match['lon']),
        'display_name': str(match.get('display_name', query)),
    }


def _geocode_candidate(
    query: str,
    *,
    cache: dict[str, dict[str, Any]],
    throttle_state: dict[str, float],
) -> dict[str, Any] | None:
    """Returns a cached or freshly geocoded location candidate."""

    cached = cache.get(query)
    if isinstance(cached, dict):
        return cached

    elapsed = time.monotonic() - throttle_state.get('last_request_at', 0.0)
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    try:
        result = _query_nominatim(query)
    except Exception as exc:
        logger.warning('Location geocoding failed for %s: %s', query, exc)
        result = {
            'status': 'error',
            'error': str(exc),
        }
    finally:
        throttle_state['last_request_at'] = time.monotonic()

    if result is None:
        result = {'status': 'not_found'}

    cache[query] = result
    return result


def _build_location_points_dataframe(
    df: pd.DataFrame,
    *,
    cache_path: Path,
) -> pd.DataFrame:
    """Builds a geocoded dataframe used by the accident location map."""

    cache = _load_geocode_cache(cache_path)
    throttle_state = {'last_request_at': 0.0}
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        location_label = _format_location_label(row)
        candidates = _build_location_candidates(row)
        if not location_label or not candidates:
            continue

        resolved: dict[str, Any] | None = None
        resolved_query = ''
        for candidate in candidates:
            geocoded = _geocode_candidate(
                candidate,
                cache=cache,
                throttle_state=throttle_state,
            )
            if geocoded and geocoded.get('status') == 'ok':
                resolved = geocoded
                resolved_query = candidate
                break

        if resolved is None:
            continue

        rows.append({
            'row_id': row.get('row_id', 'NA'),
            'report_pdf': row.get('report_pdf', 'NA'),
            'source_report': row.get('source_report', 'NA'),
            'address': row.get('address', 'NA'),
            'city': row.get('city', 'NA'),
            'county': row.get('county', 'NA'),
            'zipcode': row.get('zipcode', 'NA'),
            'location_label': location_label,
            'location_query': resolved_query,
            'resolved_location': resolved.get('display_name', 'NA'),
            'latitude': float(resolved['lat']),
            'longitude': float(resolved['lon']),
        })

    _save_geocode_cache(cache, cache_path)
    return pd.DataFrame(rows)


def create_accident_location_map_figure(
    df: pd.DataFrame,
    *,
    cache_path: Path,
) -> tuple[go.Figure, pd.DataFrame]:
    """Creates a Plotly accident point map from questionnaire location fields."""

    location_df = _build_location_points_dataframe(df, cache_path=cache_path)
    if location_df.empty:
        return go.Figure(), location_df

    hover_data = location_df[
        ['row_id', 'source_report', 'resolved_location', 'location_query']
    ].fillna('NA')

    fig = go.Figure(
        data=[
            go.Scattergeo(
                lat=location_df['latitude'],
                lon=location_df['longitude'],
                mode='markers',
                text=location_df['location_label'].fillna('Unknown location'),
                customdata=hover_data.to_numpy(),
                marker=dict(
                    size=9,
                    color='red',
                    opacity=0.85,
                    line=dict(width=0.8, color='white'),
                ),
                hovertemplate=(
                    '<b>%{text}</b><br><br>'
                    'Row ID: %{customdata[0]}<br>'
                    'Source report: %{customdata[1]}<br>'
                    'Resolved location: %{customdata[2]}<br>'
                    'Query used: %{customdata[3]}<extra></extra>'
                ),
            )
        ]
    )
    fig.update_geos(
        fitbounds='locations',
        visible=True,
        scope='usa',
        resolution=50,
        showland=True,
        landcolor='#E7D8B1',
        showocean=True,
        oceancolor='#9EC9FF',
        showlakes=True,
        lakecolor='#9EC9FF',
        showcountries=True,
        countrycolor='#777777',
        countrywidth=0.8,
        showsubunits=True,
        subunitcolor='#888888',
        subunitwidth=0.8,
        showcoastlines=True,
        coastlinecolor='#555555',
        coastlinewidth=1.2,
        bgcolor='#9EC9FF',
        showframe=False,
    )
    fig.update_layout(
        title='',
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig, location_df



def _safe_counts(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Builds a count table for a categorical field.

    Args:
        df: Source dataframe.
        field: Field name to aggregate.

    Returns:
        A dataframe containing formatted category labels and counts.
    """

    counts = df[field].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [field, 'count']
    counts[field] = counts[field].map(humanize_text)
    return counts


def _to_list(values: Any) -> list[Any]:
    """Coerces arbitrary values into a list for logging and inspection.

    Args:
        values: Input value that may already be list like or scalar.

    Returns:
        A list representation of the input value.
    """

    if values is None:
        return []

    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())
        except Exception:
            pass

    if isinstance(values, (list, tuple)):
        return list(values)

    try:
        return list(values)
    except TypeError:
        return [values]


def _uses_brace_formatting_logger() -> bool:
    """Returns whether the active logger uses brace style formatting.

    The project sometimes routes logs through ``CustomLogger``, which applies
    ``str.format`` to message text. Plotly trace strings contain many literal
    braces, so those must be escaped before logging.
    """

    return logger.__class__.__name__ == 'CustomLogger'


def _escape_braces(text: str) -> str:
    """Escapes literal braces for brace style loggers."""

    return text.replace('{', '{{').replace('}', '}}')


def _log_info(message: str) -> None:
    """Logs a message safely for both standard and brace style loggers."""

    text = _escape_braces(message) if _uses_brace_formatting_logger() else message
    logger.info(text)


def _log_dataframe(title: str, df: pd.DataFrame) -> None:
    """Logs a dataframe in a readable table form.

    Args:
        title: Log heading for the dataframe.
        df: Dataframe to log.
    """

    _log_info(f"{title}")
    _log_info(f"rows={len(df)} columns={list(df.columns)}")
    if df.empty:
        _log_info('table=<empty>')
        return

    for line in df.to_string(index=False).splitlines():
        _log_info(f"{line}")


def _trace_to_dataframe(trace: Any) -> pd.DataFrame:
    """Converts a Plotly trace into a dataframe for logging.

    Args:
        trace: Plotly trace object.

    Returns:
        A dataframe representation of the trace contents. Falls back to a
        simple string representation when no structured extraction is
        available.
    """

    trace_type = getattr(trace, 'type', 'unknown')

    if trace_type == 'bar':
        x = _to_list(getattr(trace, 'x', None))
        y = _to_list(getattr(trace, 'y', None))
        if x or y:
            return pd.DataFrame({'x': x, 'y': y})

    if trace_type == 'scatter':
        x = _to_list(getattr(trace, 'x', None))
        y = _to_list(getattr(trace, 'y', None))
        text = _to_list(getattr(trace, 'text', None))
        data: dict[str, Any] = {}
        if x:
            data['x'] = x
        if y:
            data['y'] = y
        if text and len(text) == max(len(x), len(y), len(text)):
            data['text'] = text
        if data:
            return pd.DataFrame(data)

    if trace_type == 'scattergeo':
        lat = _to_list(getattr(trace, 'lat', None))
        lon = _to_list(getattr(trace, 'lon', None))
        text = _to_list(getattr(trace, 'text', None))
        data: dict[str, Any] = {}
        if lat:
            data['lat'] = lat
        if lon:
            data['lon'] = lon
        if text and len(text) == max(len(lat), len(lon), len(text)):
            data['text'] = text
        if data:
            return pd.DataFrame(data)

    if trace_type == 'histogram':
        x = _to_list(getattr(trace, 'x', None))
        if x:
            series = pd.Series(x, name='x')
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().all():
                return pd.DataFrame([
                    {
                        'count': int(len(series)),
                        'min': float(numeric.min()),
                        'max': float(numeric.max()),
                        'mean': float(numeric.mean()),
                        'nbinsx': getattr(trace, 'nbinsx', None),
                    }
                ])

            counts = series.astype(str).value_counts(dropna=False).reset_index()
            counts.columns = ['x', 'count']
            return counts

    if trace_type == 'sunburst':
        labels = _to_list(getattr(trace, 'labels', None))
        parents = _to_list(getattr(trace, 'parents', None))
        values = _to_list(getattr(trace, 'values', None))
        data: dict[str, Any] = {}
        if labels:
            data['label'] = labels
        if parents:
            data['parent'] = parents
        if values:
            data['value'] = values
        if data:
            return pd.DataFrame(data)

    if trace_type == 'sankey':
        node = getattr(trace, 'node', None)
        link = getattr(trace, 'link', None)

        labels = _to_list(getattr(node, 'label', None)) if node is not None else []
        sources = _to_list(getattr(link, 'source', None)) if link is not None else []
        targets = _to_list(getattr(link, 'target', None)) if link is not None else []
        values = _to_list(getattr(link, 'value', None)) if link is not None else []

        rows: list[dict[str, Any]] = []
        for source, target, value in zip(sources, targets, values):
            source_label = (
                labels[source]
                if isinstance(source, int) and 0 <= source < len(labels)
                else source
            )
            target_label = (
                labels[target]
                if isinstance(target, int) and 0 <= target < len(labels)
                else target
            )
            rows.append({
                'source': source_label,
                'target': target_label,
                'value': value,
            })
        if rows:
            return pd.DataFrame(rows)

    if trace_type in {'heatmap', 'histogram2d'}:
        x = _to_list(getattr(trace, 'x', None))
        y = _to_list(getattr(trace, 'y', None))
        z = getattr(trace, 'z', None)
        if z is not None and x and y:
            try:
                matrix = pd.DataFrame(z, index=y, columns=x)
                matrix.index.name = 'y'
                return matrix.reset_index()
            except Exception:
                pass

    labels = _to_list(getattr(trace, 'labels', None))
    values = _to_list(getattr(trace, 'values', None))
    if labels or values:
        data: dict[str, Any] = {}
        if labels:
            data['label'] = labels
        if values:
            data['value'] = values
        return pd.DataFrame(data)

    return pd.DataFrame({'trace_repr': [str(trace)]})


def _log_figure_values(plot_name: str, fig: go.Figure) -> None:
    """Logs the tabular values behind each trace in a figure.

    Args:
        plot_name: Logical plot name for logging.
        fig: Figure whose traces should be logged.
    """

    _log_info(f"Plot values for {plot_name}")
    _log_info(f"trace_count={len(fig.data)}")

    for index, trace in enumerate(fig.data, start=1):
        trace_name = getattr(trace, 'name', '') or f'trace_{index}'
        trace_type = getattr(trace, 'type', 'unknown')
        _log_info(
            f'trace_index={index} trace_name={trace_name} trace_type={trace_type}'
        )
        trace_df = _trace_to_dataframe(trace)
        _log_dataframe(f'Trace table for {plot_name} [{trace_name}]', trace_df)


def create_histogram_figure(df: pd.DataFrame, field: str) -> go.Figure:
    """Creates a bar chart style histogram for a categorical field.

    Args:
        df: Source dataframe.
        field: Field name to plot.

    Returns:
        A Plotly figure.
    """

    counts = _safe_counts(df, field)
    fig = px.bar(counts, x=field, y='count', title='')
    fig.update_xaxes(title_text=humanize_field_name(field))
    fig.update_yaxes(title_text=humanize_field_name('count'))
    return fig


def create_sunburst_figure(df: pd.DataFrame, plot_fields: list[str]) -> go.Figure:
    """Creates a sunburst chart across the ordered plot fields.

    Args:
        df: Source dataframe.
        plot_fields: Ordered categorical fields for the sunburst path.

    Returns:
        A Plotly figure.
    """

    working = df.copy()
    formatted_fields: list[str] = []

    for field in plot_fields:
        label = humanize_field_name(field)
        working[label] = working[field].astype(str).map(humanize_text)
        formatted_fields.append(label)

    fig = px.sunburst(working, path=formatted_fields, title='')
    return fig


def create_transition_graph_figure(
    df: pd.DataFrame,
    plot_fields: list[str],
) -> go.Figure:
    """Creates a network style transition graph between consecutive fields.

    Args:
        df: Source dataframe.
        plot_fields: Ordered categorical fields defining transitions.

    Returns:
        A Plotly figure. Returns an empty figure when no graph nodes exist.
    """

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

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo='none',
        mode='lines',
    )

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


def _save_logged_figure(
    fig: go.Figure,
    plot_name: str,
    output_dir: Any,
    final_dir: Any,
    **export_kwargs: Any,
) -> dict[str, Any]:
    """Logs figure values and exports the figure to disk.

    Args:
        fig: Figure to export.
        plot_name: Logical name used for logging and filenames.
        output_dir: Primary output directory.
        final_dir: Optional final export directory.
        **export_kwargs: Additional keyword arguments passed to the save
            helper.

    Returns:
        The export manifest returned by ``save_plotly_figure``.
    """

    try:
        _log_figure_values(plot_name, fig)
    except Exception:
        logger.exception('Failed to log figure values for %s. Continuing with export.', plot_name)

    return save_plotly_figure(
        fig,
        plot_name,
        output_dir=output_dir,
        final_dir=final_dir,
        **export_kwargs,
    )


def create_all_plots(
    parsed_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    plot_fields: list[str],
    story_plot_fields: list[str],
    output_dirs: Any,
    auto_open_html: bool,
    histogram_fields: list[str],
    min_count: int,
    max_categories: int,
    save_final: bool,
    paper_plot_top_n: int,
    blind_spot_fields: list[str],
    image_export_timeout_seconds: int,
) -> dict[str, Any]:
    """Builds and exports all overview, histogram, and paper figures.

    Args:
        parsed_df: Parsed dataframe before empirical subset filtering.
        filtered_df: Filtered dataframe used for overview and paper figures.
        plot_fields: Ordered fields used for overview flow plots.
        story_plot_fields: Ordered fields used for the 5W1H Sankey plot.
        output_dirs: Container of output directories for plot exports.
        auto_open_html: Whether exported HTML files should be opened.
        histogram_fields: Fields for which histogram figures should be built.
        min_count: Minimum count threshold for Sankey links.
        max_categories: Maximum number of categories per Sankey stage.
        save_final: Whether to copy outputs into final figure directories.
        paper_plot_top_n: Top N parameter for selected paper figures.
        blind_spot_fields: Fields used in the blind spot figure.
        image_export_timeout_seconds: Timeout for static image exports.

    Returns:
        A nested manifest describing all successfully exported plots.
    """

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

    # Build overview flow figures.
    sankey_fig = build_sankey_figure(
        filtered_df,
        plot_fields,
        min_count=min_count,
        max_categories=max_categories,
        show_stage_headers=False,
    )
    manifest['plots']['accident_overview_sankey'] = _save_logged_figure(
        sankey_fig,
        'accident_overview_sankey',
        output_dir=output_dirs.plots,
        final_dir=output_dirs.figures,
        width=1600,
        height=900,
        **export_kwargs,
    )

    if len(story_plot_fields) >= 2:
        story_sankey_fig = build_sankey_figure(
            filtered_df,
            story_plot_fields,
            min_count=min_count,
            max_categories=max_categories,
            show_stage_headers=True,
        )
        manifest['plots']['accident_5w1h_sankey'] = _save_logged_figure(
            story_sankey_fig,
            'accident_5w1h_sankey',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            width=1600,
            height=900,
            **export_kwargs,
        )

    if len(plot_fields) >= 2:
        sunburst_fig = create_sunburst_figure(filtered_df, plot_fields)
        manifest['plots']['accident_overview_sunburst'] = _save_logged_figure(
            sunburst_fig,
            'accident_overview_sunburst',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            **export_kwargs,
        )

        transition_fig = create_transition_graph_figure(filtered_df, plot_fields)
        manifest['plots']['accident_transition_graph'] = _save_logged_figure(
            transition_fig,
            'accident_transition_graph',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            **export_kwargs,
        )

    location_map_fig, location_points_df = create_accident_location_map_figure(
        parsed_df,
        cache_path=Path(output_dirs.base) / 'geocode_cache.json',
    )
    if not location_points_df.empty:
        location_points_path = Path(output_dirs.base) / 'accident_location_points.csv'
        location_points_df.to_csv(location_points_path, index=False)
        manifest['plots']['accident_location_map'] = _save_logged_figure(
            location_map_fig,
            'accident_location_map',
            output_dir=output_dirs.plots,
            final_dir=output_dirs.figures,
            width=1600,
            height=900,
            **export_kwargs,
        )
        manifest['plots']['accident_location_points_csv'] = str(location_points_path)

    # Build categorical histograms from the parsed dataframe.
    histogram_manifest: dict[str, Any] = {}
    for field in histogram_fields:
        if field not in parsed_df.columns:
            continue

        histogram_fig = create_histogram_figure(parsed_df, field)
        histogram_manifest[field] = _save_logged_figure(
            histogram_fig,
            field,
            output_dir=output_dirs.histograms,
            final_dir=output_dirs.figures_histograms,
            **export_kwargs,
        )
    manifest['plots']['histograms'] = histogram_manifest

    # Build paper style figures used in the research summary.
    paper_manifest: dict[str, Any] = {}
    figure_specs = [
        ('taxonomy_overview', create_taxonomy_bar_figure(filtered_df, top_n=paper_plot_top_n)),
        ('blind_spots_missingness', create_blind_spot_figure(parsed_df, fields=blind_spot_fields)),
        ('accountability_by_taxonomy', create_accountability_by_taxonomy_figure(filtered_df, top_n=paper_plot_top_n)),
        ('report_completeness', create_completeness_figure(parsed_df)),
        ('taxonomy_by_road_user', create_taxonomy_by_road_user_figure(filtered_df, top_n=paper_plot_top_n)),
        ('provenance_availability', create_provenance_availability_figure(parsed_df)),
        ('context_gap', create_context_gap_figure(parsed_df)),
        ('movement_consistency', create_consistency_figure(parsed_df)),
        ('scenario_determinability', create_determinability_figure(filtered_df)),
        ('environment_profile', create_environment_profile_figure(filtered_df)),
        ('blame_confidence_alignment', create_blame_alignment_figure(parsed_df)),
        ('stopped_av_subtype', create_stopped_av_subtype_figure(filtered_df)),
        ('intersection_detail_quality', create_intersection_detail_figure(parsed_df)),
    ]

    for plot_name, fig in figure_specs:
        paper_manifest[plot_name] = _save_logged_figure(
            fig,
            plot_name,
            output_dir=output_dirs.paper,
            final_dir=output_dirs.figures_paper,
            **export_kwargs,
        )

    manifest['plots']['paper'] = paper_manifest
    return manifest
