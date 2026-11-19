"""Shared visual design for project figures.

One palette and a small set of chart builders, so every exported figure uses the same colours, label format and
bar geometry. Colours are the validated categorical palette (fixed slot order, never cycled) plus a neutral grey
reserved for missing or unknown categories, so "not available" never looks like a real category.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import plotly.graph_objects as go

# Categorical slots in fixed order (validated for colour vision deficiency on adjacent pairs).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
PRIMARY = SERIES[0]
MISSING_COLOUR = '#bdbcb6'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e8e7e2'
SURFACE = '#ffffff'
# Sequential blue ramp, light to dark, for magnitudes (heatmaps).
SEQUENTIAL_BLUES = ['#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

MISSING_LABELS = {'', 'na', 'n/a', 'nan', 'none', 'unknown', 'not stated', 'unavailable', 'missing'}

ROW_PX = 46
BASE_PX = 150


def is_missing_label(label: object) -> bool:
    """Returns whether a display label stands for a missing or unknown value."""

    return str(label).strip().lower() in MISSING_LABELS


def with_alpha(hex_colour: str, alpha: float) -> str:
    """Returns an rgba() string for a hex colour."""

    value = hex_colour.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def fold_long_tail(
    labels: Sequence[str],
    counts: Sequence[int],
    *,
    max_bars: int,
    min_count: int = 1,
    other_label: str = 'Other',
) -> tuple[list[str], list[int]]:
    """Keeps the largest categories and folds the rest into one "Other (k categories)" bar.

    Missing or unknown categories are never folded, so their size stays visible.
    """

    rows = list(zip(labels, counts))
    present = [r for r in rows if not is_missing_label(r[0])]
    missing = [r for r in rows if is_missing_label(r[0])]
    keep = max(max_bars - len(missing), 1)
    small = [r for r in present if r[1] < min_count]
    if len(small) > 1:
        present = [r for r in present if r[1] >= min_count]
        present.sort(key=lambda r: -r[1])
        folded = (f'{other_label} ({len(small)} categories)', int(sum(r[1] for r in small)))
        return [r[0] for r in present] + [folded[0]] + [r[0] for r in missing], \
            [int(r[1]) for r in present] + [folded[1]] + [int(r[1]) for r in missing]
    if len(present) <= keep:
        return [r[0] for r in present + missing], [int(r[1]) for r in present + missing]
    present.sort(key=lambda r: -r[1])
    head, tail = present[:keep - 1], present[keep - 1:]
    folded = (f'{other_label} ({len(tail)} categories)', int(sum(r[1] for r in tail)))
    rows = head + [folded] + missing
    return [r[0] for r in rows], [int(r[1]) for r in rows]


def order_rows(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    order: Iterable[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Orders rows by a given category order, else by value; missing rows always go last."""

    rows = list(zip(labels, values))
    if order is not None:
        rank = {label: i for i, label in enumerate(order)}
        rows.sort(key=lambda r: (is_missing_label(r[0]), rank.get(r[0], len(rank)), -r[1]))
    else:
        rows.sort(key=lambda r: (is_missing_label(r[0]) or r[0].startswith('Other ('), -r[1]))
    return [r[0] for r in rows], [r[1] for r in rows]


def apply_base_layout(fig: go.Figure, *, height: int | None = None, legend: bool = False) -> go.Figure:
    """Applies recessive axes, light grid and the shared surface to a figure."""

    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT),
        showlegend=legend,
        margin=dict(t=20, r=30, b=10, l=10),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=TEXT)),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID, ticks='',
                     tickfont=dict(color=TEXT_SECONDARY), title_font=dict(color=TEXT_SECONDARY), automargin=True)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=TEXT_SECONDARY, ticks='',
                     tickfont=dict(color=TEXT), title_font=dict(color=TEXT_SECONDARY), automargin=True)
    return fig


def horizontal_bar_height(n_rows: int) -> int:
    """Returns a figure height that keeps bar thickness constant across charts."""

    return max(360, BASE_PX + ROW_PX * max(n_rows, 1))


def count_bar_figure(
    labels: Sequence[str],
    counts: Sequence[int],
    *,
    total: int | None = None,
    order: Iterable[str] | None = None,
    max_bars: int = 12,
    min_count: int = 1,
    value_title: str = 'Number of reports',
    category_title: str = '',
    colours: dict[str, str] | None = None,
) -> go.Figure:
    """Horizontal bar chart of category counts labelled "n (share%)".

    Long tails fold into "Other (k categories)", missing or unknown categories are grey and listed last, and
    rows run top to bottom in the given order (or by count).
    """

    labels, counts = fold_long_tail(list(labels), list(counts), max_bars=max_bars, min_count=min_count)
    labels, counts = order_rows(labels, counts, order=order)
    total = total if total else max(sum(counts), 1)
    colour_for = colours or {}
    bar_colours = [MISSING_COLOUR if is_missing_label(lbl) else colour_for.get(lbl, PRIMARY) for lbl in labels]
    text = [f'{int(c):,} ({100 * c / total:.1f}%)' for c in counts]
    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation='h', marker=dict(color=bar_colours, line=dict(width=0)),
        text=text, textposition='outside', cliponaxis=False, textfont=dict(color=TEXT_SECONDARY),
        hovertemplate='%{y}<br>%{x:,} reports<extra></extra>', width=0.62,
    ))
    apply_base_layout(fig, height=horizontal_bar_height(len(labels)))
    fig.update_xaxes(title_text=value_title, range=[0, max(counts) * 1.28 if counts else 1], tickformat=',')
    fig.update_yaxes(title_text=category_title, categoryorder='array', categoryarray=labels, autorange='reversed')
    return fig


def rate_bar_figure(
    labels: Sequence[str],
    rates: Sequence[float],
    *,
    groups: Sequence[str] | None = None,
    group_colours: dict[str, str] | None = None,
    value_title: str = 'Share of reports',
    category_title: str = '',
    decimals: int = 0,
) -> go.Figure:
    """Horizontal bar chart of shares (0 to 1) on a 0 to 100% axis.

    With ``groups``, bars are coloured by group and a legend names each group (rows stay in the given order).
    """

    fig = go.Figure()
    fmt = f'{{:.{decimals}%}}'
    if groups is None:
        fig.add_trace(go.Bar(
            x=list(rates), y=list(labels), orientation='h', marker=dict(color=PRIMARY), width=0.62,
            text=[fmt.format(r) for r in rates], textposition='outside', cliponaxis=False,
            textfont=dict(color=TEXT_SECONDARY), hovertemplate='%{y}: %{x:.1%}<extra></extra>',
        ))
    else:
        colours = group_colours or {}
        for index, group in enumerate(dict.fromkeys(groups)):
            xs = [r if g == group else None for r, g in zip(rates, groups)]
            fig.add_trace(go.Bar(
                x=xs, y=list(labels), orientation='h', name=group, width=0.62,
                marker=dict(color=colours.get(group, SERIES[index % len(SERIES)])),
                text=[fmt.format(r) if g == group else '' for r, g in zip(rates, groups)],
                textposition='outside', cliponaxis=False, textfont=dict(color=TEXT_SECONDARY),
                hovertemplate='%{y}: %{x:.1%}<extra>' + group + '</extra>',
            ))
        fig.update_layout(barmode='overlay')
    apply_base_layout(fig, height=horizontal_bar_height(len(labels)), legend=groups is not None)
    fig.update_xaxes(title_text=value_title, range=[0, 1.12], tickformat='.0%', dtick=0.2)
    fig.update_yaxes(title_text=category_title, categoryorder='array', categoryarray=list(labels), autorange='reversed')
    if groups is not None:
        fig.update_layout(legend=dict(orientation='h', x=0, y=1.02, xanchor='left', yanchor='bottom', title_text=''),
                          margin=dict(t=70))
    return fig
