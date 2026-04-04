from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import common
import plotly as py
import plotly.graph_objects as go

from utils.io import maybe_open_html

logger = logging.getLogger(__name__)

_WORKER_CODE = r'''
import sys
from pathlib import Path
import plotly.io as pio

json_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
fmt = sys.argv[3]
width = int(sys.argv[4])
height = int(sys.argv[5])
scale = int(sys.argv[6])
fig = pio.from_json(json_path.read_text(encoding="utf-8"))
fig.write_image(str(out_path), format=fmt, width=width, height=height, scale=scale)
'''


# Edit figure styles directly here.
#
# The key must match the figure filename passed to save_plotly_figure().
# Examples in this project include:
# - histogram figures: road_user_type, av_mode_group, collision_group, ...
# - paper figures: taxonomy_overview, taxonomy_by_road_user, context_gap, ...
# - overview figures: accident_overview_sankey, accident_overview_sunburst, ...
#
# The default block is loaded from the shared config via common.get_configs().
# Figure specific blocks override only that figure.
FIGURE_STYLE_OVERRIDES: dict[str, dict[str, Any]] = {
    'accountability_by_taxonomy': {
        # 'legend_orientation': 'h',
        'legend_x': 0.9,
        'legend_y': 0.8,
        'legend_xanchor': 'center',
        'legend_yanchor': 'bottom',
    },
}


def _get_common_config(*keys: str, default: Any = None) -> Any:
    for key in keys:
        try:
            value = common.get_configs(key)
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_default_figure_style() -> dict[str, Any]:
    font_family = _get_common_config('font_family', default='Arial')
    font_size = _coerce_int(_get_common_config('font_size', default=12), 12)
    title_font_size = _coerce_int(_get_common_config('title_font_size', default=18), 18)
    legend_font_size = _coerce_int(_get_common_config('legend_font_size', default=font_size), font_size)
    legend_title_font_size = _coerce_int(_get_common_config('legend_title_font_size', default=legend_font_size), legend_font_size)
    axis_title_font_size = _coerce_int(_get_common_config('axis_title_font_size', default=font_size), font_size)
    axis_tick_font_size = _coerce_int(_get_common_config('axis_tick_font_size', default=font_size), font_size)
    return {
        'font_family': font_family,
        'font_size': font_size,
        'title_font_size': title_font_size,
        'legend_font_size': legend_font_size,
        'legend_title_font_size': legend_title_font_size,
        'axis_title_font_size': axis_title_font_size,
        'axis_tick_font_size': axis_tick_font_size,
        'template': _get_common_config('plotly_template', 'template', default='plotly_white'),
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _run_image_worker(fig: go.Figure, out_path: Path, fmt: str, width: int, height: int, scale: int,
                      timeout_seconds: int) -> tuple[bool, str]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / 'figure.json'
        json_path.write_text(fig.to_json(), encoding='utf-8')
        cmd = [
            sys.executable,
            '-c',
            _WORKER_CODE,
            str(json_path),
            str(out_path),
            fmt,
            str(width),
            str(height),
            str(scale),
        ]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            if out_path.exists():
                return True, completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode == 0:
                return False, 'worker finished without creating output file'
            return False, completed.stderr.strip() or completed.stdout.strip() or f'worker exited with code {completed.returncode}'
        except subprocess.TimeoutExpired:
            if out_path.exists():
                return True, f'timed out after {timeout_seconds}s after writing file'
            return False, f'timed out after {timeout_seconds}s'
        except Exception as exc:
            if out_path.exists():
                return True, str(exc)
            return False, str(exc)


def _save_png(fig: go.Figure, png_path: Path, width: int, height: int, scale: int, timeout_seconds: int) -> tuple[bool, str]:
    return _run_image_worker(fig, png_path, 'png', width=width, height=height, scale=scale, timeout_seconds=timeout_seconds)


def _save_eps(fig: go.Figure, eps_path: Path, width: int, height: int, scale: int, timeout_seconds: int) -> tuple[bool, str]:
    ok, message = _run_image_worker(fig, eps_path, 'eps', width=width, height=height, scale=scale, timeout_seconds=timeout_seconds)
    if ok:
        return True, message

    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = Path(tmpdir) / 'temp.svg'
        ok_svg, svg_message = _run_image_worker(fig, svg_path, 'svg', width=width, height=height, scale=scale, timeout_seconds=timeout_seconds)
        if ok_svg and svg_path.exists():
            try:
                cairosvg.svg2ps(url=str(svg_path), write_to=str(eps_path))
                if eps_path.exists():
                    return True, svg_message
            except Exception as exc:
                message = f'{message}; svg fallback failed: {exc}' if message else f'svg fallback failed: {exc}'

    pdftops = shutil.which('pdftops')
    if pdftops:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / 'temp.pdf'
            ok_pdf, pdf_message = _run_image_worker(fig, pdf_path, 'pdf', width=width, height=height, scale=scale, timeout_seconds=timeout_seconds)
            if ok_pdf and pdf_path.exists():
                try:
                    subprocess.run([pdftops, '-eps', str(pdf_path), str(eps_path)], check=True, capture_output=True, text=True, timeout=timeout_seconds)
                    if eps_path.exists():
                        return True, pdf_message
                except Exception as exc:
                    message = f'{message}; pdf fallback failed: {exc}' if message else f'pdf fallback failed: {exc}'

    return False, message or 'EPS export failed in all available backends'


def _merged_figure_style(filename: str) -> dict[str, Any]:
    style: dict[str, Any] = {}
    style.update(_load_default_figure_style())
    style.update(FIGURE_STYLE_OVERRIDES.get(filename, {}))
    return style


def _font_dict(*, family: Any = None, size: Any = None, color: Any = None) -> dict[str, Any]:
    font: dict[str, Any] = {}
    if family is not None:
        font['family'] = family
    if size is not None:
        font['size'] = size
    if color is not None:
        font['color'] = color
    return font


def _apply_figure_style(fig: go.Figure, filename: str) -> go.Figure:
    style = _merged_figure_style(filename)
    if not style:
        return fig

    base_font = _font_dict(
        family=style.get('font_family'),
        size=style.get('font_size'),
        color=style.get('font_color'),
    )
    title_font = _font_dict(
        family=style.get('title_font_family', style.get('font_family')),
        size=style.get('title_font_size'),
        color=style.get('title_font_color', style.get('font_color')),
    )
    legend_font = _font_dict(
        family=style.get('legend_font_family', style.get('font_family')),
        size=style.get('legend_font_size', style.get('font_size')),
        color=style.get('legend_font_color', style.get('font_color')),
    )
    legend_title_font = _font_dict(
        family=style.get('legend_title_font_family', style.get('legend_font_family', style.get('font_family'))),
        size=style.get('legend_title_font_size', style.get('legend_font_size', style.get('font_size'))),
        color=style.get('legend_title_font_color', style.get('legend_font_color', style.get('font_color'))),
    )

    layout_updates: dict[str, Any] = {}
    if base_font:
        layout_updates['font'] = base_font
    if style.get('template'):
        layout_updates['template'] = style['template']
    if style.get('paper_bgcolor') is not None:
        layout_updates['paper_bgcolor'] = style['paper_bgcolor']
    if style.get('plot_bgcolor') is not None:
        layout_updates['plot_bgcolor'] = style['plot_bgcolor']
    if title_font:
        layout_updates['title'] = {'font': title_font}
    if legend_font or legend_title_font or style.get('legend_orientation') is not None or style.get('legend_x') is not None or style.get('legend_y') is not None or style.get('legend_xanchor') is not None or style.get('legend_yanchor') is not None:
        layout_updates['legend'] = {}
        if legend_font:
            layout_updates['legend']['font'] = legend_font
        if legend_title_font:
            layout_updates['legend']['title'] = {'font': legend_title_font}
        if style.get('legend_orientation') is not None:
            layout_updates['legend']['orientation'] = style['legend_orientation']
        if style.get('legend_x') is not None:
            layout_updates['legend']['x'] = style['legend_x']
        if style.get('legend_y') is not None:
            layout_updates['legend']['y'] = style['legend_y']
        if style.get('legend_xanchor') is not None:
            layout_updates['legend']['xanchor'] = style['legend_xanchor']
        if style.get('legend_yanchor') is not None:
            layout_updates['legend']['yanchor'] = style['legend_yanchor']

    if layout_updates:
        fig.update_layout(**layout_updates)

    axis_title_font = _font_dict(
        family=style.get('axis_title_font_family', style.get('font_family')),
        size=style.get('axis_title_font_size'),
        color=style.get('axis_title_font_color', style.get('font_color')),
    )
    axis_tick_font = _font_dict(
        family=style.get('axis_tick_font_family', style.get('font_family')),
        size=style.get('axis_tick_font_size', style.get('font_size')),
        color=style.get('axis_tick_font_color', style.get('font_color')),
    )
    xaxis_title_font = _font_dict(
        family=style.get('xaxis_title_font_family', style.get('axis_title_font_family', style.get('font_family'))),
        size=style.get('xaxis_title_font_size', style.get('axis_title_font_size')),
        color=style.get('xaxis_title_font_color', style.get('axis_title_font_color', style.get('font_color'))),
    )
    yaxis_title_font = _font_dict(
        family=style.get('yaxis_title_font_family', style.get('axis_title_font_family', style.get('font_family'))),
        size=style.get('yaxis_title_font_size', style.get('axis_title_font_size')),
        color=style.get('yaxis_title_font_color', style.get('axis_title_font_color', style.get('font_color'))),
    )
    xaxis_tick_font = _font_dict(
        family=style.get('xaxis_tick_font_family', style.get('axis_tick_font_family', style.get('font_family'))),
        size=style.get('xaxis_tick_font_size', style.get('axis_tick_font_size', style.get('font_size'))),
        color=style.get('xaxis_tick_font_color', style.get('axis_tick_font_color', style.get('font_color'))),
    )
    yaxis_tick_font = _font_dict(
        family=style.get('yaxis_tick_font_family', style.get('axis_tick_font_family', style.get('font_family'))),
        size=style.get('yaxis_tick_font_size', style.get('axis_tick_font_size', style.get('font_size'))),
        color=style.get('yaxis_tick_font_color', style.get('axis_tick_font_color', style.get('font_color'))),
    )

    xaxis_updates: dict[str, Any] = {}
    yaxis_updates: dict[str, Any] = {}

    if axis_title_font:
        xaxis_updates['title_font'] = axis_title_font
        yaxis_updates['title_font'] = axis_title_font
    if xaxis_title_font:
        xaxis_updates['title_font'] = xaxis_title_font
    if yaxis_title_font:
        yaxis_updates['title_font'] = yaxis_title_font

    if axis_tick_font:
        xaxis_updates['tickfont'] = axis_tick_font
        yaxis_updates['tickfont'] = axis_tick_font
    if xaxis_tick_font:
        xaxis_updates['tickfont'] = xaxis_tick_font
    if yaxis_tick_font:
        yaxis_updates['tickfont'] = yaxis_tick_font

    if style.get('xaxis_tick_angle') is not None:
        xaxis_updates['tickangle'] = style['xaxis_tick_angle']
    if style.get('yaxis_tick_angle') is not None:
        yaxis_updates['tickangle'] = style['yaxis_tick_angle']

    if xaxis_updates:
        fig.update_xaxes(**xaxis_updates)
    if yaxis_updates:
        fig.update_yaxes(**yaxis_updates)

    return fig


def save_plotly_figure(
    fig: go.Figure,
    filename: str,
    output_dir: Path,
    final_dir: Path | None = None,
    auto_open_html: bool = False,
    width: int = 1600,
    height: int = 900,
    scale: int = 1,
    save_final: bool = True,
    save_png: bool = True,
    save_eps: bool = True,
    export_timeout_seconds: int = 60,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if final_dir is not None:
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

    fig = _apply_figure_style(fig, filename)

    manifest: dict[str, str] = {}

    html_path = output_dir / f'{filename}.html'
    py.offline.plot(fig, filename=str(html_path), auto_open=False)
    manifest['html'] = str(html_path)
    maybe_open_html(html_path, auto_open_html)
    if save_final and final_dir is not None:
        final_html = final_dir / f'{filename}.html'
        py.offline.plot(fig, filename=str(final_html), auto_open=False)
        manifest['html_final'] = str(final_html)

    if save_png:
        png_path = output_dir / f'{filename}.png'
        ok_png, png_message = _save_png(fig, png_path, width=width, height=height, scale=scale, timeout_seconds=export_timeout_seconds)
        if ok_png and png_path.exists():
            manifest['png'] = str(png_path)
            if save_final and final_dir is not None:
                final_png = final_dir / f'{filename}.png'
                _copy_if_exists(png_path, final_png)
                manifest['png_final'] = str(final_png)
        else:
            logger.error('PNG export failed for %s: %s', filename, png_message)

    if save_eps:
        eps_path = output_dir / f'{filename}.eps'
        ok_eps, eps_message = _save_eps(fig, eps_path, width=width, height=height, scale=scale, timeout_seconds=export_timeout_seconds)
        if ok_eps and eps_path.exists():
            manifest['eps'] = str(eps_path)
            if save_final and final_dir is not None:
                final_eps = final_dir / f'{filename}.eps'
                _copy_if_exists(eps_path, final_eps)
                manifest['eps_final'] = str(final_eps)
        else:
            logger.error('EPS export failed for %s: %s', filename, eps_message)

    logger.info(
        'Exported plot %s | html=%s png=%s eps=%s',
        filename,
        'yes' if 'html' in manifest else 'no',
        'yes' if 'png' in manifest else 'no',
        'yes' if 'eps' in manifest else 'no',
    )
    return manifest
