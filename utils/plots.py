from __future__ import annotations

"""Helpers for styling and exporting Plotly figures.

This module centralises Plotly figure styling and file export. It provides:

* shared default style loading from project config
* figure specific style overrides
* HTML, PNG, PDF, and EPS export helpers
* stable EPS export through a PDF to EPS conversion path
* optional browser opening for generated HTML files
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import common
import plotly.graph_objects as go

from utils.io import maybe_open_html

logger = logging.getLogger(__name__)

# Worker script executed in a separate Python process for static image export.
# Running export out of process helps isolate some backend issues.
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
    "accountability_by_taxonomy": {
        # "legend_orientation": "h",
        "legend_x": 0.9,
        "legend_y": 0.8,
        "legend_xanchor": "center",
        "legend_yanchor": "bottom",
    },
}


_POSTSCRIPT_FONT_MAP = {
    "arial": "Helvetica",
    "arial black": "Helvetica-Bold",
    "helvetica": "Helvetica",
    "times new roman": "Times-Roman",
    "times": "Times-Roman",
    "courier new": "Courier",
    "courier": "Courier",
}


def _get_common_config(*keys: str, default: Any = None) -> Any:
    """Returns the first non null config value found for the given keys.

    Args:
        *keys: Candidate config keys to try in order.
        default: Fallback value when no key resolves successfully.

    Returns:
        The first resolved config value, or ``default`` when none is available.
    """

    for key in keys:
        try:
            value = common.get_configs(key)
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _coerce_int(value: Any, default: int) -> int:
    """Converts a value to ``int`` with a safe fallback.

    Args:
        value: Raw input value.
        default: Value to return when conversion fails.

    Returns:
        The coerced integer value, or ``default``.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_default_figure_style() -> dict[str, Any]:
    """Loads the shared default figure style from project config.

    Returns:
        A style dictionary containing resolved default font and template
        settings for Plotly figures.
    """

    font_family = _get_common_config("font_family", default="Arial")
    font_size = _coerce_int(_get_common_config("font_size", default=12), 12)
    title_font_size = _coerce_int(
        _get_common_config("title_font_size", default=18),
        18,
    )
    legend_font_size = _coerce_int(
        _get_common_config("legend_font_size", default=font_size),
        font_size,
    )
    legend_title_font_size = _coerce_int(
        _get_common_config(
            "legend_title_font_size",
            default=legend_font_size,
        ),
        legend_font_size,
    )
    axis_title_font_size = _coerce_int(
        _get_common_config("axis_title_font_size", default=font_size),
        font_size,
    )
    axis_tick_font_size = _coerce_int(
        _get_common_config("axis_tick_font_size", default=font_size),
        font_size,
    )

    return {
        "font_family": font_family,
        "font_size": font_size,
        "title_font_size": title_font_size,
        "legend_font_size": legend_font_size,
        "legend_title_font_size": legend_title_font_size,
        "axis_title_font_size": axis_title_font_size,
        "axis_tick_font_size": axis_tick_font_size,
        "template": _get_common_config(
            "plotly_template",
            "template",
            default="plotly_white",
        ),
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    """Copies a file only when the source exists.

    Args:
        src: Source file path.
        dst: Destination file path.
    """

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _run_image_worker(
    fig: go.Figure,
    out_path: Path,
    fmt: str,
    width: int,
    height: int,
    scale: int,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Exports a Plotly figure through a separate worker process.

    Args:
        fig: Figure to export.
        out_path: Destination image path.
        fmt: Output format such as ``png``, ``svg``, ``pdf``, or ``eps``.
        width: Export width in pixels.
        height: Export height in pixels.
        scale: Plotly export scale factor.
        timeout_seconds: Maximum worker runtime.

    Returns:
        A tuple containing:
            * ``True`` if export produced a file, else ``False``
            * a status or error message from the export attempt
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "figure.json"
        json_path.write_text(fig.to_json(), encoding="utf-8")

        cmd = [
            sys.executable,
            "-c",
            _WORKER_CODE,
            str(json_path),
            str(out_path),
            fmt,
            str(width),
            str(height),
            str(scale),
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if out_path.exists():
                return True, completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode == 0:
                return False, "worker finished without creating output file"
            return (
                False,
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"worker exited with code {completed.returncode}",
            )
        except subprocess.TimeoutExpired:
            if out_path.exists():
                return True, f"timed out after {timeout_seconds}s after writing file"
            return False, f"timed out after {timeout_seconds}s"
        except Exception as exc:
            if out_path.exists():
                return True, str(exc)
            return False, str(exc)


def _save_png(
    fig: go.Figure,
    png_path: Path,
    width: int,
    height: int,
    scale: int,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Exports a figure to PNG.

    Args:
        fig: Figure to export.
        png_path: Destination PNG path.
        width: Export width in pixels.
        height: Export height in pixels.
        scale: Plotly export scale factor.
        timeout_seconds: Maximum export runtime.

    Returns:
        A success flag and status message.
    """

    return _run_image_worker(
        fig,
        png_path,
        "png",
        width=width,
        height=height,
        scale=scale,
        timeout_seconds=timeout_seconds,
    )


def _save_pdf(
    fig: go.Figure,
    pdf_path: Path,
    width: int,
    height: int,
    scale: int,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Exports a figure to PDF.

    Args:
        fig: Figure to export.
        pdf_path: Destination PDF path.
        width: Export width in pixels.
        height: Export height in pixels.
        scale: Plotly export scale factor.
        timeout_seconds: Maximum export runtime.

    Returns:
        A success flag and status message.
    """

    return _run_image_worker(
        fig,
        pdf_path,
        "pdf",
        width=width,
        height=height,
        scale=scale,
        timeout_seconds=timeout_seconds,
    )


def _save_eps(
    fig: go.Figure,
    eps_path: Path,
    width: int,
    height: int,
    scale: int,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Exports a figure to EPS.

    The preferred path is PDF export followed by ``pdftops -eps`` because it
    tends to preserve layout more consistently than direct EPS export. Direct
    EPS export remains as a fallback when ``pdftops`` is unavailable.

    Args:
        fig: Figure to export.
        eps_path: Destination EPS path.
        width: Export width in pixels.
        height: Export height in pixels.
        scale: Plotly export scale factor.
        timeout_seconds: Maximum export runtime for each backend step.

    Returns:
        A success flag and status message.
    """

    pdftops = shutil.which("pdftops")
    if pdftops:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "temp.pdf"
            ok_pdf, pdf_message = _save_pdf(
                fig,
                pdf_path,
                width=width,
                height=height,
                scale=scale,
                timeout_seconds=timeout_seconds,
            )
            if ok_pdf and pdf_path.exists():
                try:
                    subprocess.run(
                        [pdftops, "-eps", str(pdf_path), str(eps_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout_seconds,
                    )
                    if eps_path.exists():
                        return True, pdf_message or "exported via pdf to eps"
                except Exception as exc:
                    pdf_message = (
                        f"{pdf_message}; pdf to eps conversion failed: {exc}"
                        if pdf_message
                        else f"pdf to eps conversion failed: {exc}"
                    )
                    logger.warning(pdf_message)

    # Direct EPS export is left as a final fallback only.
    return _run_image_worker(
        fig,
        eps_path,
        "eps",
        width=width,
        height=height,
        scale=scale,
        timeout_seconds=timeout_seconds,
    )


def _merged_figure_style(filename: str) -> dict[str, Any]:
    """Builds the final style dictionary for a figure.

    Args:
        filename: Logical figure name used as the style override key.

    Returns:
        A merged style dictionary combining defaults and figure specific
        overrides.
    """

    style: dict[str, Any] = {}
    style.update(_load_default_figure_style())
    style.update(FIGURE_STYLE_OVERRIDES.get(filename, {}))
    return style


def _font_dict(*, family: Any = None, size: Any = None, color: Any = None) -> dict[str, Any]:
    """Builds a Plotly font dictionary while skipping null values.

    Args:
        family: Font family value.
        size: Font size value.
        color: Font colour value.

    Returns:
        A dictionary containing only the provided font properties.
    """

    font: dict[str, Any] = {}
    if family is not None:
        font["family"] = family
    if size is not None:
        font["size"] = size
    if color is not None:
        font["color"] = color
    return font


def _apply_figure_style(fig: go.Figure, filename: str) -> go.Figure:
    """Applies merged style settings to a Plotly figure.

    Args:
        fig: Figure to style.
        filename: Logical figure name used for override lookup.

    Returns:
        The styled figure.
    """

    style = _merged_figure_style(filename)
    if not style:
        return fig

    base_font = _font_dict(
        family=style.get("font_family"),
        size=style.get("font_size"),
        color=style.get("font_color"),
    )
    title_font = _font_dict(
        family=style.get("title_font_family", style.get("font_family")),
        size=style.get("title_font_size"),
        color=style.get("title_font_color", style.get("font_color")),
    )
    legend_font = _font_dict(
        family=style.get("legend_font_family", style.get("font_family")),
        size=style.get("legend_font_size", style.get("font_size")),
        color=style.get("legend_font_color", style.get("font_color")),
    )
    legend_title_font = _font_dict(
        family=style.get(
            "legend_title_font_family",
            style.get("legend_font_family", style.get("font_family")),
        ),
        size=style.get(
            "legend_title_font_size",
            style.get("legend_font_size", style.get("font_size")),
        ),
        color=style.get(
            "legend_title_font_color",
            style.get("legend_font_color", style.get("font_color")),
        ),
    )

    layout_updates: dict[str, Any] = {}
    if base_font:
        layout_updates["font"] = base_font
    if style.get("template"):
        layout_updates["template"] = style["template"]
    if style.get("paper_bgcolor") is not None:
        layout_updates["paper_bgcolor"] = style["paper_bgcolor"]
    if style.get("plot_bgcolor") is not None:
        layout_updates["plot_bgcolor"] = style["plot_bgcolor"]
    if title_font:
        layout_updates["title"] = {"font": title_font}

    has_legend_updates = (
        legend_font
        or legend_title_font
        or style.get("legend_orientation") is not None
        or style.get("legend_x") is not None
        or style.get("legend_y") is not None
        or style.get("legend_xanchor") is not None
        or style.get("legend_yanchor") is not None
    )
    if has_legend_updates:
        layout_updates["legend"] = {}
        if legend_font:
            layout_updates["legend"]["font"] = legend_font
        if legend_title_font:
            layout_updates["legend"]["title"] = {"font": legend_title_font}
        if style.get("legend_orientation") is not None:
            layout_updates["legend"]["orientation"] = style["legend_orientation"]
        if style.get("legend_x") is not None:
            layout_updates["legend"]["x"] = style["legend_x"]
        if style.get("legend_y") is not None:
            layout_updates["legend"]["y"] = style["legend_y"]
        if style.get("legend_xanchor") is not None:
            layout_updates["legend"]["xanchor"] = style["legend_xanchor"]
        if style.get("legend_yanchor") is not None:
            layout_updates["legend"]["yanchor"] = style["legend_yanchor"]

    if layout_updates:
        fig.update_layout(**layout_updates)

    axis_title_font = _font_dict(
        family=style.get("axis_title_font_family", style.get("font_family")),
        size=style.get("axis_title_font_size"),
        color=style.get("axis_title_font_color", style.get("font_color")),
    )
    axis_tick_font = _font_dict(
        family=style.get("axis_tick_font_family", style.get("font_family")),
        size=style.get("axis_tick_font_size", style.get("font_size")),
        color=style.get("axis_tick_font_color", style.get("font_color")),
    )
    xaxis_title_font = _font_dict(
        family=style.get(
            "xaxis_title_font_family",
            style.get("axis_title_font_family", style.get("font_family")),
        ),
        size=style.get("xaxis_title_font_size", style.get("axis_title_font_size")),
        color=style.get(
            "xaxis_title_font_color",
            style.get("axis_title_font_color", style.get("font_color")),
        ),
    )
    yaxis_title_font = _font_dict(
        family=style.get(
            "yaxis_title_font_family",
            style.get("axis_title_font_family", style.get("font_family")),
        ),
        size=style.get("yaxis_title_font_size", style.get("axis_title_font_size")),
        color=style.get(
            "yaxis_title_font_color",
            style.get("axis_title_font_color", style.get("font_color")),
        ),
    )
    xaxis_tick_font = _font_dict(
        family=style.get(
            "xaxis_tick_font_family",
            style.get("axis_tick_font_family", style.get("font_family")),
        ),
        size=style.get(
            "xaxis_tick_font_size",
            style.get("axis_tick_font_size", style.get("font_size")),
        ),
        color=style.get(
            "xaxis_tick_font_color",
            style.get("axis_tick_font_color", style.get("font_color")),
        ),
    )
    yaxis_tick_font = _font_dict(
        family=style.get(
            "yaxis_tick_font_family",
            style.get("axis_tick_font_family", style.get("font_family")),
        ),
        size=style.get(
            "yaxis_tick_font_size",
            style.get("axis_tick_font_size", style.get("font_size")),
        ),
        color=style.get(
            "yaxis_tick_font_color",
            style.get("axis_tick_font_color", style.get("font_color")),
        ),
    )

    xaxis_updates: dict[str, Any] = {}
    yaxis_updates: dict[str, Any] = {}

    if axis_title_font:
        xaxis_updates["title_font"] = axis_title_font
        yaxis_updates["title_font"] = axis_title_font
    if xaxis_title_font:
        xaxis_updates["title_font"] = xaxis_title_font
    if yaxis_title_font:
        yaxis_updates["title_font"] = yaxis_title_font

    if axis_tick_font:
        xaxis_updates["tickfont"] = axis_tick_font
        yaxis_updates["tickfont"] = axis_tick_font
    if xaxis_tick_font:
        xaxis_updates["tickfont"] = xaxis_tick_font
    if yaxis_tick_font:
        yaxis_updates["tickfont"] = yaxis_tick_font

    if style.get("xaxis_tick_angle") is not None:
        xaxis_updates["tickangle"] = style["xaxis_tick_angle"]
    if style.get("yaxis_tick_angle") is not None:
        yaxis_updates["tickangle"] = style["yaxis_tick_angle"]

    if xaxis_updates:
        fig.update_xaxes(**xaxis_updates)
    if yaxis_updates:
        fig.update_yaxes(**yaxis_updates)

    return fig


def _clone_figure(fig: go.Figure) -> go.Figure:
    """Returns a detached clone of a figure.

    Args:
        fig: Source figure.

    Returns:
        A deep cloned ``go.Figure``.
    """

    return go.Figure(fig)


def _map_postscript_font_name(value: str) -> str:
    """Maps common desktop font names to PostScript friendly names.

    Args:
        value: Raw font family string.

    Returns:
        The mapped font family string.
    """

    fonts = [part.strip() for part in value.split(",") if part.strip()]
    mapped: list[str] = []
    for font in fonts:
        mapped.append(_POSTSCRIPT_FONT_MAP.get(font.lower(), font))
    return ", ".join(mapped)


def _coerce_postscript_fonts(fig: go.Figure) -> go.Figure:
    """Rewrites common font family names to PostScript safe equivalents.

    Args:
        fig: Figure to adjust.

    Returns:
        The updated figure.
    """

    for trace in fig.data:
        try:
            if getattr(trace, "textfont", None) and getattr(trace.textfont, "family", None):
                trace.textfont.family = _map_postscript_font_name(trace.textfont.family)
        except Exception:
            pass

        try:
            if getattr(trace, "insidetextfont", None) and getattr(trace.insidetextfont, "family", None):
                trace.insidetextfont.family = _map_postscript_font_name(trace.insidetextfont.family)
        except Exception:
            pass

        try:
            if getattr(trace, "outsidetextfont", None) and getattr(trace.outsidetextfont, "family", None):
                trace.outsidetextfont.family = _map_postscript_font_name(trace.outsidetextfont.family)
        except Exception:
            pass

        try:
            if getattr(trace, "hoverlabel", None) and getattr(trace.hoverlabel, "font", None):
                family = getattr(trace.hoverlabel.font, "family", None)
                if family:
                    trace.hoverlabel.font.family = _map_postscript_font_name(family)
        except Exception:
            pass

    try:
        if fig.layout.font and fig.layout.font.family:
            fig.layout.font.family = _map_postscript_font_name(fig.layout.font.family)
    except Exception:
        pass

    try:
        if fig.layout.title and fig.layout.title.font and fig.layout.title.font.family:
            fig.layout.title.font.family = _map_postscript_font_name(
                fig.layout.title.font.family
            )
    except Exception:
        pass

    try:
        if fig.layout.legend and fig.layout.legend.font and fig.layout.legend.font.family:
            fig.layout.legend.font.family = _map_postscript_font_name(
                fig.layout.legend.font.family
            )
    except Exception:
        pass

    try:
        if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.font and fig.layout.legend.title.font.family:  # noqa:E501
            fig.layout.legend.title.font.family = _map_postscript_font_name(
                fig.layout.legend.title.font.family
            )
    except Exception:
        pass

    for axis_name in (
        "xaxis",
        "xaxis2",
        "xaxis3",
        "yaxis",
        "yaxis2",
        "yaxis3",
    ):
        axis = getattr(fig.layout, axis_name, None)
        if axis is None:
            continue
        try:
            if axis.title and axis.title.font and axis.title.font.family:
                axis.title.font.family = _map_postscript_font_name(axis.title.font.family)
        except Exception:
            pass
        try:
            if axis.tickfont and axis.tickfont.family:
                axis.tickfont.family = _map_postscript_font_name(axis.tickfont.family)
        except Exception:
            pass

    return fig


def _resolve_export_dimension(
    requested: int,
    layout_value: Any,
    *,
    default: int,
) -> int:
    """Returns the export dimension, preferring explicit figure geometry.

    When callers leave width or height at the shared default, a figure level
    layout width or height should still be respected. This is useful for charts
    whose category counts differ and therefore need different heights to keep
    bar thickness and spacing visually consistent.

    Args:
        requested: Requested export dimension from the save helper.
        layout_value: Dimension already set on the figure layout.
        default: Shared default dimension used by the save helper.

    Returns:
        The resolved integer dimension.
    """

    if layout_value is not None and requested == default:
        try:
            return int(layout_value)
        except (TypeError, ValueError):
            pass

    try:
        return int(requested)
    except (TypeError, ValueError):
        return int(default)


def _apply_export_geometry(fig: go.Figure, width: int, height: int) -> go.Figure:
    """Makes export dimensions explicit for all renderers.

    Args:
        fig: Figure to adjust.
        width: Target export width.
        height: Target export height.

    Returns:
        The updated figure.
    """

    fig.update_layout(autosize=False, width=width, height=height)
    return fig


def _prepare_html_figure(fig: go.Figure, filename: str, width: int, height: int) -> go.Figure:
    """Returns the figure variant used for HTML export."""

    html_fig = _clone_figure(fig)
    html_fig = _apply_figure_style(html_fig, filename)
    html_fig = _apply_export_geometry(html_fig, width=width, height=height)
    return html_fig


def _prepare_static_figure(fig: go.Figure, filename: str, width: int, height: int) -> go.Figure:
    """Returns the figure variant used for PNG, PDF, and EPS export."""

    static_fig = _clone_figure(fig)
    static_fig = _apply_figure_style(static_fig, filename)
    static_fig = _apply_export_geometry(static_fig, width=width, height=height)
    static_fig = _coerce_postscript_fonts(static_fig)
    return static_fig


def _write_html(fig: go.Figure, path: Path, auto_open_html: bool) -> None:
    """Writes an HTML figure file.

    Args:
        fig: Figure to export.
        path: Destination HTML path.
        auto_open_html: Whether to open the HTML file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), auto_open=False, include_plotlyjs="cdn", full_html=True)
    maybe_open_html(path, auto_open_html)


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
    """Saves a Plotly figure in one or more output formats.

    Args:
        fig: Figure to export.
        filename: Base filename without extension.
        output_dir: Primary output directory.
        final_dir: Optional secondary directory for copied final outputs.
        auto_open_html: Whether to open the exported HTML file automatically.
        width: Export width in pixels for static formats.
        height: Export height in pixels for static formats.
        scale: Plotly export scale factor.
        save_final: Whether to copy generated files into ``final_dir``.
        save_png: Whether to export PNG output.
        save_eps: Whether to export EPS output.
        export_timeout_seconds: Timeout for static image export steps.

    Returns:
        A manifest mapping output types to file paths for successfully exported
        files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if final_dir is not None:
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

    resolved_width = _resolve_export_dimension(
        width,
        getattr(fig.layout, "width", None),
        default=1600,
    )
    resolved_height = _resolve_export_dimension(
        height,
        getattr(fig.layout, "height", None),
        default=900,
    )

    html_fig = _prepare_html_figure(
        fig,
        filename,
        width=resolved_width,
        height=resolved_height,
    )
    static_fig = _prepare_static_figure(
        fig,
        filename,
        width=resolved_width,
        height=resolved_height,
    )

    manifest: dict[str, str] = {}

    html_path = output_dir / f"{filename}.html"
    _write_html(html_fig, html_path, auto_open_html=auto_open_html)
    manifest["html"] = str(html_path)

    if save_final and final_dir is not None:
        final_html = final_dir / f"{filename}.html"
        _write_html(html_fig, final_html, auto_open_html=False)
        manifest["html_final"] = str(final_html)

    if save_png:
        png_path = output_dir / f"{filename}.png"
        ok_png, png_message = _save_png(
            static_fig,
            png_path,
            width=resolved_width,
            height=resolved_height,
            scale=scale,
            timeout_seconds=export_timeout_seconds,
        )
        if ok_png and png_path.exists():
            manifest["png"] = str(png_path)
            if save_final and final_dir is not None:
                final_png = final_dir / f"{filename}.png"
                _copy_if_exists(png_path, final_png)
                manifest["png_final"] = str(final_png)
        else:
            logger.error("PNG export failed for %s: %s", filename, png_message)

    if save_eps:
        eps_path = output_dir / f"{filename}.eps"
        ok_eps, eps_message = _save_eps(
            static_fig,
            eps_path,
            width=resolved_width,
            height=resolved_height,
            scale=scale,
            timeout_seconds=export_timeout_seconds,
        )
        if ok_eps and eps_path.exists():
            manifest["eps"] = str(eps_path)
            if save_final and final_dir is not None:
                final_eps = final_dir / f"{filename}.eps"
                _copy_if_exists(eps_path, final_eps)
                manifest["eps_final"] = str(final_eps)
        else:
            logger.error("EPS export failed for %s: %s", filename, eps_message)

    logger.info(
        "Exported plot %s | html=%s png=%s eps=%s",
        filename,
        "yes" if "html" in manifest else "no",
        "yes" if "png" in manifest else "no",
        "yes" if "eps" in manifest else "no",
    )
    return manifest
