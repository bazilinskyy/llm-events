from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import plotly as py
import plotly.graph_objects as go

from utils.io import maybe_open_html

logger = logging.getLogger(__name__)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def _try_direct_eps(fig: go.Figure, eps_path: Path, width: int, height: int) -> bool:
    try:
        fig.write_image(str(eps_path), format='eps', width=width, height=height)
        return eps_path.exists()
    except Exception:
        return False


def _try_svg_to_eps(fig: go.Figure, eps_path: Path, width: int, height: int) -> bool:
    try:
        import cairosvg  # type: ignore
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / 'temp.svg'
            fig.write_image(str(svg_path), format='svg', width=width, height=height)
            cairosvg.svg2ps(url=str(svg_path), write_to=str(eps_path))
        return eps_path.exists()
    except Exception:
        return False


def _try_pdf_to_eps(fig: go.Figure, eps_path: Path, width: int, height: int, scale: int) -> bool:
    pdftops = shutil.which('pdftops')
    if not pdftops:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / 'temp.pdf'
            fig.write_image(str(pdf_path), format='pdf', width=width, height=height, scale=scale)
            subprocess.run([pdftops, '-eps', str(pdf_path), str(eps_path)], check=True, capture_output=True, text=True)
        return eps_path.exists()
    except Exception:
        return False


def _save_eps(fig: go.Figure, eps_path: Path, width: int, height: int, scale: int) -> bool:
    return (
        _try_direct_eps(fig, eps_path, width=width, height=height)
        or _try_svg_to_eps(fig, eps_path, width=width, height=height)
        or _try_pdf_to_eps(fig, eps_path, width=width, height=height, scale=scale)
    )


def save_plotly_figure(
    fig: go.Figure,
    filename: str,
    output_dir: Path,
    final_dir: Path | None = None,
    auto_open_html: bool = True,
    width: int = 1600,
    height: int = 900,
    scale: int = 1,
    save_final: bool = True,
    save_png: bool = True,
    save_eps: bool = True,
) -> dict[str, str]:
    """
    Saves a Plotly figure as HTML, PNG, and EPS formats.

    Args:
        fig: Plotly figure object.
        filename: Name of the file without extension.
        output_dir: Directory for regular outputs.
        final_dir: Directory for final figure copies.
        auto_open_html: Whether to open the HTML file in a browser.
        width: Width of the PNG and EPS images in pixels.
        height: Height of the PNG and EPS images in pixels.
        scale: Scaling factor for the PNG image.
        save_final: Whether to save the final figure copy.
        save_png: Whether to save PNG output.
        save_eps: Whether to save EPS output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if final_dir is not None:
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}

    html_path = output_dir / f'{filename}.html'
    logger.info('Saving html file for %s.', filename)
    py.offline.plot(fig, filename=str(html_path), auto_open=False)
    manifest['html'] = str(html_path)
    maybe_open_html(html_path, auto_open_html)
    if save_final and final_dir is not None:
        final_html = final_dir / f'{filename}.html'
        py.offline.plot(fig, filename=str(final_html), auto_open=False)
        manifest['html_final'] = str(final_html)

    if save_png:
        try:
            png_path = output_dir / f'{filename}.png'
            logger.info('Saving png file for %s.', filename)
            fig.write_image(str(png_path), width=width, height=height, scale=scale)
            manifest['png'] = str(png_path)
            if save_final and final_dir is not None:
                final_png = final_dir / f'{filename}.png'
                _copy_if_exists(png_path, final_png)
                manifest['png_final'] = str(final_png)
        except ValueError as exc:
            logger.error('Value error raised when attempting to save PNG image %s: %s', filename, exc)
        except Exception as exc:
            logger.warning('Failed to save PNG file for %s: %s', filename, exc)

    if save_eps:
        eps_path = output_dir / f'{filename}.eps'
        logger.info('Saving eps file for %s.', filename)
        if _save_eps(fig, eps_path, width=width, height=height, scale=scale):
            manifest['eps'] = str(eps_path)
            if save_final and final_dir is not None:
                final_eps = final_dir / f'{filename}.eps'
                _copy_if_exists(eps_path, final_eps)
                manifest['eps_final'] = str(final_eps)
        else:
            logger.warning('EPS export skipped for %s. Install CairoSVG or pdftops if you need EPS.', filename)

    return manifest
