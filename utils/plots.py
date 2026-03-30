from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import plotly.graph_objects as go

from utils.io import maybe_open_html

logger = logging.getLogger(__name__)


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _try_direct_eps(fig: go.Figure, eps_path: Path) -> bool:
    try:
        fig.write_image(str(eps_path), format='eps')
        return eps_path.exists()
    except Exception:
        return False


def _try_pdf_to_eps(pdf_path: Path, eps_path: Path) -> bool:
    pdftops = shutil.which('pdftops')
    if not pdftops:
        return False
    try:
        subprocess.run([pdftops, '-eps', str(pdf_path), str(eps_path)], check=True, capture_output=True, text=True)
        return eps_path.exists()
    except Exception:
        return False


def _try_svg_to_eps_with_inkscape(svg_path: Path, eps_path: Path) -> bool:
    inkscape = shutil.which('inkscape')
    if not inkscape:
        return False
    commands = [
        [inkscape, str(svg_path), '--export-type=eps', f'--export-filename={eps_path}'],
        [inkscape, str(svg_path), '--export-filename', str(eps_path)],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if eps_path.exists():
                return True
        except Exception:
            continue
    return False


def _try_svg_to_eps_with_cairosvg(svg_path: Path, eps_path: Path) -> bool:
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2ps(url=str(svg_path), write_to=str(eps_path))
        return eps_path.exists()
    except Exception:
        return False


def _write_eps_via_temp_files(fig: go.Figure, eps_path: Path) -> bool:
    if _try_direct_eps(fig, eps_path):
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        pdf_path = tmp / 'figure.pdf'
        try:
            fig.write_image(str(pdf_path), format='pdf')
            if _try_pdf_to_eps(pdf_path, eps_path):
                return True
        except Exception:
            pass

        svg_path = tmp / 'figure.svg'
        try:
            fig.write_image(str(svg_path), format='svg')
        except Exception:
            return False

        if _try_svg_to_eps_with_cairosvg(svg_path, eps_path):
            return True
        if _try_svg_to_eps_with_inkscape(svg_path, eps_path):
            return True

    return False


def export_plotly_figure(
    fig: go.Figure,
    stem: Path,
    auto_open_html: bool = True,
    save_final: bool = False,
    final_stem: Path | None = None,
) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    html_path = stem.with_suffix('.html')
    png_path = stem.with_suffix('.png')
    eps_path = stem.with_suffix('.eps')

    fig.write_html(str(html_path), include_plotlyjs='cdn')
    manifest['html'] = str(html_path)
    maybe_open_html(html_path, auto_open_html)

    try:
        fig.write_image(str(png_path), format='png')
        manifest['png'] = str(png_path)
    except Exception as exc:
        logger.warning('PNG export skipped for %s: %s', stem.name, exc)

    if _write_eps_via_temp_files(fig, eps_path):
        manifest['eps'] = str(eps_path)
    else:
        logger.warning('EPS export skipped for %s. Install CairoSVG or Poppler pdftops or Inkscape if you need EPS.', stem.name)

    if save_final and final_stem is not None:
        final_html = final_stem.with_suffix('.html')
        final_png = final_stem.with_suffix('.png')
        final_eps = final_stem.with_suffix('.eps')
        _copy_if_exists(html_path, final_html)
        _copy_if_exists(png_path, final_png)
        _copy_if_exists(eps_path, final_eps)
        manifest['final_html'] = str(final_html)
        if final_png.exists():
            manifest['final_png'] = str(final_png)
        if final_eps.exists():
            manifest['final_eps'] = str(final_eps)

    return manifest
