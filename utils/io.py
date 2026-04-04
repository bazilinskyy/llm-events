from __future__ import annotations

"""Utilities for loading event data and writing analysis outputs.

This module provides small I O helpers for the analysis pipeline. It handles:

* creating output directories
* loading the input CSV and choosing the best text column
* scoring candidate model output text
* saving CSV, JSON, and Markdown outputs
* optionally opening generated HTML files in a browser
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from utils.normalise import is_missing

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OutputDirs:
    """Container for resolved output directory paths.

    Attributes:
        base: Root output directory for generated artefacts.
        plots: Directory used for plot outputs.
        histograms: Directory used for histogram outputs.
        paper: Directory used for paper style figure outputs.
        figures: Final figure export directory.
        figures_histograms: Final histogram export directory.
        figures_paper: Final paper figure export directory.
    """

    base: Path
    plots: Path
    histograms: Path
    paper: Path
    figures: Path
    figures_histograms: Path
    figures_paper: Path


def ensure_output_dirs(base_dir: Path, figures_dir: Path) -> OutputDirs:
    """Creates and returns the output directory structure.

    The current project layout writes all generated figures directly into the
    configured output directory. When final figure export is enabled elsewhere
    in the pipeline, the same files may also be copied into ``figures_dir``.

    Args:
        base_dir: Base directory for generated outputs.
        figures_dir: Directory for final figure exports.

    Returns:
        An ``OutputDirs`` instance containing the resolved directory paths.
    """

    base = Path(base_dir)
    figures = Path(figures_dir)

    # Save all figures directly into the configured output directory.
    # When save_final is true, the same files are also copied directly into
    # the configured figures_dir.
    plots = base
    histograms = base
    paper = base
    figures_histograms = figures
    figures_paper = figures

    for path in [base, figures]:
        path.mkdir(parents=True, exist_ok=True)

    return OutputDirs(
        base=base,
        plots=plots,
        histograms=histograms,
        paper=paper,
        figures=figures,
        figures_histograms=figures_histograms,
        figures_paper=figures_paper,
    )


def _score_text(text: Any) -> int:
    """Scores a text blob by how much structured signal it appears to contain.

    The score is a lightweight heuristic used when choosing the best model
    output column. It rewards the presence of expected structured keys and
    gives a small bonus to longer texts.

    Args:
        text: Raw text candidate from the input dataframe.

    Returns:
        A non negative score for valid text, or ``-1`` when the value is
        missing.
    """

    if is_missing(text):
        return -1

    value = str(text)
    keys = [
        "AV_gu",
        "Explanation=",
        "Factors=",
        "v2_id=",
        "collision_v",
        "move_v",
        "weather_v",
        "light_v",
    ]

    # Count the number of expected structured markers present in the text.
    key_score = sum(value.count(key) for key in keys)

    # Add a small length bonus while capping the effect of very long strings.
    length_bonus = min(len(value) // 200, 20)

    return key_score + length_bonus


def _choose_text_column(
    df: pd.DataFrame,
    preferred: str | None,
    row_keep_policy: str = "output_only",
) -> str:
    """Chooses the text column to use for downstream parsing.

    The pipeline now reads only the main ``Output`` column. Any preferred
    column setting that points elsewhere is ignored so that downstream parsing
    always remains anchored to the primary response export.

    Args:
        df: Input dataframe containing raw model output text.
        preferred: Preferred column name, if supplied by configuration.
        row_keep_policy: Retained for backwards compatibility with existing
            configuration, but no longer changes column selection.

    Returns:
        The selected column name, always ``"Output"``.

    Raises:
        KeyError: If the ``Output`` column is missing.
    """

    if "Output" in df.columns:
        return "Output"

    raise KeyError(
        f'Could not find the required Output column. Available columns: {list(df.columns)}'
    )


def load_input_events(
    input_csv: Path,
    preferred_text_column: str | None = None,
    row_keep_policy: str = "output_only",
) -> tuple[pd.DataFrame, str]:
    """Loads the input CSV and prepares the text selected for parsing.

    The pipeline now reads only the main ``Output`` column. Rows with missing
    ``Output`` are dropped at load time, and any auxiliary same chat column is
    ignored entirely even when it is present in the CSV.

    Args:
        input_csv: Path to the input CSV file.
        preferred_text_column: Optional preferred text column name. Kept for
            backwards compatibility but ignored unless it is ``"Output"``.
        row_keep_policy: Kept for backwards compatibility with existing
            configuration. It no longer changes how rows are retained.

    Returns:
        A tuple of:
            * the prepared dataframe
            * the name of the text column to parse

    Raises:
        KeyError: If the required ``Output`` column is missing.
    """

    df = pd.read_csv(input_csv)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    if "Output" not in df.columns:
        raise KeyError('Input CSV must contain an Output column.')

    total_rows = len(df)
    keep_mask = ~df["Output"].apply(is_missing)

    dropped = total_rows - int(keep_mask.sum())
    if dropped:
        logger.info(
            "Dropped %s rows because the Output column was empty.",
            dropped,
        )

    df = df.loc[keep_mask].copy()
    df.attrs["dropped_empty_output"] = dropped
    df.attrs["row_keep_policy"] = row_keep_policy

    text_column = _choose_text_column(
        df,
        preferred_text_column,
        row_keep_policy=row_keep_policy,
    )

    df["selected_text_column"] = text_column
    df["selected_text_score"] = df[text_column].map(_score_text)

    logger.info(
        "Text selection summary: %s",
        {"Output": int(len(df))},
    )

    return df.reset_index(drop=True), text_column


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Saves a dataframe as CSV, creating parent directories as needed.

    Args:
        df: Dataframe to write.
        path: Destination CSV path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    """Saves a dictionary payload as UTF 8 encoded JSON.

    Args:
        payload: JSON serialisable dictionary to write.
        path: Destination JSON path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_markdown(text: str, path: Path) -> None:
    """Saves Markdown text to disk.

    Args:
        text: Markdown content to write.
        path: Destination Markdown path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def maybe_open_html(path: Path, auto_open_html: bool) -> None:
    """Opens an HTML file in the default browser when enabled.

    On macOS this uses ``open``. On Linux it uses ``xdg-open``. On other
    platforms it falls back to the standard ``webbrowser`` module.

    Args:
        path: HTML file path to open.
        auto_open_html: Whether automatic opening is enabled.
    """

    if not auto_open_html:
        return

    try:
        uri = path.resolve().as_uri()

        if sys.platform == "darwin":
            subprocess.Popen(
                ["open", uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform.startswith("linux"):
            subprocess.Popen(
                ["xdg-open", uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            import webbrowser

            webbrowser.open(uri)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to open %s in browser: %s", path, exc)
