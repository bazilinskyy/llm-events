from __future__ import annotations

import json
import logging
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.normalise import is_missing

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OutputDirs:
    base: Path
    plots: Path
    histograms: Path
    figures: Path
    figures_histograms: Path


def ensure_output_dirs(base_dir: Path, figures_dir: Path) -> OutputDirs:
    base = Path(base_dir)
    plots = base / 'plots'
    histograms = plots / 'histograms'
    figures = Path(figures_dir)
    figures_histograms = figures / 'histograms'

    histograms.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    figures_histograms.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        base=base,
        plots=plots,
        histograms=histograms,
        figures=figures,
        figures_histograms=figures_histograms,
    )


def _choose_text_column(df: pd.DataFrame, preferred: str | None, row_keep_policy: str = 'output_only') -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    if row_keep_policy == 'output_only':
        candidates.extend(['Output', 'Output - same chat'])
    else:
        candidates.extend(['Output - same chat', 'Output'])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f'Could not find a usable text column. Available columns: {list(df.columns)}')


def load_input_events(
    input_csv: Path,
    preferred_text_column: str | None = None,
    row_keep_policy: str = 'output_only',
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_csv)
    available_text_cols = [col for col in ['Output', 'Output - same chat'] if col in df.columns]
    if not available_text_cols:
        raise KeyError('Input CSV must contain Output and/or Output - same chat columns.')

    total_rows = len(df)
    if row_keep_policy == 'best_available':
        keep_mask = pd.Series(False, index=df.index)
        for col in available_text_cols:
            keep_mask = keep_mask | ~df[col].apply(is_missing)
    else:
        if 'Output' not in df.columns:
            raise KeyError('row_keep_policy="output_only" requires an Output column.')
        keep_mask = ~df['Output'].apply(is_missing)

    dropped = total_rows - int(keep_mask.sum())
    if dropped:
        logger.info(
            'Dropped %s rows because the selected row_keep_policy=%s marked them as empty.',
            dropped,
            row_keep_policy,
        )
    df = df.loc[keep_mask].copy()
    df.attrs['dropped_empty_output'] = dropped
    df.attrs['row_keep_policy'] = row_keep_policy

    text_column = _choose_text_column(df, preferred_text_column, row_keep_policy=row_keep_policy)
    if text_column == 'Output - same chat' and 'Output' in df.columns:
        if row_keep_policy == 'best_available':
            df[text_column] = df[text_column].fillna(df['Output'])
        else:
            df[text_column] = df[text_column].fillna('')
    return df.reset_index(drop=True), text_column


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def maybe_open_html(path: Path, auto_open_html: bool) -> None:
    if auto_open_html:
        try:
            webbrowser.open(path.resolve().as_uri())
        except Exception as exc:
            logger.warning('Failed to open %s in browser: %s', path, exc)
