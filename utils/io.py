from __future__ import annotations

import json
import logging
import subprocess
import sys
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
    paper: Path
    figures: Path
    figures_histograms: Path
    figures_paper: Path


def ensure_output_dirs(base_dir: Path, figures_dir: Path) -> OutputDirs:
    base = Path(base_dir)
    plots = base / 'plots'
    histograms = plots / 'histograms'
    paper = plots / 'paper'
    figures = Path(figures_dir)
    figures_histograms = figures / 'histograms'
    figures_paper = figures / 'paper'

    for path in [base, plots, histograms, paper, figures, figures_histograms, figures_paper]:
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


def _score_text(text: object) -> int:
    if is_missing(text):
        return -1
    value = str(text)
    keys = ['AV_gu', 'Explanation=', 'Factors=', 'v2_id=', 'collision_v', 'move_v', 'weather_v', 'light_v']
    return sum(value.count(key) for key in keys) + min(len(value) // 200, 20)


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
    df = df.reset_index(drop=True)
    df['row_id'] = df.index.astype(int)
    available_text_cols = [col for col in ['Output', 'Output - same chat'] if col in df.columns]
    if not available_text_cols:
        raise KeyError('Input CSV must contain Output and/or Output - same chat columns.')

    total_rows = len(df)
    if row_keep_policy in {'best_available', 'best_per_row'}:
        keep_mask = pd.Series(False, index=df.index)
        for col in available_text_cols:
            keep_mask = keep_mask | ~df[col].apply(is_missing)
    else:
        if 'Output' not in df.columns:
            raise KeyError('row_keep_policy="output_only" requires an Output column.')
        keep_mask = ~df['Output'].apply(is_missing)

    dropped = total_rows - int(keep_mask.sum())
    if dropped:
        logger.info('Dropped %s rows because the selected row_keep_policy=%s marked them as empty.', dropped, row_keep_policy)
    df = df.loc[keep_mask].copy()
    df.attrs['dropped_empty_output'] = dropped
    df.attrs['row_keep_policy'] = row_keep_policy

    if row_keep_policy == 'best_per_row':
        score_map = {col: df[col].map(_score_text) for col in available_text_cols}
        score_df = pd.DataFrame(score_map)
        chosen = score_df.idxmax(axis=1)
        chosen_score = score_df.max(axis=1)
        df['selected_text_column'] = chosen
        df['selected_text_score'] = chosen_score
        df['model_output_text'] = [df.at[idx, col] for idx, col in chosen.items()]
        logger.info('Per row text selection summary: %s', dict(chosen.value_counts()))
        return df.reset_index(drop=True), 'model_output_text'

    text_column = _choose_text_column(df, preferred_text_column, row_keep_policy=row_keep_policy)
    if text_column == 'Output - same chat' and 'Output' in df.columns:
        if row_keep_policy == 'best_available':
            df[text_column] = df[text_column].fillna(df['Output'])
        else:
            df[text_column] = df[text_column].fillna('')
    df['selected_text_column'] = text_column
    df['selected_text_score'] = df[text_column].map(_score_text)
    return df.reset_index(drop=True), text_column


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def save_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def maybe_open_html(path: Path, auto_open_html: bool) -> None:
    if not auto_open_html:
        return
    try:
        uri = path.resolve().as_uri()
        if sys.platform == 'darwin':
            subprocess.Popen(['open', uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import webbrowser
            webbrowser.open(uri)
    except Exception as exc:
        logger.warning('Failed to open %s in browser: %s', path, exc)
