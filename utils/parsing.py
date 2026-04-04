from __future__ import annotations

"""Parsing helpers for extracting structured fields from model output text.

This module converts semi structured key value text into a normalised pandas
dataframe. It defines the canonical field names used across the project,
extracts alias based matches from raw output text, and enriches parsed rows
with metadata required by downstream analysis.
"""

import re
from typing import Any

import pandas as pd

from utils.normalise import (
    clean_value,
    first_non_missing,
    first_token_csv_style,
    normalise_boolish,
    normalise_category,
)

# Mapping from canonical field names to the raw labels that may appear in the
# model output text.
FIELD_ALIASES: dict[str, list[str]] = {
    'report_pdf': ['REPORT_PDF'],
    'av_guilty': ['AV_guity', 'AV_guilty'],
    'q0_explanation': ['Explanation'],
    'main_factor': ['Factors'],
    'q0_confidence': ['confidence'],
    'av_manufacturer': ['av_manufacturer'],
    'av_make': ['av_make'],
    'av_year': ['av_year'],
    'av_model': ['av_model'],
    'vehicle_was': ['vehicle_was'],
    'accident_year': ['accident_year'],
    'accident_month': ['accident_month'],
    'accident_day': ['accident_day'],
    'time': ['time'],
    'zipcode': ['zipcode'],
    'county': ['county'],
    'city': ['city'],
    'address': ['address'],
    'lane_number': ['Lane number'],
    'street_type': ['Street type'],
    'speed_limit': ['Speed'],
    'street_busy': ['street_busy'],
    'damage': ['Damage'],
    'damaged_area': ['Damaged_area'],
    'v2_id': ['v2_id'],
    'v2_year': ['v2_year'],
    'v2_model': ['v2_model'],
    'v2_state': ['v2_state'],
    'v2_mov': ['v2_mov'],
    'v1_injury': ['v1_injury'],
    'v2_injury': ['v2_injury'],
    'v1_av': ['v1_AV'],
    'v1_lane': ['v1_lane'],
    'v1_intersection': ['v1_intersection'],
    'v1_move': ['v1_move'],
    'v1_speed': ['v1_speed'],
    'v2_lane': ['v2_lane'],
    'v2_intersection': ['v2_intersection'],
    'v2_move': ['v2_move'],
    'v2_speed': ['v2_speed'],
    'direction': ['Direction'],
    'v1_damage_desc': ['v1_demage', 'v1_damage'],
    'v2_damage_desc': ['v2_demage', 'v2_damage'],
    'weather_v1': ['weather_v1'],
    'weather_v2': ['weather_v2'],
    'light_v1': ['light_v1'],
    'light_v2': ['light_v2'],
    'surface_v1': ['surface_v1'],
    'surface_v2': ['surface_v2'],
    'condition_v1': ['condition_v1'],
    'condition_v2': ['condition_v2'],
    'move_v1': ['move_v1'],
    'move_v2': ['move_v2'],
    'collision_v1': ['collision_v1'],
    'collision_v2': ['collision_v2'],
    'other_factor': ['other_factor'],
}

# Build a single regex that prefers longer labels first so overlapping labels
# are parsed correctly.
ALL_LABELS = [alias for aliases in FIELD_ALIASES.values() for alias in aliases]
ALL_LABELS_SORTED = sorted(ALL_LABELS, key=len, reverse=True)
LABEL_PATTERN = '|'.join(re.escape(label) for label in ALL_LABELS_SORTED)

# Match "key=value" segments while stopping at the next recognised label.
KV_PATTERN = re.compile(
    rf'(?P<key>{LABEL_PATTERN})\s*=\s*(?P<value>.*?)(?=,\s*(?:{LABEL_PATTERN})\s*=|$)',
    re.IGNORECASE,
)


def _extract_line_kvs(text: str) -> dict[str, list[str]]:
    """Extracts all key value pairs from raw multi line response text.

    Each line is scanned independently. Recognised keys are stored in lowercase
    and may accumulate multiple matched values.

    Args:
        text: Raw response text containing inline ``key=value`` pairs.

    Returns:
        A mapping from lowercased raw keys to lists of extracted values.
    """

    values: dict[str, list[str]] = {}

    for raw_line in str(text or '').splitlines():
        line = clean_value(raw_line)
        if '=' not in line:
            continue

        for match in KV_PATTERN.finditer(line):
            key = clean_value(match.group('key'))
            value = clean_value(match.group('value'))
            values.setdefault(key.lower(), []).append(value)

    return values


def _extract_first(kvs: dict[str, list[str]], aliases: list[str]) -> str:
    """Returns the first non missing value for a list of aliases.

    Args:
        kvs: Extracted key value mapping from the raw response text.
        aliases: Candidate alias labels to search in priority order.

    Returns:
        The first non missing matched value, or ``'NA'`` when none is found.
    """

    for alias in aliases:
        matches = kvs.get(alias.lower(), [])
        for match in matches:
            if clean_value(match) != 'NA':
                return clean_value(match)

    return 'NA'


def parse_response_text(text: str) -> dict[str, str]:
    """Parses a single response text blob into canonical fields.

    The function first extracts alias based key value pairs, then maps them
    into canonical field names, derives a few summary parse metrics, and
    finally normalises all values for downstream use.

    Args:
        text: Raw model output text.

    Returns:
        A dictionary containing canonical parsed fields and parse metadata.
    """

    text = str(text or '')
    kvs = _extract_line_kvs(text)
    parsed: dict[str, str] = {}

    for canonical_name, aliases in FIELD_ALIASES.items():
        parsed[canonical_name] = _extract_first(kvs, aliases)

    # Apply field specific normalisation before deriving helper columns.
    parsed['av_guilty'] = normalise_boolish(parsed['av_guilty'])
    parsed['v1_av'] = normalise_boolish(parsed['v1_av'])

    # Derive a unified collision type from the first available collision field.
    parsed['collision_type'] = first_non_missing(
        first_token_csv_style(parsed['collision_v1']),
        first_token_csv_style(parsed['collision_v2']),
    )

    hit_count = sum(1 for value in parsed.values() if clean_value(value) != 'NA')
    total_fields = len(FIELD_ALIASES)

    parsed['parse_key_hits'] = str(hit_count)
    parsed['parse_key_total'] = str(total_fields)
    parsed['parse_coverage'] = f"{hit_count / max(total_fields, 1):.3f}"

    # Run a final normalisation pass so all stored values are clean and
    # consistently formatted.
    for key, value in list(parsed.items()):
        if key in {'av_guilty', 'v1_av'}:
            parsed[key] = normalise_boolish(value)
        else:
            parsed[key] = normalise_category(value)

    return parsed


def parse_events_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Parses a dataframe of raw event outputs into canonical structured rows.

    In addition to the parsed fields, the returned dataframe preserves useful
    metadata from the source dataframe such as row identifiers, source report
    names, text selection details, and the raw output strings.

    Args:
        df: Input dataframe containing raw model output text.
        text_column: Column name containing the text to parse.

    Returns:
        A dataframe of parsed records with source metadata attached. Any
        dataframe attributes present on the input are copied to the output.
    """

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        parsed = parse_response_text(row[text_column])

        parsed['row_id'] = row.get('row_id', 'NA')
        parsed['source_report'] = clean_value(
            row.get('Report', parsed.get('report_pdf', 'NA'))
        )
        parsed['raw_text_column'] = text_column
        parsed['selected_text_column'] = clean_value(
            row.get('selected_text_column', text_column)
        )
        parsed['selected_text_score'] = clean_value(
            row.get('selected_text_score', 'NA')
        )
        parsed['model_output_text'] = clean_value(row.get(text_column, 'NA'))
        parsed['raw_output_text'] = clean_value(row.get('Output', 'NA'))
        parsed['output_text_score'] = clean_value(
            row.get('output_text_score', row.get('selected_text_score', 'NA'))
        )

        records.append(parsed)

    parsed_df = pd.DataFrame.from_records(records)
    parsed_df.attrs.update(df.attrs)
    return parsed_df
