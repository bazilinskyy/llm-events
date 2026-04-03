from __future__ import annotations

import re
from typing import Any

import pandas as pd

from utils.normalise import clean_value, first_non_missing, first_token_csv_style, normalise_boolish, normalise_category

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

ALL_LABELS = [alias for aliases in FIELD_ALIASES.values() for alias in aliases]
ALL_LABELS_SORTED = sorted(ALL_LABELS, key=len, reverse=True)
LABEL_PATTERN = '|'.join(re.escape(label) for label in ALL_LABELS_SORTED)
KV_PATTERN = re.compile(
    rf'(?P<key>{LABEL_PATTERN})\s*=\s*(?P<value>.*?)(?=,\s*(?:{LABEL_PATTERN})\s*=|$)',
    re.IGNORECASE,
)


def _extract_line_kvs(text: str) -> dict[str, list[str]]:
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
    for alias in aliases:
        matches = kvs.get(alias.lower(), [])
        for match in matches:
            if clean_value(match) != 'NA':
                return clean_value(match)
    return 'NA'


def parse_response_text(text: str) -> dict[str, str]:
    text = str(text or '')
    kvs = _extract_line_kvs(text)
    parsed: dict[str, str] = {}

    for canonical_name, aliases in FIELD_ALIASES.items():
        parsed[canonical_name] = _extract_first(kvs, aliases)

    parsed['av_guilty'] = normalise_boolish(parsed['av_guilty'])
    parsed['v1_av'] = normalise_boolish(parsed['v1_av'])
    parsed['collision_type'] = first_non_missing(
        first_token_csv_style(parsed['collision_v1']),
        first_token_csv_style(parsed['collision_v2']),
    )
    parsed['parse_key_hits'] = str(sum(1 for value in parsed.values() if clean_value(value) != 'NA'))
    parsed['parse_key_total'] = str(len(FIELD_ALIASES))
    parsed['parse_coverage'] = f"{sum(1 for value in parsed.values() if clean_value(value) != 'NA') / max(len(FIELD_ALIASES), 1):.3f}"

    for key, value in list(parsed.items()):
        if key in {'av_guilty', 'v1_av'}:
            parsed[key] = normalise_boolish(value)
        else:
            parsed[key] = normalise_category(value)
    return parsed


def parse_events_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        parsed = parse_response_text(row[text_column])
        parsed['row_id'] = row.get('row_id', 'NA')
        parsed['source_report'] = clean_value(row.get('Report', parsed.get('report_pdf', 'NA')))
        parsed['raw_text_column'] = text_column
        parsed['selected_text_column'] = clean_value(row.get('selected_text_column', text_column))
        parsed['selected_text_score'] = clean_value(row.get('selected_text_score', 'NA'))
        parsed['model_output_text'] = clean_value(row.get(text_column, 'NA'))
        parsed['raw_output_text'] = clean_value(row.get('Output', 'NA'))
        parsed['raw_same_chat_text'] = clean_value(row.get('Output - same chat', 'NA'))
        parsed['output_text_score'] = clean_value(row.get('output_text_score', 'NA'))
        parsed['same_chat_text_score'] = clean_value(row.get('same_chat_text_score', 'NA'))
        records.append(parsed)

    parsed_df = pd.DataFrame.from_records(records)
    parsed_df.attrs.update(df.attrs)
    return parsed_df
