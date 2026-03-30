from __future__ import annotations

import re
from typing import Any

import pandas as pd

from utils.normalise import clean_value, first_non_missing, first_token_csv_style, normalise_boolish, normalise_category

_LINE_FLAGS = re.IGNORECASE | re.MULTILINE


def _extract(pattern: str, text: str, flags: int = _LINE_FLAGS) -> str:
    match = re.search(pattern, text, flags)
    if not match:
        return 'NA'
    return clean_value(match.group(1))


def _extract_prefixed_line(text: str, prefix_pattern: str) -> str:
    match = re.search(rf'^{prefix_pattern}\s*(.*)$', text, _LINE_FLAGS)
    if not match:
        return ''
    return clean_value(match.group(1))


def _extract_kvs_from_line(text: str, prefix_pattern: str, keys: list[str]) -> dict[str, str]:
    line = _extract_prefixed_line(text, prefix_pattern)
    if not line:
        return {key: 'NA' for key in keys}

    escaped_keys = [re.escape(key) for key in keys]
    key_pattern = '|'.join(sorted(escaped_keys, key=len, reverse=True))
    pattern = re.compile(
        rf'(?P<key>{key_pattern})\s*=\s*(?P<value>.*?)(?=,\s*(?:{key_pattern})\s*=|$)',
        re.IGNORECASE,
    )
    found: dict[str, str] = {key: 'NA' for key in keys}
    canonical = {key.lower(): key for key in keys}
    for match in pattern.finditer(line):
        key = canonical[match.group('key').lower()]
        found[key] = clean_value(match.group('value'))
    return found


def _extract_single_value_from_line(text: str, prefix_pattern: str, key: str) -> str:
    return _extract_kvs_from_line(text, prefix_pattern, [key]).get(key, 'NA')


def parse_response_text(text: str) -> dict[str, str]:
    text = str(text or '')
    parsed: dict[str, str] = {}

    parsed['report_pdf'] = _extract(r'REPORT_PDF\s*=\s*([^\n]+)', text)
    parsed['av_guilty'] = normalise_boolish(_extract(r'Q0\.\s*AV_gui(?:l)?ty\s*=\s*([^\n]+)', text))
    parsed['q0_explanation'] = _extract(r'Q0\.\s*Explanation\s*=\s*([^\n]+)', text)
    parsed['main_factor'] = _extract(r'Q0\s*:\s*Factors\s*=\s*([^\n]+)', text)
    parsed['q0_confidence'] = _extract(r'Q0\.\s*confidence\s*=\s*([^\n]+)', text)

    parsed.update(_extract_kvs_from_line(text, r'Q1\.', ['av_manufacturer', 'av_make', 'av_year', 'av_model', 'vehicle_was']))
    parsed.update(_extract_kvs_from_line(text, r'Q2\.', ['accident_year', 'accident_month', 'accident_day', 'time']))
    parsed.update(_extract_kvs_from_line(text, r'Q3\.', ['zipcode', 'county', 'city', 'address']))
    parsed.update(_extract_kvs_from_line(text, r'Q4\.', ['Lane number', 'Street type']))
    parsed['lane_number'] = parsed.pop('Lane number')
    parsed['street_type'] = parsed.pop('Street type')
    parsed['speed_limit'] = _extract_single_value_from_line(text, r'Q5\.', 'Speed')
    parsed['street_busy'] = _extract_single_value_from_line(text, r'Q6\.', 'street_busy')
    parsed['damage'] = _extract_single_value_from_line(text, r'Q7\.', 'Damage')
    parsed['damaged_area'] = _extract_single_value_from_line(text, r'Q8\.', 'Damaged_area')

    parsed.update(_extract_kvs_from_line(text, r'Q9\.', ['v2_id', 'v2_year', 'v2_model', 'v2_state', 'v2_mov']))
    if parsed['v2_id'] == 'NA':
        parsed['v2_id'] = _extract_single_value_from_line(text, r'Q10\.', 'v2_id')

    parsed['v1_injury'] = _extract_single_value_from_line(text, r'Q11\.', 'v1_injury')
    parsed['v2_injury'] = _extract_single_value_from_line(text, r'Q12\.', 'v2_injury')
    parsed['v1_av'] = normalise_boolish(_extract_single_value_from_line(text, r'Q13\.', 'v1_AV'))

    parsed.update(_extract_kvs_from_line(text, r'Q14\.', ['v1_lane', 'v1_intersection', 'v1_move', 'v1_speed']))
    parsed.update(_extract_kvs_from_line(text, r'Q15\.', ['v2_lane', 'v2_intersection', 'v2_move', 'v2_speed']))

    parsed['direction'] = _extract_single_value_from_line(text, r'Q16\.', 'Direction')
    parsed.update(_extract_kvs_from_line(text, r'Q17\.', ['v1_demage', 'v2_demage']))
    parsed['v1_damage_desc'] = parsed.pop('v1_demage')
    parsed['v2_damage_desc'] = parsed.pop('v2_demage')

    parsed.update(_extract_kvs_from_line(text, r'Q18[:\.]', ['weather_v1', 'weather_v2']))
    parsed.update(_extract_kvs_from_line(text, r'Q19[:\.]', ['light_v1', 'light_v2']))
    parsed.update(_extract_kvs_from_line(text, r'Q20[:\.]', ['surface_v1', 'surface_v2']))
    parsed.update(_extract_kvs_from_line(text, r'Q21[:\.]', ['condition_v1', 'condition_v2']))
    parsed.update(_extract_kvs_from_line(text, r'Q22[:\.]', ['move_v1', 'move_v2']))
    parsed.update(_extract_kvs_from_line(text, r'Q23[:\.]', ['collision_v1', 'collision_v2']))
    parsed['other_factor'] = _extract_single_value_from_line(text, r'Q24[:\.]', 'other_factor')

    parsed['collision_type'] = first_non_missing(
        first_token_csv_style(parsed['collision_v1']),
        first_token_csv_style(parsed['collision_v2']),
    )

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
        parsed['source_report'] = clean_value(row.get('Report', parsed.get('report_pdf', 'NA')))
        parsed['raw_text_column'] = text_column
        records.append(parsed)

    parsed_df = pd.DataFrame.from_records(records)
    parsed_df.attrs.update(df.attrs)
    return parsed_df
