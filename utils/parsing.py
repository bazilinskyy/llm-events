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
    'av_company': ['av_company'],
    'v1_company': ['v1_company'],
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
    'v2_make': ['v2_make'],
    'v2_company': ['v2_company'],
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

# Locate every recognised ``key=`` marker. Values are recovered from the span
# between consecutive markers because many archived responses place the whole
# questionnaire on one line and separate questions with ``Q14.`` rather than
# a comma or newline.
KEY_START_PATTERN = re.compile(
    rf'(?<![A-Za-z0-9_])(?P<key>{LABEL_PATTERN})\s*=\s*',
    re.IGNORECASE,
)

QUESTION_SUFFIX_PATTERN = re.compile(
    r'(?:[,.;]\s*)?Q\s*\d+(?:\.\d+)?\s*[:.]?\s*$',
    re.IGNORECASE,
)
SOURCE_NOTE_PATTERN = re.compile(r'\s+Source\(s\)\s*:', re.IGNORECASE)
NEXT_QUESTION_ASSIGNMENT_PATTERN = re.compile(
    r'\s+Q\s*\d+(?:\.\d+)?\s*[:.]?\s*[A-Za-z][A-Za-z0-9_ ]*\s*=',
    re.IGNORECASE,
)

AMBIGUOUS_PLACEHOLDERS = {
    'true/false',
    'false/true',
    'yes/no',
    'no/yes',
}

_KNOWN_MAKE_PREFIXES = {
    'acura',
    'alfa romeo',
    'aston martin',
    'audi',
    'bentley',
    'bmw',
    'buick',
    'cadillac',
    'chevrolet',
    'chevy',
    'chrysler',
    'dodge',
    'fiat',
    'ford',
    'genesis',
    'gmc',
    'honda',
    'hyundai',
    'infiniti',
    'jaguar',
    'jeep',
    'kia',
    'land rover',
    'lexus',
    'lincoln',
    'lucid',
    'mazda',
    'mercedes',
    'mercedes benz',
    'mini',
    'mitsubishi',
    'nissan',
    'polestar',
    'porsche',
    'ram',
    'rivian',
    'smart',
    'subaru',
    'tesla',
    'toyota',
    'volkswagen',
    'volvo',
}

_MAKE_CANONICAL_OVERRIDES = {
    'chevy': 'Chevrolet',
    'mercedes': 'Mercedes',
    'mercedes benz': 'Mercedes Benz',
    'gmc': 'GMC',
    'bmw': 'BMW',
}

ONLINE_FIELD_PATTERNS = {
    'lane_number': re.compile(r'^\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE),
    'street_type': re.compile(
        r'^\s*(one[- ]way|two[- ]way|divided|undivided)\b',
        re.IGNORECASE,
    ),
    'speed_limit': re.compile(
        r'^\s*(\d+(?:\.\d+)?\s*(?:mph)?)\b',
        re.IGNORECASE,
    ),
    'street_busy': re.compile(
        r'^\s*(true|false|yes|no|high|low|above average|below average)\b',
        re.IGNORECASE,
    ),
}


def _extract_line_kvs(text: str) -> dict[str, list[str]]:
    """Extracts all key value pairs from compact or multi line response text.

    Recognised key markers are located across the complete response. This
    supports both comma separated fields and compact questionnaire text such
    as ``move_v1=stop, move_v2=straight Q23: collision_v1=rear``.

    Args:
        text: Raw response text containing inline ``key=value`` pairs.

    Returns:
        A mapping from lowercased raw keys to lists of extracted values.
    """

    source_text = str(text or '')
    matches = list(KEY_START_PATTERN.finditer(source_text))
    values: dict[str, list[str]] = {}

    for index, match in enumerate(matches):
        value_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(source_text)
        )
        raw_value = source_text[match.end():value_end]

        # Online enrichment fields sometimes append a provenance note before
        # the next questionnaire item. The note is not part of the field.
        source_note = SOURCE_NOTE_PATTERN.search(raw_value)
        if source_note:
            raw_value = raw_value[:source_note.start()]

        # Stop at an unrecognised assignment in the next numbered question.
        # This protects the current field when a response uses a label variant
        # that is not part of the canonical alias map.
        next_question = NEXT_QUESTION_ASSIGNMENT_PATTERN.search(raw_value)
        if next_question:
            raw_value = raw_value[:next_question.start()]

        # Remove the question marker belonging to the next key and the common
        # comma delimiter between fields. Preserve punctuation inside the
        # actual value.
        raw_value = QUESTION_SUFFIX_PATTERN.sub('', raw_value)
        value = clean_value(raw_value.rstrip(' ,'))
        key = clean_value(match.group('key'))
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
            cleaned = clean_value(match)
            if (
                cleaned != 'NA'
                and cleaned.lower() not in AMBIGUOUS_PLACEHOLDERS
            ):
                return cleaned

    return 'NA'


def _title_case_token(text: str) -> str:
    """Converts a token sequence into a readable title case form."""

    cleaned = normalise_category(text)
    if cleaned in {'NA', 'unknown'}:
        return 'NA'

    lower = cleaned.lower()
    if lower in _MAKE_CANONICAL_OVERRIDES:
        return _MAKE_CANONICAL_OVERRIDES[lower]

    return ' '.join(part.capitalize() for part in lower.split())


def _normalise_online_lookup_value(field: str, value: Any) -> str:
    """Keeps only the requested value from an online lookup response.

    The model sometimes appended a source note or explanatory sentence to an
    ``NA`` result. Treating the complete sentence as a field value would turn
    an unsuccessful lookup into apparent availability.
    """

    cleaned = clean_value(value)
    if cleaned.lower().startswith(('na.', 'n/a.', 'none.', 'unknown.')):
        return 'NA'
    if normalise_category(cleaned) in {'NA', 'unknown'}:
        return 'NA'

    pattern = ONLINE_FIELD_PATTERNS[field]
    match = pattern.match(cleaned)
    if not match:
        return 'NA'
    return clean_value(match.group(1))


def _infer_make_from_model_text(model_text: Any) -> str:
    """Infers a likely vehicle make from a free text model field.

    This is useful because some generated answers collapse make and model into
    one field such as ``Nissan Leaf`` or ``2016 Nissan Leaf``.

    Args:
        model_text: Raw model text.

    Returns:
        The inferred make when recognised, otherwise ``'NA'``.
    """

    text = normalise_category(model_text)
    if text in {'NA', 'unknown'}:
        return 'NA'

    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    if not cleaned:
        return 'NA'

    cleaned = re.sub(r'^\d{4}\s+', '', cleaned)

    for token_count in (3, 2, 1):
        prefix = ' '.join(cleaned.split()[:token_count]).strip()
        if prefix in _KNOWN_MAKE_PREFIXES:
            return _title_case_token(prefix)

    return 'NA'


def _strip_make_from_model_text(model_text: Any, make_text: Any) -> str:
    """Removes a recognised make prefix from a combined make and model string.

    Args:
        model_text: Raw model or make plus model text.
        make_text: The recognised make to strip.

    Returns:
        The cleaned model text, or the original normalised text when stripping
        is not possible.
    """

    model = normalise_category(model_text)
    make = normalise_category(make_text)
    if model in {'NA', 'unknown'}:
        return 'NA'
    if make in {'NA', 'unknown'}:
        return model

    cleaned_model = re.sub(r'\s+', ' ', model).strip()
    cleaned_model = re.sub(r'^\d{4}\s+', '', cleaned_model)

    pattern = re.compile(rf'^\s*{re.escape(make)}\b\s*', re.IGNORECASE)
    stripped = pattern.sub('', cleaned_model).strip(' ,-')
    return stripped or cleaned_model


def _derive_vehicle_company_fields(parsed: dict[str, str]) -> None:
    """Derives company and make helper fields for both involved vehicles.

    Args:
        parsed: Parsed field dictionary updated in place.
    """

    parsed['av_company'] = first_non_missing(
        parsed.get('av_company', 'NA'),
        parsed.get('av_make', 'NA'),
        parsed.get('av_manufacturer', 'NA'),
    )
    parsed['v1_company'] = first_non_missing(
        parsed.get('v1_company', 'NA'),
        parsed.get('av_make', 'NA'),
        parsed.get('av_manufacturer', 'NA'),
    )

    inferred_v2_make = first_non_missing(
        parsed.get('v2_make', 'NA'),
        parsed.get('v2_company', 'NA'),
        _infer_make_from_model_text(parsed.get('v2_model', 'NA')),
    )
    parsed['v2_make'] = inferred_v2_make
    parsed['v2_company'] = first_non_missing(
        parsed.get('v2_company', 'NA'),
        inferred_v2_make,
    )

    if inferred_v2_make not in {'NA', 'unknown'}:
        parsed['v2_model'] = first_non_missing(
            _strip_make_from_model_text(parsed.get('v2_model', 'NA'), inferred_v2_make),
            parsed.get('v2_model', 'NA'),
        )


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

    # Preserve an explicit ``None`` returned for an injury question. Generic
    # missing value handling otherwise converts this marker to ``NA`` and
    # makes it indistinguishable from a question that was not recovered.
    injury_none_markers = {
        field: clean_value(parsed.get(field)).lower() == 'none'
        for field in ['v1_injury', 'v2_injury']
    }

    _derive_vehicle_company_fields(parsed)

    # Apply field specific normalisation before deriving helper columns.
    parsed['av_guilty'] = normalise_boolish(parsed['av_guilty'])
    parsed['v1_av'] = normalise_boolish(parsed['v1_av'])

    for online_field in ONLINE_FIELD_PATTERNS:
        parsed[online_field] = _normalise_online_lookup_value(
            online_field,
            parsed.get(online_field),
        )

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
        if key in injury_none_markers and injury_none_markers[key]:
            parsed[key] = 'no_injury_marker'
        elif key in {'av_guilty', 'v1_av'}:
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
