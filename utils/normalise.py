from __future__ import annotations

import math
import re
from typing import Any

MISSING_TOKENS = {'', 'na', 'n/a', 'nan', 'none', 'null', 'unknown', 'not specified'}


def strip_invisibles(text: str) -> str:
    text = text.replace('\ufeff', ' ')
    text = text.replace('\ufffc', ' ')
    text = text.replace('￼', ' ')
    text = text.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def clean_value(value: Any) -> str:
    if value is None:
        return 'NA'
    if isinstance(value, float) and math.isnan(value):
        return 'NA'
    text = strip_invisibles(str(value)).strip(' .;,:')
    return text or 'NA'


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = clean_value(value).lower()
    return text in MISSING_TOKENS


def normalise_boolish(value: Any) -> str:
    if is_missing(value):
        return 'NA'
    text = clean_value(value).lower()
    if text in {'true', 'yes', '1'}:
        return 'True'
    if text in {'false', 'no', '0'}:
        return 'False'
    return clean_value(value)


def first_non_missing(*values: Any) -> str:
    for value in values:
        if not is_missing(value):
            return clean_value(value)
    return 'NA'


def normalise_category(value: Any) -> str:
    text = clean_value(value)
    if is_missing(text):
        return 'NA'
    text = re.sub(r'\s+', ' ', text)
    return text


def first_token_csv_style(value: Any) -> str:
    text = normalise_category(value)
    if text == 'NA':
        return 'NA'
    parts = [part.strip() for part in re.split(r'[,/|]', text) if part.strip()]
    return parts[0] if parts else text
