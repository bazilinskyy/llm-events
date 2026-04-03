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
    return re.sub(r'\s+', ' ', text)


def first_token_csv_style(value: Any) -> str:
    text = normalise_category(value)
    if text == 'NA':
        return 'NA'
    parts = [part.strip() for part in re.split(r'[,/|]', text) if part.strip()]
    return parts[0] if parts else text


def normalise_road_user(value: Any) -> str:
    text = normalise_category(value).lower()
    if text == 'na':
        return 'unknown'
    if 'pedestrian' in text:
        return 'pedestrian'
    if 'bicycl' in text or 'cyclist' in text or text == 'bike':
        return 'cyclist'
    if 'scooter' in text:
        return 'scooter'
    if 'motorcycle' in text or 'motorbike' in text:
        return 'motorcycle'
    if 'truck' in text:
        return 'truck'
    if 'bus' in text:
        return 'bus'
    if 'vehicle' in text or 'car' in text or text in {'sedan', 'suv'}:
        return 'vehicle'
    if 'object' in text:
        return 'object'
    return text


def normalise_mode(value: Any) -> str:
    text = normalise_boolish(value)
    if text == 'True':
        return 'autonomous'
    if text == 'False':
        return 'conventional'
    return 'unknown'


def normalise_movement(value: Any) -> str:
    text = normalise_category(value).lower()
    mapping = {
        'stop': 'stop', 'stopped': 'stop', 'stopping': 'stop', 'slowing': 'stop', 'slowing/stopping': 'stop',
        'straight': 'straight', 'proceeding straight': 'straight',
        'turn_left': 'turn_left', 'making left turn': 'turn_left', 'left turn': 'turn_left',
        'turn_right': 'turn_right', 'making right turn': 'turn_right', 'right turn': 'turn_right',
        'turn_other': 'turn_other', 'other unsafe turning': 'turn_other',
        'turn_u': 'turn_u', 'u turn': 'turn_u',
        'change_lane': 'change_lane', 'changing lanes': 'change_lane',
        'merging': 'merging', 'merge': 'merging',
        'parked': 'parked', 'parking': 'parked',
        'backing': 'backing',
        'entering traffic': 'entering_traffic', 'entering_traffic': 'entering_traffic',
        'passing other vehicle': 'passing', 'passing': 'passing',
        'wrong_way': 'wrong_way', 'traveling wrong way': 'wrong_way',
    }
    return mapping.get(text, text)


def normalise_collision(value: Any) -> str:
    text = normalise_category(value).lower()
    mapping = {
        'rear': 'rear_end',
        'rear end': 'rear_end',
        'rear_end': 'rear_end',
        'side': 'side_swipe',
        'side swipe': 'side_swipe',
        'side_swipe': 'side_swipe',
        'broad': 'broadside',
        'broadside': 'broadside',
        'head': 'head_on',
        'head on': 'head_on',
        'head_on': 'head_on',
        'object': 'object',
        'pedestrian': 'pedestrian',
        'other': 'other',
    }
    return mapping.get(text, text)


def normalise_factor(value: Any) -> str:
    text = normalise_category(value).lower()
    if text == 'na':
        return 'unknown'
    if 'weather' in text or 'road condition' in text or 'poor road' in text:
        return 'weather_or_road'
    if 'v1' in text or 'av' in text or 'mode failure' in text:
        return 'v1_or_av'
    if 'v2' in text or 'other driver' in text or 'other party' in text:
        return 'v2_or_other_road_user'
    return text


def bucket_completeness(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    if score < 0.33:
        return 'low'
    if score < 0.67:
        return 'medium'
    return 'high'


def bucket_score(value: Any, low: float = 0.33, high: float = 0.67) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    if score < low:
        return 'low'
    if score < high:
        return 'medium'
    return 'high'


def extract_percentage_number(value: Any) -> float | None:
    if is_missing(value):
        return None
    text = clean_value(value)
    match = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def contains_any(value: Any, needles: list[str]) -> bool:
    text = normalise_category(value).lower()
    if text == 'na':
        return False
    return any(needle.lower() in text for needle in needles)


def safe_int_dict(values: dict[Any, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in values.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result
