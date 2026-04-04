from __future__ import annotations

"""Normalisation helpers for parsed values and lightweight categorisation.

This module standardises raw string like values extracted from reports and
model outputs. It provides utilities for:

* cleaning whitespace and invisible characters
* detecting missing values
* normalising common categories such as road user, mode, and movement
* bucketing numeric scores
* extracting numeric percentages from free text
* safely converting dictionary values to integers
"""

import math
import re
from typing import Any

# Canonical tokens treated as missing after values have been cleaned.
MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "not specified",
}


def strip_invisibles(text: str) -> str:
    """Removes invisible characters and collapses repeated whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text with invisible characters replaced and whitespace
        normalised.
    """

    text = text.replace("\ufeff", " ")
    text = text.replace("\ufffc", " ")
    text = text.replace("￼", " ")
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_value(value: Any) -> str:
    """Cleans an arbitrary value and converts missing values to ``"NA"``.

    Args:
        value: Raw input value.

    Returns:
        A cleaned string representation, or ``"NA"`` when the value is missing.
    """

    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"

    text = strip_invisibles(str(value)).strip(" .;,:")
    return text or "NA"


def is_missing(value: Any) -> bool:
    """Returns whether a value should be treated as missing.

    Args:
        value: Raw input value.

    Returns:
        ``True`` when the value is considered missing, otherwise ``False``.
    """

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True

    text = clean_value(value).lower()
    return text in MISSING_TOKENS


def normalise_boolish(value: Any) -> str:
    """Normalises truthy and falsy values into canonical string labels.

    Args:
        value: Raw input value.

    Returns:
        ``"True"``, ``"False"``, ``"NA"``, or the cleaned original value when
        it does not match a known boolean like token.
    """

    if is_missing(value):
        return "NA"

    text = clean_value(value).lower()
    if text in {"true", "yes", "1"}:
        return "True"
    if text in {"false", "no", "0"}:
        return "False"
    return clean_value(value)


def first_non_missing(*values: Any) -> str:
    """Returns the first non missing value from a sequence of candidates.

    Args:
        *values: Candidate values in priority order.

    Returns:
        The first cleaned non missing value, or ``"NA"`` when all values are
        missing.
    """

    for value in values:
        if not is_missing(value):
            return clean_value(value)
    return "NA"


def normalise_category(value: Any) -> str:
    """Normalises a free text category value.

    Args:
        value: Raw input value.

    Returns:
        A cleaned category string with collapsed whitespace, or ``"NA"`` when
        missing.
    """

    text = clean_value(value)
    if is_missing(text):
        return "NA"
    return re.sub(r"\s+", " ", text)


def first_token_csv_style(value: Any) -> str:
    """Returns the first token from a CSV style multi value field.

    The value is split on commas, slashes, and pipes after standard category
    normalisation.

    Args:
        value: Raw input value.

    Returns:
        The first non empty token, or ``"NA"`` when the value is missing.
    """

    text = normalise_category(value)
    if text == "NA":
        return "NA"

    parts = [part.strip() for part in re.split(r"[,/|]", text) if part.strip()]
    return parts[0] if parts else text


def normalise_road_user(value: Any) -> str:
    """Maps raw road user text into a smaller canonical label set.

    Args:
        value: Raw road user value.

    Returns:
        A normalised road user category.
    """

    text = normalise_category(value).lower()
    if text == "na":
        return "unknown"
    if "pedestrian" in text:
        return "pedestrian"
    if "bicycl" in text or "cyclist" in text or text == "bike":
        return "cyclist"
    if "scooter" in text:
        return "scooter"
    if "motorcycle" in text or "motorbike" in text:
        return "motorcycle"
    if "truck" in text:
        return "truck"
    if "bus" in text:
        return "bus"
    if "vehicle" in text or "car" in text or text in {"sedan", "suv"}:
        return "vehicle"
    if "object" in text:
        return "object"
    return text


def normalise_mode(value: Any) -> str:
    """Normalises autonomous mode flags into canonical mode labels.

    Args:
        value: Raw autonomous mode value.

    Returns:
        ``"autonomous"``, ``"conventional"``, or ``"unknown"``.
    """

    text = normalise_boolish(value)
    if text == "True":
        return "autonomous"
    if text == "False":
        return "conventional"
    return "unknown"


def normalise_movement(value: Any) -> str:
    """Maps movement descriptions into a reduced canonical movement set.

    Args:
        value: Raw movement value.

    Returns:
        A normalised movement category.
    """

    text = normalise_category(value).lower()
    mapping = {
        "stop": "stop",
        "stopped": "stop",
        "stopping": "stop",
        "slowing": "stop",
        "slowing/stopping": "stop",
        "straight": "straight",
        "proceeding straight": "straight",
        "turn_left": "turn_left",
        "making left turn": "turn_left",
        "left turn": "turn_left",
        "turn_right": "turn_right",
        "making right turn": "turn_right",
        "right turn": "turn_right",
        "turn_other": "turn_other",
        "other unsafe turning": "turn_other",
        "turn_u": "turn_u",
        "u turn": "turn_u",
        "change_lane": "change_lane",
        "changing lanes": "change_lane",
        "merging": "merging",
        "merge": "merging",
        "parked": "parked",
        "parking": "parked",
        "backing": "backing",
        "entering traffic": "entering_traffic",
        "entering_traffic": "entering_traffic",
        "passing other vehicle": "passing",
        "passing": "passing",
        "wrong_way": "wrong_way",
        "traveling wrong way": "wrong_way",
    }
    return mapping.get(text, text)


def normalise_collision(value: Any) -> str:
    """Maps collision descriptions into a reduced canonical collision set.

    Args:
        value: Raw collision value.

    Returns:
        A normalised collision category.
    """

    text = normalise_category(value).lower()
    mapping = {
        "rear": "rear_end",
        "rear end": "rear_end",
        "rear_end": "rear_end",
        "side": "side_swipe",
        "side swipe": "side_swipe",
        "side_swipe": "side_swipe",
        "broad": "broadside",
        "broadside": "broadside",
        "head": "head_on",
        "head on": "head_on",
        "head_on": "head_on",
        "object": "object",
        "pedestrian": "pedestrian",
        "other": "other",
    }
    return mapping.get(text, text)


def normalise_factor(value: Any) -> str:
    """Collapses broad causal factor text into canonical groups.

    Args:
        value: Raw factor value.

    Returns:
        A normalised factor category.
    """

    text = normalise_category(value).lower()
    if text == "na":
        return "unknown"
    if "weather" in text or "road condition" in text or "poor road" in text:
        return "weather_or_road"
    if "v1" in text or "av" in text or "mode failure" in text:
        return "v1_or_av"
    if "v2" in text or "other driver" in text or "other party" in text:
        return "v2_or_other_road_user"
    return text


def bucket_completeness(value: Any) -> str:
    """Buckets a numeric completeness score into low, medium, or high.

    Args:
        value: Numeric like value.

    Returns:
        ``"low"``, ``"medium"``, ``"high"``, or ``"unknown"`` when parsing
        fails.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if score < 0.33:
        return "low"
    if score < 0.67:
        return "medium"
    return "high"


def bucket_score(value: Any, low: float = 0.33, high: float = 0.67) -> str:
    """Buckets a numeric score using configurable low and high thresholds.

    Args:
        value: Numeric like value.
        low: Threshold below which the score is labelled low.
        high: Threshold below which the score is labelled medium.

    Returns:
        ``"low"``, ``"medium"``, ``"high"``, or ``"unknown"`` when parsing
        fails.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"


def extract_percentage_number(value: Any) -> float | None:
    """Extracts the first numeric value embedded in free text.

    Args:
        value: Raw input value.

    Returns:
        The extracted floating point number, or ``None`` when no numeric token
        is present.
    """

    if is_missing(value):
        return None

    text = clean_value(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def contains_any(value: Any, needles: list[str]) -> bool:
    """Returns whether the normalised text contains any requested substring.

    Args:
        value: Raw input value.
        needles: Candidate substrings to search for.

    Returns:
        ``True`` when at least one needle is found, otherwise ``False``.
    """

    text = normalise_category(value).lower()
    if text == "na":
        return False
    return any(needle.lower() in text for needle in needles)


def safe_int_dict(values: dict[Any, Any]) -> dict[str, int]:
    """Converts dictionary values to integers when possible.

    Keys are converted to strings. Entries that cannot be coerced to integers
    are skipped.

    Args:
        values: Input mapping with arbitrary keys and values.

    Returns:
        A dictionary containing only successfully converted integer values.
    """

    result: dict[str, int] = {}
    for key, value in values.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result
