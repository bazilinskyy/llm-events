from __future__ import annotations

"""Helpers for converting internal field names into presentation labels.

This module centralises two related transformations used across plots and
reports:

* converting raw field names such as ``scenario_class`` into readable labels
* converting raw values such as ``other_road_user`` into presentation text

The functions here keep formatting consistent across figures, tables, and
Markdown outputs.
"""

import re
from typing import Any

_FIELD_LABEL_OVERRIDES = {
    'scenario_class': 'Scenario class',
    'road_user_type': 'Road user type',
    'blame_group': 'Blame group',
    'collision_group': 'Collision group',
    'main_factor_grouped': 'Main factor',
    'report_completeness_score': 'Report completeness score',
    'coarse_context_score': 'Coarse context score',
    'fine_context_score': 'Fine context score',
    'report_explicitness_score': 'Report explicitness score',
    'movement_consistency_overall': 'Movement consistency status',
    'scenario_determinability_group': 'Scenario determinability',
    'environment_friction_profile': 'Environment profile',
    'blame_confidence_alignment': 'Blame confidence alignment',
    'stopped_av_subtype': 'Stopped AV subtype',
    'intersection_detail_quality': 'Intersection detail quality',
    'missing_rate': 'Missing rate',
    'mean_score': 'Mean score',
    'availability_rate': 'Availability rate',
    'count': 'Count',
    'share': 'Share',
    'field': 'Field',
    'provenance': 'Field provenance',
    'context_type': 'Metric',
    'who_group': 'Who',
    'where_group': 'Where',
    'what_group': 'What',
    'when_group': 'When',
    'why_group': 'Why',
    'how_group': 'How',
}

_TOKEN_OVERRIDES = {
    'av': 'AV',
    'v1': 'V1',
    'v2': 'V2',
    'q0': 'Q0',
    'q1': 'Q1',
    'q2': 'Q2',
    'q3': 'Q3',
    'q4': 'Q4',
    'q5': 'Q5',
    'q6': 'Q6',
    'q7': 'Q7',
    'q8': 'Q8',
    'q9': 'Q9',
    'q10': 'Q10',
    'q11': 'Q11',
    'q12': 'Q12',
    'q13': 'Q13',
    'q14': 'Q14',
    'q15': 'Q15',
    'q16': 'Q16',
    'q17': 'Q17',
    'q18': 'Q18',
    'q19': 'Q19',
    'q20': 'Q20',
    'q21': 'Q21',
    'q22': 'Q22',
    'q23': 'Q23',
    'q24': 'Q24',
    'na': 'NA',
}


def _normalise_spaces(text: str) -> str:
    """Normalises spacing and underscores in display text.

    Args:
        text: Raw text that may contain underscores or repeated whitespace.

    Returns:
        A cleaned string with underscores replaced by spaces and internal
        whitespace collapsed.
    """

    text = text.replace('_', ' ')
    return re.sub(r'\s+', ' ', text).strip()



def _pretty_token(token: str) -> str:
    """Formats a single token for human readable display.

    This helper preserves known abbreviations such as ``AV`` and ``Q1`` while
    capitalising ordinary tokens.

    Args:
        token: A single token extracted from a larger text value.

    Returns:
        A human readable token.
    """

    if not token:
        return token

    lower = token.lower()
    if lower in _TOKEN_OVERRIDES:
        return _TOKEN_OVERRIDES[lower]

    if token.isupper():
        return token

    return token.capitalize()



def _sentence_case_preserve_abbreviations(text: str) -> str:
    """Converts formatted text to sentence case while preserving abbreviations.

    Args:
        text: Already formatted display text.

    Returns:
        Sentence case text that keeps tokens such as ``AV``, ``V1``, ``Q0``,
        and ``NA`` in uppercase.
    """

    if not text:
        return text

    words = text.split()
    if not words:
        return text

    formatted_words: list[str] = []
    for index, word in enumerate(words):
        lower = word.lower()

        if lower in _TOKEN_OVERRIDES:
            formatted_words.append(_TOKEN_OVERRIDES[lower])
            continue

        if word.isupper():
            formatted_words.append(word)
            continue

        if index == 0:
            formatted_words.append(word[:1].upper() + word[1:].lower())
        else:
            formatted_words.append(word.lower())

    return ' '.join(formatted_words)



def humanize_text(value: Any) -> str:
    """Converts a raw value into presentation friendly text.

    The function normalises spacing, preserves separators such as ``-`` and
    ``/``, applies token level formatting, converts the result to sentence
    case, and converts empty values to ``NA``.

    Args:
        value: Raw value to format.

    Returns:
        A human readable string suitable for display in charts and tables.
    """

    if value is None:
        return 'NA'

    text = _normalise_spaces(str(value))
    if not text:
        return 'NA'

    # Split while preserving separators so compound labels remain readable.
    parts = re.split(r'([\-\/])', text)
    pretty_parts: list[str] = []

    for part in parts:
        if part in {'-', '/'}:
            pretty_parts.append(part)
            continue

        words = part.split()
        pretty_words = [_pretty_token(word) for word in words]
        pretty_parts.append(' '.join(pretty_words))

    pretty = ''.join(pretty_parts)
    pretty = pretty.replace(' - ', ' – ')
    pretty = pretty.strip()
    return _sentence_case_preserve_abbreviations(pretty)



def humanize_field_name(field_name: str) -> str:
    """Returns a display label for an internal field name.

    Known field names use explicit overrides so chart labels stay stable.
    Unknown field names fall back to :func:`humanize_text`.

    Args:
        field_name: Internal field name such as ``scenario_class``.

    Returns:
        A readable field label for display.
    """

    return _FIELD_LABEL_OVERRIDES.get(field_name, humanize_text(field_name))
