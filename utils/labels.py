from __future__ import annotations

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
    text = text.replace('_', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def _pretty_token(token: str) -> str:
    if not token:
        return token
    lower = token.lower()
    if lower in _TOKEN_OVERRIDES:
        return _TOKEN_OVERRIDES[lower]
    if token.isupper():
        return token
    return token.capitalize()


def humanize_text(value: Any) -> str:
    if value is None:
        return 'NA'
    text = _normalise_spaces(str(value))
    if not text:
        return 'NA'

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
    return pretty.strip()


def humanize_field_name(field_name: str) -> str:
    return _FIELD_LABEL_OVERRIDES.get(field_name, humanize_text(field_name))
