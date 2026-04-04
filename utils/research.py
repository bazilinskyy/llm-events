from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from utils.normalise import (
    bucket_completeness,
    bucket_score,
    contains_any,
    extract_percentage_number,
    first_non_missing,
    is_missing,
    normalise_boolish,
    normalise_category,
    normalise_collision,
    normalise_factor,
    normalise_mode,
    normalise_movement,
    normalise_road_user,
    safe_int_dict,
)
from utils.parsing import parse_events_dataframe

logger = logging.getLogger(__name__)

DEFAULT_COMPLETENESS_FIELDS = [
    'v2_id', 'v1_av', 'move_v1', 'move_v2', 'collision_type', 'av_guilty', 'main_factor',
    'weather_v1', 'light_v1', 'v1_lane', 'v2_lane', 'v1_speed', 'v2_speed', 'v1_intersection', 'v2_intersection',
]

FORM_FIELDS = [
    'av_manufacturer', 'av_make', 'av_year', 'av_model', 'vehicle_was',
    'accident_year', 'accident_month', 'accident_day', 'time', 'zipcode', 'county', 'city', 'address',
    'damage', 'damaged_area', 'v2_id', 'v2_year', 'v2_model', 'v2_state', 'v2_mov',
    'v1_injury', 'v2_injury', 'v1_av',
]

CHECKBOX_FIELDS = [
    'weather_v1', 'weather_v2', 'light_v1', 'light_v2', 'surface_v1', 'surface_v2',
    'condition_v1', 'condition_v2', 'move_v1', 'move_v2', 'collision_v1', 'collision_v2', 'other_factor',
]

NARRATIVE_FIELDS = [
    'av_guilty', 'q0_explanation', 'main_factor', 'q0_confidence',
    'v1_lane', 'v1_intersection', 'v1_move', 'v1_speed',
    'v2_lane', 'v2_intersection', 'v2_move', 'v2_speed',
    'direction', 'v1_damage_desc', 'v2_damage_desc',
]

ONLINE_FIELDS = ['lane_number', 'street_type', 'speed_limit', 'street_busy']

COARSE_CONTEXT_FIELDS = [
    'road_user_type', 'road_user_vulnerability_group', 'av_mode_group', 'av_movement_group',
    'other_party_movement_group', 'collision_group', 'blame_group', 'environment_friction_profile',
]

FINE_CONTEXT_FIELDS = [
    'v1_lane', 'v2_lane', 'v1_speed', 'v2_speed', 'v1_intersection', 'v2_intersection',
    'direction', 'lane_number', 'street_type', 'street_busy',
]

SCENARIO_EVIDENCE_FIELDS = [
    'road_user_type', 'av_movement_group', 'other_party_movement_group', 'collision_group', 'intersection_context',
]

FIELD_PROVENANCE_ORDER = ['bounded_form', 'bounded_checkbox', 'narrative', 'online_enriched']

PROVENANCE_GROUPS = {
    'bounded_form': FORM_FIELDS,
    'bounded_checkbox': CHECKBOX_FIELDS,
    'narrative': NARRATIVE_FIELDS,
    'online_enriched': ONLINE_FIELDS,
}

SOURCE_COMPARE_FIELDS = [
    'av_guilty', 'road_user_type', 'av_mode_group', 'av_movement_group', 'other_party_movement_group',
    'collision_group', 'blame_group', 'scenario_class', 'direction', 'intersection_context',
]


def _available_count(row: pd.Series, fields: list[str]) -> int:
    return sum(0 if is_missing(row.get(field)) else 1 for field in fields)


def _score_rate(row: pd.Series, fields: list[str]) -> float:
    if not fields:
        return 0.0
    return _available_count(row, fields) / len(fields)


def _normalise_intersection(v1_value: Any, v2_value: Any) -> str:
    v1 = normalise_boolish(v1_value)
    v2 = normalise_boolish(v2_value)
    if 'True' in {v1, v2}:
        return 'intersection'
    if v1 == 'False' or v2 == 'False':
        return 'non_intersection'
    return 'unknown'


def _derive_blame_group(av_guilty: Any, main_factor: Any) -> str:
    av = normalise_boolish(av_guilty)
    factor = normalise_factor(main_factor).lower()
    if av == 'True':
        return 'AV_primary'
    if av == 'False':
        if 'v2' in factor:
            return 'other_road_user'
        if 'weather' in factor or 'road' in factor:
            return 'environment_or_conditions'
        return 'other_road_user'
    if 'v1' in factor:
        return 'AV_primary'
    if 'v2' in factor:
        return 'other_road_user'
    if 'weather' in factor or 'road' in factor:
        return 'environment_or_conditions'
    return 'unclear'


def _derive_road_user_vulnerability_group(road_user_type: Any) -> str:
    value = normalise_category(road_user_type).lower()
    if value in {'pedestrian', 'cyclist', 'scooter', 'motorcycle'}:
        return 'vulnerable_road_user'
    if value in {'vehicle', 'truck', 'bus'}:
        return 'motor_vehicle'
    if value in {'object'}:
        return 'object'
    return 'unknown_or_other'


def _derive_harm_scope_group(v1_injury: Any, v2_injury: Any) -> str:
    combined = ' '.join([normalise_category(v1_injury).lower(), normalise_category(v2_injury).lower()])
    if 'deceased' in combined or 'injured' in combined or 'driver' in combined or 'passenger' in combined or 'bicyclist' in combined:
        return 'bodily_harm_or_fatality'
    if 'property' in combined:
        return 'property_damage_only'
    if 'none' in combined:
        return 'no_recorded_harm'
    return 'unknown'


def _derive_environment_friction_profile(weather: Any, light: Any, surface: Any, condition: Any) -> str:
    weather_text = normalise_category(weather).lower()
    light_text = normalise_category(light).lower()
    surface_text = normalise_category(surface).lower()
    condition_text = normalise_category(condition).lower()

    visibility_degraded = weather_text in {'raining', 'fog', 'wind', 'other'} or light_text in {'dawn', 'streetlight', 'no_streetlight', 'not_function'}
    surface_degraded = surface_text in {'wet', 'icy', 'slippery'}
    roadway_unusual = condition_text not in {'na', 'no', 'unknown'}

    if visibility_degraded and (surface_degraded or roadway_unusual):
        return 'compounded_degradation'
    if visibility_degraded:
        return 'visibility_degraded'
    if surface_degraded:
        return 'surface_degraded'
    if roadway_unusual:
        return 'roadway_unusual'
    return 'nominal'


def _derive_scenario_assignment(row: pd.Series) -> tuple[str, str, str]:
    av_move = row.get('av_movement_group', 'NA')
    other_move = row.get('other_party_movement_group', 'NA')
    collision = row.get('collision_group', 'NA')
    intersection = row.get('intersection_context', 'unknown')
    explanation = normalise_category(row.get('q0_explanation', '')).lower()
    v2_raw = normalise_category(row.get('v2_id', '')).lower()

    if row.get('road_user_vulnerability_group') == 'vulnerable_road_user':
        return 'vulnerable_road_user_interaction', 'vulnerable_road_user_trigger', 'road_user_vulnerability_group'
    if av_move == 'stop' and collision == 'rear_end':
        return 'AV_stopped_rear_end', 'av_stop_plus_rear_end', 'av_movement_group+collision_group'
    if intersection == 'intersection' and collision in {'broadside', 'side_swipe', 'head_on'}:
        return 'intersection_lateral_conflict', 'intersection_plus_lateral_collision', 'intersection_context+collision_group'
    if av_move == 'straight' and other_move in {'turn_left', 'turn_right', 'turn_other', 'turn_u'}:
        return 'turn_across_path_conflict', 'straight_vs_turning_conflict', 'av_movement_group+other_party_movement_group'
    if av_move in {'change_lane', 'merging'} or other_move in {'change_lane', 'merging'} or collision == 'side_swipe':
        return 'lane_change_or_merge_conflict', 'lane_change_merge_or_side_swipe', 'movement_or_collision_rule'
    if 'parked' in explanation or 'double-parked' in explanation or 'parked' in v2_raw or collision == 'object':
        return 'curbside_or_parked_vehicle_conflict', 'parked_vehicle_or_object_cue', 'q0_explanation_or_v2_id_or_collision_group'
    if av_move == 'stop' and ('uncertainty' in explanation or 'obstruction' in explanation or 'yield' in explanation):
        return 'low_speed_stop_or_obstruction_case', 'stop_for_uncertainty_or_obstruction', 'av_movement_group+q0_explanation'
    return 'other_or_ambiguous', 'no_rule_fired', 'insufficient_or_mixed_evidence'


def _consistency_status(value_a: Any, value_b: Any, normaliser) -> str:
    a = normaliser(value_a)
    b = normaliser(value_b)
    if a in {'NA', 'unknown'} and b in {'NA', 'unknown'}:
        return 'insufficient_both_missing'
    if a in {'NA', 'unknown'} or b in {'NA', 'unknown'}:
        return 'insufficient_one_missing'
    if a == b:
        return 'consistent'
    return 'inconsistent'


def _overall_movement_consistency(av_status: str, other_status: str) -> str:
    statuses = {av_status, other_status}
    if statuses == {'consistent'}:
        return 'consistent'
    if 'inconsistent' in statuses:
        return 'inconsistent'
    if 'consistent' in statuses:
        return 'partial_consistency'
    return 'insufficient'


def _derive_scenario_determinability_group(row: pd.Series) -> str:
    evidence_count = int(row.get('scenario_evidence_count', 0))
    scenario_class = normalise_category(row.get('scenario_class')).lower()
    if scenario_class != 'other_or_ambiguous' and evidence_count >= 4:
        return 'high'
    if evidence_count >= 3:
        return 'medium'
    return 'low'


def _derive_intersection_detail_quality(row: pd.Series) -> str:
    intersection = normalise_category(row.get('intersection_context')).lower()
    lanes_present = not is_missing(row.get('v1_lane')) or not is_missing(row.get('v2_lane'))
    direction_present = not is_missing(row.get('direction'))
    if intersection == 'intersection' and (lanes_present or direction_present):
        return 'contextualised_intersection'
    if intersection == 'intersection':
        return 'intersection_flag_only'
    if intersection == 'unknown' and row.get('collision_group') in {'broadside', 'head_on'}:
        return 'possible_intersection_underspecified'
    return 'not_intersection_focused'


def _derive_stopped_av_subtype(row: pd.Series) -> str:
    if row.get('av_movement_group') != 'stop':
        return 'NA'
    explanation = normalise_category(row.get('q0_explanation')).lower()
    vehicle_was = normalise_category(row.get('vehicle_was')).lower()
    if row.get('collision_group') == 'rear_end':
        return 'stopped_rear_end'
    if row.get('intersection_context') == 'intersection':
        return 'stopped_at_intersection'
    if contains_any(explanation, ['uncertainty', 'obstruction', 'yield', 'blocked']):
        return 'stop_for_obstruction_or_uncertainty'
    if 'stopped in traffic' in vehicle_was or 'traffic' in explanation:
        return 'traffic_stop'
    return 'other_stopped_case'


def _derive_blame_explicitness_group(row: pd.Series) -> str:
    av_present = not is_missing(row.get('av_guilty'))
    factor_present = not is_missing(row.get('main_factor'))
    explanation_present = not is_missing(row.get('q0_explanation'))
    if av_present and factor_present and explanation_present:
        return 'explicit'
    if (av_present and factor_present) or (factor_present and explanation_present):
        return 'partial'
    return 'weak'


def _derive_blame_conflict_reason(row: pd.Series) -> str:
    av = normalise_boolish(row.get('av_guilty'))
    factor = normalise_factor(row.get('main_factor')).lower()
    if av == 'True' and factor in {'v2_or_other_road_user', 'weather_or_road'}:
        return 'av_guilty_conflicts_with_factor'
    if av == 'False' and factor == 'v1_or_av':
        return 'non_av_guilty_conflicts_with_factor'
    return 'none'


def _derive_blame_confidence_alignment(row: pd.Series) -> str:
    confidence = row.get('q0_confidence_numeric')
    evidence = float(row.get('blame_evidence_score', 0.0))
    completeness = float(row.get('report_completeness_score', 0.0))
    if confidence is None:
        return 'insufficient_confidence_signal'
    if confidence >= 80 and (evidence < 0.75 or completeness < 0.5):
        return 'high_confidence_low_evidence'
    if confidence < 50 and evidence >= 0.75:
        return 'low_confidence_strong_evidence'
    return 'aligned_or_unclear'


def _derive_blame_evidence_strength(row: pd.Series) -> str:
    evidence = float(row.get('blame_evidence_score', 0.0))
    explicitness = row.get('blame_explicitness_group', 'weak')
    conflict = row.get('blame_conflict_reason', 'none') != 'none'
    if conflict:
        return 'conflicting'
    if explicitness == 'explicit' and evidence >= 1.0:
        return 'strong'
    if evidence >= 0.75:
        return 'moderate'
    if evidence >= 0.5:
        return 'weak'
    return 'very_weak'


def _derive_external_enrichment_group(row: pd.Series) -> str:
    count = int(row.get('online_field_count', 0))
    if count == 0:
        return 'none'
    if count <= 2:
        return 'partial'
    return 'rich'


def _derive_report_explicitness_score(row: pd.Series) -> float:
    structured_count = int(row.get('form_field_count', 0)) + int(row.get('checkbox_field_count', 0))
    narrative_count = int(row.get('narrative_field_count', 0))
    online_count = int(row.get('online_field_count', 0))
    numerator = structured_count + 0.5 * narrative_count + 0.25 * online_count
    denominator = len(FORM_FIELDS) + len(CHECKBOX_FIELDS) + 0.5 * len(NARRATIVE_FIELDS) + 0.25 * len(ONLINE_FIELDS)
    return numerator / denominator if denominator else 0.0


def _derive_initial_movement_inconsistency_diagnosis(row: pd.Series) -> str:
    overall = row.get('movement_consistency_overall', 'insufficient')
    if overall == 'consistent':
        return 'none'
    if overall in {'partial_consistency', 'insufficient'}:
        return 'missing_or_under_specified'
    if row.get('scenario_class') == 'other_or_ambiguous':
        return 'ambiguous_scenario_context'
    return 'within_report_conflict_or_parser_issue'


def derive_research_columns(parsed_df: pd.DataFrame, blind_spot_fields: list[str]) -> pd.DataFrame:
    df = parsed_df.copy()
    df.attrs.update(getattr(parsed_df, 'attrs', {}))

    df['road_user_type'] = df['v2_id'].map(normalise_road_user) if 'v2_id' in df.columns else 'NA'
    df['road_user_vulnerability_group'] = df['road_user_type'].map(_derive_road_user_vulnerability_group)
    df['av_mode_group'] = df['v1_av'].map(normalise_mode) if 'v1_av' in df.columns else 'unknown'
    df['av_movement_group'] = df.apply(lambda row: normalise_movement(first_non_missing(row.get('move_v1'), row.get('v1_move'))), axis=1)
    df['other_party_movement_group'] = df.apply(
        lambda row: normalise_movement(first_non_missing(row.get('move_v2'), row.get('v2_move'), row.get('v2_mov'))),
        axis=1,
    )
    df['collision_group'] = df['collision_type'].map(normalise_collision) if 'collision_type' in df.columns else 'NA'
    df['main_factor_grouped'] = df['main_factor'].map(normalise_factor) if 'main_factor' in df.columns else 'NA'
    df['blame_group'] = df.apply(lambda row: _derive_blame_group(row.get('av_guilty'), row.get('main_factor')), axis=1)
    df['intersection_context'] = df.apply(lambda row: _normalise_intersection(row.get('v1_intersection'), row.get('v2_intersection')), axis=1)
    df['environment_friction_profile'] = df.apply(
        lambda row: _derive_environment_friction_profile(row.get('weather_v1'), row.get('light_v1'), row.get('surface_v1'), row.get('condition_v1')),
        axis=1,
    )
    df['harm_scope_group'] = df.apply(lambda row: _derive_harm_scope_group(row.get('v1_injury'), row.get('v2_injury')), axis=1)

    completeness_fields = [field for field in DEFAULT_COMPLETENESS_FIELDS if field in df.columns]
    completeness = []
    for _, row in df.iterrows():
        non_missing = sum(0 if is_missing(row.get(field)) else 1 for field in completeness_fields)
        completeness.append(non_missing / max(len(completeness_fields), 1))
    df['report_completeness_score'] = completeness
    df['report_completeness_band'] = df['report_completeness_score'].map(bucket_completeness)

    for prefix, fields in [('form', FORM_FIELDS), ('checkbox', CHECKBOX_FIELDS), ('narrative', NARRATIVE_FIELDS), ('online', ONLINE_FIELDS)]:
        present_fields = [field for field in fields if field in df.columns]
        df[f'{prefix}_field_count'] = df.apply(lambda row: _available_count(row, present_fields), axis=1)
        df[f'{prefix}_field_rate'] = df.apply(lambda row: _score_rate(row, present_fields), axis=1)

    df['structured_field_count'] = df['form_field_count'] + df['checkbox_field_count']
    df['structured_field_rate'] = (
        df['structured_field_count'] /
        max(len([field for field in FORM_FIELDS if field in df.columns]) + len([field for field in CHECKBOX_FIELDS if field in df.columns]), 1)
    )
    df['report_explicitness_score'] = df.apply(_derive_report_explicitness_score, axis=1)
    df['report_explicitness_band'] = df['report_explicitness_score'].map(bucket_score)
    df['evidence_provenance_dominant'] = df[[
        'form_field_rate', 'checkbox_field_rate', 'narrative_field_rate', 'online_field_rate'
    ]].idxmax(axis=1).str.replace('_field_rate', '', regex=False)

    for field in blind_spot_fields:
        if field in df.columns:
            df[f'blind_spot__{field}'] = df[field].map(is_missing)

    df['coarse_context_score'] = df.apply(lambda row: _score_rate(row, [field for field in COARSE_CONTEXT_FIELDS if field in df.columns]), axis=1)
    df['fine_context_score'] = df.apply(lambda row: _score_rate(row, [field for field in FINE_CONTEXT_FIELDS if field in df.columns]), axis=1)
    df['context_granularity_gap'] = df['coarse_context_score'] - df['fine_context_score']
    df['context_granularity_gap_band'] = df['context_granularity_gap'].map(lambda value: 'large' if value >= 0.4 else 'moderate' if value >= 0.2 else 'small')

    scenario_assignment = df.apply(_derive_scenario_assignment, axis=1, result_type='expand')
    scenario_assignment.columns = ['scenario_class', 'scenario_rule_trigger', 'scenario_assignment_evidence']
    df = pd.concat([df, scenario_assignment], axis=1)
    df['scenario_evidence_count'] = df.apply(lambda row: _available_count(row, [field for field in SCENARIO_EVIDENCE_FIELDS if field in df.columns]), axis=1)
    df['scenario_determinability_group'] = df.apply(_derive_scenario_determinability_group, axis=1)

    df['av_move_consistency_status'] = df.apply(lambda row: _consistency_status(row.get('move_v1'), row.get('v1_move'), normalise_movement), axis=1)
    df['other_move_consistency_status'] = df.apply(
        lambda row: _consistency_status(row.get('move_v2'), first_non_missing(row.get('v2_move'), row.get('v2_mov')), normalise_movement),
        axis=1,
    )
    df['movement_consistency_overall'] = df.apply(
        lambda row: _overall_movement_consistency(row.get('av_move_consistency_status'), row.get('other_move_consistency_status')),
        axis=1,
    )
    df['movement_inconsistency_diagnosis'] = df.apply(_derive_initial_movement_inconsistency_diagnosis, axis=1)

    df['q0_confidence_numeric'] = df['q0_confidence'].map(extract_percentage_number) if 'q0_confidence' in df.columns else None
    df['q0_confidence_band'] = df['q0_confidence_numeric'].map(lambda value: 'unknown' if value is None else 'low' if value < 50 else 'medium' if value < 80 else 'high')
    df['blame_evidence_score'] = df.apply(lambda row: _score_rate(row, ['av_guilty', 'main_factor', 'q0_explanation', 'q0_confidence']), axis=1)
    df['blame_explicitness_group'] = df.apply(_derive_blame_explicitness_group, axis=1)
    df['blame_conflict_reason'] = df.apply(_derive_blame_conflict_reason, axis=1)
    df['blame_conflict_flag'] = df['blame_conflict_reason'].ne('none')
    df['blame_confidence_alignment'] = df.apply(_derive_blame_confidence_alignment, axis=1)
    df['blame_evidence_strength'] = df.apply(_derive_blame_evidence_strength, axis=1)

    df['intersection_detail_quality'] = df.apply(_derive_intersection_detail_quality, axis=1)
    df['stopped_av_subtype'] = df.apply(_derive_stopped_av_subtype, axis=1)
    df['external_enrichment_group'] = df.apply(_derive_external_enrichment_group, axis=1)

    return df


def _missingness_table(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for field in fields:
        if field not in df.columns:
            rows.append({'field': field, 'missing_count': total, 'missing_rate': 1.0, 'available_count': 0, 'available_rate': 0.0})
            continue
        missing_mask = df[field].map(is_missing)
        missing_count = int(missing_mask.sum())
        rows.append({
            'field': field,
            'missing_count': missing_count,
            'missing_rate': missing_count / total,
            'available_count': int((~missing_mask).sum()),
            'available_rate': float((~missing_mask).sum()) / total,
        })
    return pd.DataFrame(rows).sort_values(['missing_rate', 'field'], ascending=[False, True]).reset_index(drop=True)


def _top_counts(df: pd.DataFrame, field: str, top_n: int) -> pd.DataFrame:
    counts = df[field].astype(str).value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = [field, 'count']
    counts['share'] = counts['count'] / max(len(df), 1)
    return counts


def _field_provenance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for provenance, fields in PROVENANCE_GROUPS.items():
        for field in fields:
            available = 0 if field not in df.columns else int((~df[field].map(is_missing)).sum())
            rows.append({
                'field': field,
                'provenance': provenance,
                'availability_count': available,
                'availability_rate': available / total,
            })
    table = pd.DataFrame(rows)
    return table.sort_values(['provenance', 'availability_rate', 'field'], ascending=[True, False, True]).reset_index(drop=True)


def _provenance_summary_table(field_provenance: pd.DataFrame) -> pd.DataFrame:
    summary = (
        field_provenance.groupby('provenance')
        .agg(field_count=('field', 'count'), mean_availability_rate=('availability_rate', 'mean'))
        .reset_index()
    )
    summary['provenance'] = pd.Categorical(summary['provenance'], categories=FIELD_PROVENANCE_ORDER, ordered=True)
    return summary.sort_values('provenance').reset_index(drop=True)


def _score_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        'report_completeness_score', 'report_explicitness_score', 'coarse_context_score',
        'fine_context_score', 'context_granularity_gap', 'blame_evidence_score',
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors='coerce')
        rows.append({
            'metric': metric,
            'mean': round(float(series.mean()), 3) if not series.empty else 0.0,
            'median': round(float(series.median()), 3) if not series.empty else 0.0,
            'min': round(float(series.min()), 3) if not series.empty else 0.0,
            'max': round(float(series.max()), 3) if not series.empty else 0.0,
        })
    return pd.DataFrame(rows)


def _distribution_table(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return _top_counts(df, field, top_n=max(df[field].astype(str).nunique(), 1)) if field in df.columns else pd.DataFrame(columns=[field, 'count', 'share'])


def _data_availability_table(research_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    attrs = getattr(research_df, 'attrs', {})
    total_rows = int(attrs.get('total_rows_original', len(research_df) + attrs.get('dropped_empty_output', 0)))
    presence = attrs.get('output_presence_summary', {})
    rows = [
        {'stage': 'total_rows_in_csv', 'count': total_rows, 'share_of_total': 1.0 if total_rows else 0.0},
        {'stage': 'both_output_columns_empty', 'count': int(presence.get('both_empty', 0)), 'share_of_total': int(presence.get('both_empty', 0)) / total_rows if total_rows else 0.0},
        {'stage': 'rows_with_any_model_output', 'count': int(presence.get('rows_with_any_model_output', len(research_df))), 'share_of_total': int(presence.get('rows_with_any_model_output', len(research_df))) / total_rows if total_rows else 0.0},
        {'stage': 'rows_parsed', 'count': int(len(research_df)), 'share_of_total': len(research_df) / total_rows if total_rows else 0.0},
        {'stage': 'rows_in_empirical_subset', 'count': int(len(filtered_df)), 'share_of_total': len(filtered_df) / total_rows if total_rows else 0.0},
        {'stage': 'rows_excluded_after_parsing', 'count': int(len(research_df) - len(filtered_df)), 'share_of_total': (len(research_df) - len(filtered_df)) / total_rows if total_rows else 0.0},
    ]
    return pd.DataFrame(rows)


def _retained_vs_dropped_comparison(research_df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
    included_ids = set(filtered_df['row_id'].tolist()) if 'row_id' in filtered_df.columns else set()
    groups = {
        'empirical_included': research_df[research_df['row_id'].isin(included_ids)].copy(),
        'parsed_but_excluded': research_df[~research_df['row_id'].isin(included_ids)].copy(),
    }
    rows: list[dict[str, Any]] = []
    for group_name, df in groups.items():
        if df.empty:
            rows.append({'group': group_name, 'row_count': 0})
            continue
        rows.append({
            'group': group_name,
            'row_count': int(len(df)),
            'mean_parse_coverage': round(float(pd.to_numeric(df['parse_coverage'], errors='coerce').mean()), 3),
            'mean_selected_text_score': round(float(pd.to_numeric(df['selected_text_score'], errors='coerce').mean()), 3),
            'mean_completeness_score': round(float(pd.to_numeric(df['report_completeness_score'], errors='coerce').mean()), 3),
            'mean_explicitness_score': round(float(pd.to_numeric(df['report_explicitness_score'], errors='coerce').mean()), 3),
            'mean_coarse_context_score': round(float(pd.to_numeric(df['coarse_context_score'], errors='coerce').mean()), 3),
            'mean_fine_context_score': round(float(pd.to_numeric(df['fine_context_score'], errors='coerce').mean()), 3),
            'mean_context_gap': round(float(pd.to_numeric(df['context_granularity_gap'], errors='coerce').mean()), 3),
            'mean_blame_evidence_score': round(float(pd.to_numeric(df['blame_evidence_score'], errors='coerce').mean()), 3),
            'share_output_selected': round(float((df['selected_text_column'] == 'Output').mean()), 3),
            'share_same_chat_selected': round(float((df['selected_text_column'] == 'Output - same chat').mean()), 3),
        })
    return pd.DataFrame(rows)


def _compare_field(field: str, left: Any, right: Any) -> bool:
    if field == 'av_guilty':
        return normalise_boolish(left) == normalise_boolish(right)
    return normalise_category(left) == normalise_category(right)


def _build_source_disagreement_tables(research_df: pd.DataFrame, blind_spot_fields: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'raw_output_text' not in research_df.columns or 'raw_same_chat_text' not in research_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    both = research_df.loc[(~research_df['raw_output_text'].map(is_missing)) & (~research_df['raw_same_chat_text'].map(is_missing))].copy()
    if both.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw_min = pd.DataFrame({
        'row_id': both['row_id'],
        'Report': both['source_report'],
        'Output': both['raw_output_text'],
        'Output - same chat': both['raw_same_chat_text'],
    })
    parsed_output = derive_research_columns(parse_events_dataframe(raw_min, text_column='Output'), blind_spot_fields=blind_spot_fields)
    parsed_same = derive_research_columns(parse_events_dataframe(raw_min, text_column='Output - same chat'), blind_spot_fields=blind_spot_fields)

    left = parsed_output[['row_id', 'source_report', 'model_output_text'] + SOURCE_COMPARE_FIELDS].copy().rename(columns={c: f'{c}__output' for c in ['model_output_text'] + SOURCE_COMPARE_FIELDS})
    right = parsed_same[['row_id', 'model_output_text'] + SOURCE_COMPARE_FIELDS].copy().rename(columns={c: f'{c}__same_chat' for c in ['model_output_text'] + SOURCE_COMPARE_FIELDS})
    detail = left.merge(right, on='row_id', how='inner')

    disagreement_rows = []
    field_rows = []
    for _, row in detail.iterrows():
        disagreement_fields = []
        for field in SOURCE_COMPARE_FIELDS:
            same = _compare_field(field, row[f'{field}__output'], row[f'{field}__same_chat'])
            field_rows.append({'field': field, 'row_id': int(row['row_id']), 'is_equal': bool(same)})
            if not same:
                disagreement_fields.append(field)
        disagreement_rows.append({
            'row_id': int(row['row_id']),
            'source_report': row['source_report'],
            'disagreement_count': len(disagreement_fields),
            'disagreement_fields': '; '.join(disagreement_fields) if disagreement_fields else 'none',
            'movement_source_disagreement': int(any(field in disagreement_fields for field in ['av_movement_group', 'other_party_movement_group'])),
            'scenario_source_disagreement': int('scenario_class' in disagreement_fields),
            'blame_source_disagreement': int(any(field in disagreement_fields for field in ['av_guilty', 'blame_group'])),
            'source_stability_group': 'identical' if not disagreement_fields else 'minor' if len(disagreement_fields) <= 2 else 'major',
            'output_text': row['model_output_text__output'],
            'same_chat_text': row['model_output_text__same_chat'],
        })
    disagreement_detail = pd.DataFrame(disagreement_rows).sort_values(['disagreement_count', 'row_id'], ascending=[False, True]).reset_index(drop=True)
    field_df = pd.DataFrame(field_rows)
    disagreement_summary = (
        field_df.groupby('field')
        .agg(
            compared_rows=('row_id', 'count'),
            equal_count=('is_equal', lambda s: int(pd.Series(s).sum())),
            disagreement_count=('is_equal', lambda s: int((~pd.Series(s)).sum())),
        )
        .reset_index()
    )
    disagreement_summary['disagreement_rate'] = disagreement_summary['disagreement_count'] / disagreement_summary['compared_rows'].clip(lower=1)
    disagreement_summary = disagreement_summary.sort_values(['disagreement_rate', 'field'], ascending=[False, True]).reset_index(drop=True)
    return disagreement_detail, disagreement_summary


def _build_movement_inconsistency_audit(research_df: pd.DataFrame, source_disagreement_detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = research_df.loc[research_df['movement_consistency_overall'] != 'consistent'].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not source_disagreement_detail.empty:
        working = working.merge(
            source_disagreement_detail[['row_id', 'movement_source_disagreement', 'disagreement_count', 'source_stability_group']],
            on='row_id',
            how='left',
        )
    else:
        working['movement_source_disagreement'] = 0
        working['disagreement_count'] = 0
        working['source_stability_group'] = 'unknown'

    def diagnose(row: pd.Series) -> str:
        movement_source_disagreement = pd.to_numeric(pd.Series([row.get('movement_source_disagreement', 0)]), errors='coerce').fillna(0).iloc[0]
        if int(movement_source_disagreement) == 1:
            return 'cross_output_disagreement'
        if row.get('movement_consistency_overall') in {'partial_consistency', 'insufficient'}:
            return 'missing_or_under_specified'
        if row.get('scenario_class') == 'other_or_ambiguous':
            return 'ambiguous_scenario_context'
        return 'within_report_conflict_or_parser_issue'

    working['movement_inconsistency_diagnosis'] = working.apply(diagnose, axis=1)
    audit_columns = [
        'row_id', 'source_report', 'selected_text_column', 'scenario_class', 'scenario_determinability_group',
        'move_v1', 'v1_move', 'av_move_consistency_status', 'move_v2', 'v2_move', 'v2_mov', 'other_move_consistency_status',
        'movement_consistency_overall', 'movement_inconsistency_diagnosis', 'movement_source_disagreement',
        'disagreement_count', 'source_stability_group', 'model_output_text',
    ]
    audit = working[[column for column in audit_columns if column in working.columns]].copy()
    summary = _distribution_table(audit, 'movement_inconsistency_diagnosis')
    return audit.sort_values(['movement_inconsistency_diagnosis', 'row_id']).reset_index(drop=True), summary


def _build_blame_evidence_table(research_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        'row_id', 'source_report', 'selected_text_column', 'scenario_class', 'blame_group', 'av_guilty', 'main_factor',
        'main_factor_grouped', 'q0_confidence', 'q0_confidence_numeric', 'q0_explanation',
        'blame_evidence_score', 'blame_explicitness_group', 'blame_conflict_reason', 'blame_confidence_alignment',
        'blame_evidence_strength', 'report_completeness_score',
    ]
    table = research_df[[column for column in columns if column in research_df.columns]].copy()
    summary = _distribution_table(table, 'blame_evidence_strength')
    return table.sort_values(['blame_evidence_strength', 'row_id']).reset_index(drop=True), summary


def _build_taxonomy_assignment_table(research_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'row_id', 'source_report', 'selected_text_column', 'scenario_class', 'scenario_rule_trigger',
        'scenario_assignment_evidence', 'scenario_evidence_count', 'scenario_determinability_group',
        'road_user_type', 'av_movement_group', 'other_party_movement_group', 'collision_group',
        'intersection_context', 'q0_explanation',
    ]
    return research_df[[column for column in columns if column in research_df.columns]].copy().sort_values(['scenario_class', 'row_id']).reset_index(drop=True)


def _build_other_or_ambiguous_review(research_df: pd.DataFrame, source_disagreement_detail: pd.DataFrame) -> pd.DataFrame:
    working = research_df.loc[research_df['scenario_class'] == 'other_or_ambiguous'].copy()
    if working.empty:
        return pd.DataFrame()
    if not source_disagreement_detail.empty:
        working = working.merge(
            source_disagreement_detail[['row_id', 'disagreement_count', 'disagreement_fields', 'source_stability_group']],
            on='row_id',
            how='left',
        )
    else:
        working['disagreement_count'] = 0
        working['disagreement_fields'] = 'none'
        working['source_stability_group'] = 'unknown'

    def priority(row: pd.Series) -> str:
        disagreement_count = pd.to_numeric(pd.Series([row.get('disagreement_count', 0)]), errors='coerce').fillna(0).iloc[0]
        if row.get('scenario_determinability_group') == 'high' or int(disagreement_count) > 0:
            return 'high_review_priority'
        if row.get('scenario_determinability_group') == 'medium':
            return 'medium_review_priority'
        return 'low_review_priority'

    working['review_priority'] = working.apply(priority, axis=1)
    columns = [
        'row_id', 'source_report', 'selected_text_column', 'scenario_determinability_group', 'review_priority',
        'scenario_rule_trigger', 'scenario_assignment_evidence', 'scenario_evidence_count',
        'road_user_type', 'av_movement_group', 'other_party_movement_group', 'collision_group', 'intersection_context',
        'movement_consistency_overall', 'blame_group', 'blame_evidence_strength', 'disagreement_count',
        'disagreement_fields', 'source_stability_group', 'q0_explanation', 'model_output_text',
    ]
    return working[[column for column in columns if column in working.columns]].copy().sort_values(['review_priority', 'row_id']).reset_index(drop=True)


def _build_field_contradiction_report(research_df: pd.DataFrame, source_disagreement_detail: pd.DataFrame) -> pd.DataFrame:
    working = research_df[['row_id', 'source_report', 'scenario_class', 'movement_consistency_overall', 'blame_conflict_flag', 'blame_conflict_reason']].copy()
    if not source_disagreement_detail.empty:
        working = working.merge(source_disagreement_detail[['row_id', 'disagreement_count', 'source_stability_group']], on='row_id', how='left')
    else:
        working['disagreement_count'] = 0
        working['source_stability_group'] = 'unknown'
    working['movement_conflict_flag'] = working['movement_consistency_overall'].eq('inconsistent')
    working['source_disagreement_flag'] = pd.to_numeric(working['disagreement_count'], errors='coerce').fillna(0).gt(0)
    working['any_contradiction_flag'] = working['movement_conflict_flag'] | working['blame_conflict_flag'] | working['source_disagreement_flag']
    return working.sort_values(['any_contradiction_flag', 'row_id'], ascending=[False, True]).reset_index(drop=True)


def build_research_summary(
    research_df: pd.DataFrame,
    filtered_research_df: pd.DataFrame,
    filter_report: dict[str, Any],
    blind_spot_fields: list[str],
    taxonomy_top_n: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    blind_spots = _missingness_table(research_df, blind_spot_fields)
    field_provenance = _field_provenance_table(research_df)
    provenance_summary = _provenance_summary_table(field_provenance)
    taxonomy_counts = _top_counts(filtered_research_df, 'scenario_class', taxonomy_top_n)
    blame_counts = _top_counts(filtered_research_df, 'blame_group', taxonomy_top_n)
    road_user_counts = _top_counts(filtered_research_df, 'road_user_type', taxonomy_top_n)
    accountability_by_taxonomy = (
        filtered_research_df.groupby(['scenario_class', 'blame_group'])
        .size()
        .reset_index(name='count')
        .sort_values(['scenario_class', 'count'], ascending=[True, False])
    )
    taxonomy_by_road_user = (
        filtered_research_df.groupby(['scenario_class', 'road_user_type'])
        .size()
        .reset_index(name='count')
        .sort_values(['scenario_class', 'count'], ascending=[True, False])
    )
    parse_quality = (
        research_df[['row_id', 'source_report', 'selected_text_column', 'selected_text_score', 'parse_key_hits', 'parse_coverage']]
        .copy()
        .sort_values('selected_text_score', ascending=False)
    )
    consistency_distribution = _distribution_table(research_df, 'movement_consistency_overall')
    determinability_distribution = _distribution_table(filtered_research_df, 'scenario_determinability_group')
    environment_distribution = _distribution_table(filtered_research_df, 'environment_friction_profile')
    vulnerability_distribution = _distribution_table(filtered_research_df, 'road_user_vulnerability_group')
    blame_alignment_distribution = _distribution_table(research_df, 'blame_confidence_alignment')
    explicitness_distribution = _distribution_table(research_df, 'report_explicitness_band')
    external_enrichment_distribution = _distribution_table(research_df, 'external_enrichment_group')
    stopped_av_subtype_counts = _distribution_table(filtered_research_df.loc[filtered_research_df['stopped_av_subtype'] != 'NA'], 'stopped_av_subtype')
    intersection_detail_counts = _distribution_table(research_df, 'intersection_detail_quality')
    score_summary = _score_summary_table(research_df)
    coarse_vs_fine_summary = pd.DataFrame([
        {'metric': 'coarse_context_score', 'mean': round(float(pd.to_numeric(research_df['coarse_context_score'], errors='coerce').mean()), 3)},
        {'metric': 'fine_context_score', 'mean': round(float(pd.to_numeric(research_df['fine_context_score'], errors='coerce').mean()), 3)},
        {'metric': 'context_granularity_gap', 'mean': round(float(pd.to_numeric(research_df['context_granularity_gap'], errors='coerce').mean()), 3)},
    ])

    data_availability = _data_availability_table(research_df, filtered_research_df)
    retained_vs_dropped = _retained_vs_dropped_comparison(research_df, filtered_research_df)
    source_disagreement_audit, source_disagreement_summary = _build_source_disagreement_tables(research_df, blind_spot_fields)
    movement_inconsistency_audit, movement_inconsistency_summary = _build_movement_inconsistency_audit(research_df, source_disagreement_audit)
    blame_evidence_table, blame_evidence_strength_distribution = _build_blame_evidence_table(research_df)
    taxonomy_assignment_explanations = _build_taxonomy_assignment_table(research_df)
    other_or_ambiguous_review = _build_other_or_ambiguous_review(research_df, source_disagreement_audit)
    field_contradiction_report = _build_field_contradiction_report(research_df, source_disagreement_audit)

    summary = {
        'rows_total': int(len(research_df)),
        'rows_used_for_empirical_analysis': int(len(filtered_research_df)),
        'rows_excluded_before_empirical_analysis': int(len(research_df) - len(filtered_research_df)),
        'filter_report': filter_report,
        'taxonomy_top_counts': safe_int_dict(dict(zip(taxonomy_counts['scenario_class'], taxonomy_counts['count']))),
        'blind_spot_top_missingness': {str(k): round(float(v), 3) for k, v in zip(blind_spots['field'].head(10),
                                                                                  blind_spots['missing_rate'])},
        'blame_distribution': safe_int_dict(dict(zip(blame_counts['blame_group'], blame_counts['count']))),
        'road_user_distribution': safe_int_dict(dict(zip(road_user_counts['road_user_type'], road_user_counts['count']))),
        'provenance_mean_availability': {str(k): round(float(v), 3) for k, v in zip(provenance_summary['provenance'],
                                                                                    provenance_summary['mean_availability_rate'])},
        'movement_consistency_distribution': safe_int_dict(dict(zip(consistency_distribution['movement_consistency_overall'], consistency_distribution['count']))),
        'scenario_determinability_distribution': safe_int_dict(dict(zip(determinability_distribution['scenario_determinability_group'], determinability_distribution['count']))),
        'environment_profile_distribution': safe_int_dict(dict(zip(environment_distribution['environment_friction_profile'], environment_distribution['count']))),
        'blame_confidence_alignment': safe_int_dict(dict(zip(blame_alignment_distribution['blame_confidence_alignment'], blame_alignment_distribution['count']))),
        'external_enrichment_distribution': safe_int_dict(dict(zip(external_enrichment_distribution['external_enrichment_group'], external_enrichment_distribution['count']))),
        'data_availability_summary': safe_int_dict(dict(zip(data_availability['stage'], data_availability['count']))),
        'source_disagreement_summary': safe_int_dict(dict(zip(source_disagreement_summary['field'],
                                                              source_disagreement_summary['disagreement_count']))) if not source_disagreement_summary.empty else {},
        'movement_inconsistency_diagnosis': safe_int_dict(dict(zip(movement_inconsistency_summary['movement_inconsistency_diagnosis'],
                                                                   movement_inconsistency_summary['count']))) if not movement_inconsistency_summary.empty else {},
        'blame_evidence_strength_distribution': safe_int_dict(dict(zip(blame_evidence_strength_distribution['blame_evidence_strength'],
                                                                       blame_evidence_strength_distribution['count']))) if not blame_evidence_strength_distribution.empty else {},
        'average_completeness_score': round(float(research_df['report_completeness_score'].mean()), 3) if len(research_df) else 0.0,
        'median_completeness_score': round(float(research_df['report_completeness_score'].median()), 3) if len(research_df) else 0.0,
        'average_explicitness_score': round(float(research_df['report_explicitness_score'].mean()), 3) if len(research_df) else 0.0,
        'average_coarse_context_score': round(float(research_df['coarse_context_score'].mean()), 3) if len(research_df) else 0.0,
        'average_fine_context_score': round(float(research_df['fine_context_score'].mean()), 3) if len(research_df) else 0.0,
        'average_context_gap': round(float(research_df['context_granularity_gap'].mean()), 3) if len(research_df) else 0.0,
        'other_or_ambiguous_review_count': int(len(other_or_ambiguous_review)),
    }
    tables = {
        'taxonomy_counts': taxonomy_counts,
        'blind_spot_missingness': blind_spots,
        'blame_counts': blame_counts,
        'road_user_counts': road_user_counts,
        'accountability_by_taxonomy': accountability_by_taxonomy,
        'taxonomy_by_road_user': taxonomy_by_road_user,
        'parse_quality': parse_quality,
        'field_provenance': field_provenance,
        'provenance_summary': provenance_summary,
        'consistency_distribution': consistency_distribution,
        'determinability_distribution': determinability_distribution,
        'environment_distribution': environment_distribution,
        'vulnerability_distribution': vulnerability_distribution,
        'blame_alignment_distribution': blame_alignment_distribution,
        'explicitness_distribution': explicitness_distribution,
        'external_enrichment_distribution': external_enrichment_distribution,
        'stopped_av_subtype_counts': stopped_av_subtype_counts,
        'intersection_detail_counts': intersection_detail_counts,
        'score_summary': score_summary,
        'coarse_vs_fine_summary': coarse_vs_fine_summary,
        'drop_reason_summary': data_availability,
        'retained_vs_dropped_comparison': retained_vs_dropped,
        'source_disagreement_audit': source_disagreement_audit,
        'source_disagreement_summary': source_disagreement_summary,
        'movement_inconsistency_audit': movement_inconsistency_audit,
        'movement_inconsistency_summary': movement_inconsistency_summary,
        'blame_evidence_strength': blame_evidence_table,
        'blame_evidence_strength_distribution': blame_evidence_strength_distribution,
        'taxonomy_assignment_explanations': taxonomy_assignment_explanations,
        'other_or_ambiguous_review': other_or_ambiguous_review,
        'field_contradiction_report': field_contradiction_report,
    }
    return summary, tables


def export_research_tables(tables: dict[str, pd.DataFrame], output_dir: Any) -> None:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(base_dir / f'{name}.csv', index=False)


def create_validation_sample(df: pd.DataFrame, sample_size: int = 100, seed: int = 42, include_text: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    working = working.sort_values(['scenario_class', 'source_report']).reset_index(drop=True)
    if 'scenario_class' in working.columns and working['scenario_class'].nunique() > 1:
        per_group = max(sample_size // max(working['scenario_class'].nunique(), 1), 1)
        sample = (
            working.groupby('scenario_class', group_keys=False)
            .apply(lambda group: group.sample(min(len(group), per_group), random_state=seed), include_groups=False)
            .reset_index(drop=True)
        )
        if len(sample) < sample_size:
            remaining = working.loc[~working['row_id'].isin(sample['row_id'])]
            top_up = remaining.sample(min(len(remaining), sample_size - len(sample)), random_state=seed)
            sample = pd.concat([sample, top_up], ignore_index=True)
    else:
        sample = working.sample(min(len(working), sample_size), random_state=seed)

    columns = [
        'row_id', 'source_report', 'selected_text_column', 'scenario_class', 'scenario_rule_trigger',
        'road_user_type', 'road_user_vulnerability_group', 'collision_group', 'av_mode_group', 'av_movement_group',
        'other_party_movement_group', 'blame_group', 'blame_evidence_strength', 'scenario_determinability_group',
        'movement_consistency_overall', 'environment_friction_profile', 'intersection_detail_quality',
        'report_completeness_score', 'report_explicitness_score', 'blame_evidence_score',
    ]
    if include_text and 'model_output_text' in sample.columns:
        columns.append('model_output_text')
    result = sample[[column for column in columns if column in sample.columns]].copy()
    result['manual_scenario_class'] = ''
    result['manual_blame_group'] = ''
    result['manual_determinability'] = ''
    result['manual_notes'] = ''
    result['adjudicated_scenario_class'] = ''
    result['adjudicated_blame_group'] = ''
    return result.sort_values('row_id').reset_index(drop=True)


def format_research_markdown(summary: dict[str, Any], config: Any) -> str:
    lines = [
        '# Run report',
        '',
        '## Corpus summary',
        f"- Total parsed reports: {summary.get('rows_total', 0)}",
        f"- Reports used for empirical analysis: {summary.get('rows_used_for_empirical_analysis', 0)}",
        f"- Reports excluded before empirical analysis: {summary.get('rows_excluded_before_empirical_analysis', 0)}",
        f"- Average completeness score: {summary.get('average_completeness_score', 0.0)}",
        f"- Median completeness score: {summary.get('median_completeness_score', 0.0)}",
        f"- Average explicitness score: {summary.get('average_explicitness_score', 0.0)}",
        f"- Average coarse context score: {summary.get('average_coarse_context_score', 0.0)}",
        f"- Average fine context score: {summary.get('average_fine_context_score', 0.0)}",
        f"- Average context gap: {summary.get('average_context_gap', 0.0)}",
        '',
        '## Data availability',
    ]
    for label, count in summary.get('data_availability_summary', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Top taxonomy classes'])
    for label, count in summary.get('taxonomy_top_counts', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Top blind spots'])
    for label, value in summary.get('blind_spot_top_missingness', {}).items():
        lines.append(f'- {label}: missing rate {value}')
    lines.extend(['', '## Provenance availability'])
    for label, value in summary.get('provenance_mean_availability', {}).items():
        lines.append(f'- {label}: mean availability {value}')
    lines.extend(['', '## Movement consistency'])
    for label, count in summary.get('movement_consistency_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Movement inconsistency diagnosis'])
    for label, count in summary.get('movement_inconsistency_diagnosis', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Scenario determinability'])
    for label, count in summary.get('scenario_determinability_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Blame evidence strength'])
    for label, count in summary.get('blame_evidence_strength_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Source disagreement'])
    for label, count in summary.get('source_disagreement_summary', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Environment profiles'])
    for label, count in summary.get('environment_profile_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## External enrichment presence'])
    for label, count in summary.get('external_enrichment_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Blame distribution'])
    for label, count in summary.get('blame_distribution', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend(['', '## Blame confidence alignment'])
    for label, count in summary.get('blame_confidence_alignment', {}).items():
        lines.append(f'- {label}: {count}')
    lines.extend([
        '',
        '## Review queues',
        f"- other_or_ambiguous_review_count: {summary.get('other_or_ambiguous_review_count', 0)}",
        '',
        '## Configuration highlights',
        f'- Row keep policy: {getattr(config, "row_keep_policy", "NA")}',
        f'- Validation sample size: {getattr(config, "validation_sample_size", "NA")}',
        f'- Blind spot fields: {", ".join(getattr(config, "blind_spot_fields", []))}',
    ])
    return "\n".join(lines) + "\n"
