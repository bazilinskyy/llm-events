from __future__ import annotations
#
# Human validation documentation and reproducibility guide
#
# This module intentionally contains extensive comments because it records a
# reviewer facing research procedure rather than only a conventional web app.
# The comments make sampling, blinding, annotation storage, and comparison
# behaviour explicit for coauthors, reviewers, and future maintainers.
#
# Core safeguards documented throughout the file:
#
# 1. The shared 100 report list is authoritative once created.
# 2. Reviewer annotations are coded from the source PDFs, not from LLM output.
# 3. LLM values remain hidden from the reviewer facing HTML.
# 4. Ambiguous and not stated source evidence remain distinct categories.
# 5. Reviewer 1 and Reviewer 2 annotations are stored as separate observations.
# 6. LLM agreement is calculated separately against each reviewer.
# 7. Human disagreement is retained for audit rather than silently overwritten.
# 8. Q0 responsibility is an interpretation task and not a bounded extraction field.
# 9. Sampling metadata are retained so the stratified design remains reproducible.
# 10. Existing validation artefacts are not replaced unless the study is intentionally restarted.
# 11. Autosave preserves incomplete work without marking a report as submitted.
# 12. Export generation is derivative and does not change the stored reviewer answers.
# 13. Source presence coding is kept separate from LLM extraction availability.
# 14. Any future adjudicated reference should be additive to the original reviewer data.
#


"""Human validation interface for the llm-events project.

This application creates one reproducible set of 100 California DMV collision
reports and presents exactly that set to two independent human reviewers in two
different random orders. Reviewers never see the ChatGPT/LLM extraction while
coding. Responses are autosaved to SQLite and can be exported to tidy CSV files
for inter-rater agreement and LLM-vs-human analyses.

Expected repository layout:

    llm-events/
        human_validation.py
        config
        data/                              # common.get_configs("data")
            Output.csv
            Reports/
                *.pdf
            validation_100.csv             # shared by both reviewers
        utils/
            parsing.py
            research.py
            normalise.py

Config entries:
    reviewer              Exactly "Reviewer 1" or "Reviewer 2".
    data                  Dataset directory, e.g. "data".
    validation_pdf_list   Optional path to the shared CSV/TXT file.
                          Default: <data>/validation_100.csv.

The shared validation file is the single source of truth for the 100 reports.
Both reviewers keep access to the full PDF repository under ``data/Reports``
and receive only the same CSV/TXT containing the 100 filenames. The application
uses that file to filter the reports. Reviewer order is derived deterministically
from the reviewer config value, so no reviewer-specific manifest needs to be
shared. If the list does not yet exist, Reviewer 1 creates it once from the
stratified sampling procedure. An existing list is never regenerated or
overwritten. Reviewer 2 will refuse to start until the shared list is available.

Environment variables:
    VALIDATION_HOST        Flask host, default 127.0.0.1.
    VALIDATION_PORT        Flask port, default 5000.
    VALIDATION_SECRET      Flask secret key.

Run from the llm-events repository root:
    uv add flask
    python human_validation.py

Reviewer URL:
    http://127.0.0.1:5000/review/1

The active reviewer is selected only through the config file.
"""

import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import common
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    send_from_directory,
    url_for,
)

from utils.normalise import (
    first_token_csv_style,
    is_missing,
    normalise_boolish,
    normalise_collision,
    normalise_mode,
    normalise_movement,
    normalise_road_user,
)
from utils.parsing import parse_events_dataframe
from utils.research import derive_research_columns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================================
# Developer notes for `_resolve_repo_path`
# ==========================================================================
# Purpose:
#   Resolve a configured path relative to the repository root.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Configuration is resolved once so every later path points to the same study resources.
#   Repository relative paths are preferred because reviewer machines may use different absolute locations.
#   Existing frozen validation artefacts take precedence over regeneration.
#   Errors should identify the exact missing path or invalid configuration value.
#   Backwards compatible path handling is retained for older project configurations.
#   Do not silently fall back to a different validation sample when the configured one is unavailable.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _resolve_repo_path(value: Any) -> Path:
    """Resolve a config path relative to the llm-events repository root."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(common.root_dir) / path
    return path.resolve()


# ==========================================================================
# Developer notes for `_reviewer_from_config`
# ==========================================================================
# Purpose:
#   Resolve the active reviewer label and stable internal reviewer identifier.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Configuration is resolved once so every later path points to the same study resources.
#   Repository relative paths are preferred because reviewer machines may use different absolute locations.
#   Existing frozen validation artefacts take precedence over regeneration.
#   Errors should identify the exact missing path or invalid configuration value.
#   Backwards compatible path handling is retained for older project configurations.
#   Do not silently fall back to a different validation sample when the configured one is unavailable.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _reviewer_from_config() -> tuple[str, str]:
    """Read the active reviewer from ``config``.

    The config must contain exactly one of the following values::

        "reviewer": "Reviewer 1"
        "reviewer": "Reviewer 2"

    The human-readable value is kept for the interface while a compact ID is
    used internally in SQLite and reviewer-order files.
    """

    try:
        reviewer_label = str(common.get_configs("reviewer")).strip()
    except KeyError as exc:
        raise RuntimeError(
            'The config file must contain "reviewer": "Reviewer 1" or '
            '"reviewer": "Reviewer 2".'
        ) from exc

    mapping = {
        "Reviewer 1": "reviewer1",
        "Reviewer 2": "reviewer2",
    }
    if reviewer_label not in mapping:
        raise RuntimeError(
            'Invalid reviewer config. Use exactly "Reviewer 1" or "Reviewer 2".'
        )
    return reviewer_label, mapping[reviewer_label]


CURRENT_REVIEWER_LABEL, CURRENT_REVIEWER_ID = _reviewer_from_config()

# Dataset paths follow the current repository layout:
#
#   data/Output.csv
#   data/Reports/*.pdf
#
# ``data`` can still point directly to Output.csv for backwards compatibility.
DATA_PATH = _resolve_repo_path(common.get_configs("data"))


# ==========================================================================
# Developer notes for `_resolve_input_csv`
# ==========================================================================
# Purpose:
#   Locate the LLM output CSV from the configured data path.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Configuration is resolved once so every later path points to the same study resources.
#   Repository relative paths are preferred because reviewer machines may use different absolute locations.
#   Existing frozen validation artefacts take precedence over regeneration.
#   Errors should identify the exact missing path or invalid configuration value.
#   Backwards compatible path handling is retained for older project configurations.
#   Do not silently fall back to a different validation sample when the configured one is unavailable.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _resolve_input_csv() -> Path:
    """Locate the LLM output CSV from the configured data path.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if DATA_PATH.is_file() and DATA_PATH.suffix.lower() == ".csv":
        return DATA_PATH.resolve()
    candidate = DATA_PATH / "Output.csv"
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    raise RuntimeError(
        f'Could not find Output.csv from common.get_configs("data")={DATA_PATH}. '
        'Expected either a direct CSV path or <data>/Output.csv.'
    )


# ==========================================================================
# Developer notes for `_resolve_pdf_dir`
# ==========================================================================
# Purpose:
#   Locate the directory containing the source collision report PDFs.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Configuration is resolved once so every later path points to the same study resources.
#   Repository relative paths are preferred because reviewer machines may use different absolute locations.
#   Existing frozen validation artefacts take precedence over regeneration.
#   Errors should identify the exact missing path or invalid configuration value.
#   Backwards compatible path handling is retained for older project configurations.
#   Do not silently fall back to a different validation sample when the configured one is unavailable.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _resolve_pdf_dir() -> Path:
    """Locate the directory containing the source collision report PDFs.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if DATA_PATH.is_dir():
        reports = DATA_PATH / "Reports"
        return reports.resolve() if reports.exists() else DATA_PATH.resolve()
    reports = DATA_PATH.parent / "Reports"
    return reports.resolve() if reports.exists() else DATA_PATH.parent.resolve()


# ==========================================================================
# Developer notes for `_resolve_validation_list_path`
# ==========================================================================
# Purpose:
#   Resolve the authoritative shared validation list path.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Configuration is resolved once so every later path points to the same study resources.
#   Repository relative paths are preferred because reviewer machines may use different absolute locations.
#   Existing frozen validation artefacts take precedence over regeneration.
#   Errors should identify the exact missing path or invalid configuration value.
#   Backwards compatible path handling is retained for older project configurations.
#   Do not silently fall back to a different validation sample when the configured one is unavailable.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _resolve_validation_list_path() -> Path:
    """Resolve the authoritative shared validation list path.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    try:
        configured = common.get_configs("validation_pdf_list")
    except KeyError:
        configured = None
    if configured:
        return _resolve_repo_path(configured)
    data_root = DATA_PATH if DATA_PATH.is_dir() else DATA_PATH.parent
    return (data_root / "validation_100.csv").resolve()


INPUT_CSV = _resolve_input_csv()
PDF_DIR = _resolve_pdf_dir()
VALIDATION_LIST_PATH = _resolve_validation_list_path()
OUTPUT_DIR = (BASE_DIR / "_validation").resolve()

DB_PATH = OUTPUT_DIR / "validation.sqlite3"
MANIFEST_PATH = OUTPUT_DIR / "validation_sample_manifest.csv"
ORDERS_PATH = OUTPUT_DIR / "reviewer_orders.csv"
MISSING_PDFS_PATH = OUTPUT_DIR / "missing_pdfs.csv"
ANNOTATIONS_PATH = OUTPUT_DIR / "human_annotations.csv"
FIELD_EVIDENCE_PATH = OUTPUT_DIR / "human_field_evidence.csv"
DISAGREEMENTS_PATH = OUTPUT_DIR / "human_disagreements.csv"
INTER_RATER_PATH = OUTPUT_DIR / "interrater_agreement.csv"
LLM_VS_HUMANS_PATH = OUTPUT_DIR / "llm_vs_humans.csv"
SOURCE_RECOVERY_PATH = OUTPUT_DIR / "source_presence_vs_llm.csv"
EXPORT_ZIP_PATH = OUTPUT_DIR / "validation_exports.zip"
VALIDATION_SET_ID_PATH = OUTPUT_DIR / "validation_set_id.txt"

SAMPLE_SIZE = 100
SAMPLE_SEED = 42
REVIEWER_ORDER_SEEDS = {
    "reviewer1": 7301,
    "reviewer2": 9173,
}

# The reviewers explicitly requested representation of major scenario classes
# and high/low rule support. These quotas deliberately oversample difficult
# low-support reports. The manifest stores sampling fractions so weighted
# analyses can be added later if desired.
SUPPORT_TARGETS = {
    "high": 55,
    "medium": 30,
    "low": 15,
}

DEFAULT_BLIND_SPOT_FIELDS = [
    "v1_lane",
    "v2_lane",
    "v1_speed",
    "v2_speed",
    "v1_intersection",
    "v2_intersection",
    "direction",
    "lane_number",
    "street_type",
    "street_busy",
    "q0_confidence",
    "v1_damage_desc",
    "v2_damage_desc",
]

REVIEWERS = tuple(REVIEWER_ORDER_SEEDS.keys())

app = Flask(__name__)
app.secret_key = os.environ.get(
    "VALIDATION_SECRET",
    "llm-events-human-validation-change-me",
)


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

ROAD_USER_OPTIONS = [
    ("vehicle", "Vehicle / car"),
    ("truck", "Truck"),
    ("bus", "Bus"),
    ("pedestrian", "Pedestrian"),
    ("cyclist", "Cyclist / bicycle"),
    ("scooter", "Scooter"),
    ("motorcycle", "Motorcycle"),
    ("object", "Object / fixed object"),
    ("other", "Other"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

AV_MODE_OPTIONS = [
    ("autonomous", "Autonomous mode"),
    ("conventional", "Conventional / manual mode"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

MOVEMENT_OPTIONS = [
    ("stop", "Stopped / stopping / slowing"),
    ("straight", "Proceeding straight"),
    ("turn_left", "Turning left"),
    ("turn_right", "Turning right"),
    ("turn_other", "Other turn"),
    ("turn_u", "U-turn"),
    ("change_lane", "Changing lane"),
    ("merging", "Merging"),
    ("parked", "Parked / parking"),
    ("backing", "Backing / reversing"),
    ("entering_traffic", "Entering traffic"),
    ("passing", "Passing / overtaking"),
    ("wrong_way", "Wrong-way movement"),
    ("other", "Other movement"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

BOOLEAN_SOURCE_OPTIONS = [
    ("true", "Yes / True"),
    ("false", "No / False"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

CUE_OPTIONS = [
    ("yes", "Yes"),
    ("no", "No"),
    ("not_stated", "Not stated / no evidence"),
    ("ambiguous", "Ambiguous"),
]

COLLISION_OPTIONS = [
    ("rear_end", "Rear end"),
    ("side_swipe", "Side / side swipe"),
    ("broadside", "Broadside"),
    ("head_on", "Head on"),
    ("object", "Object"),
    ("pedestrian", "Pedestrian"),
    ("other", "Other"),
    ("not_stated", "Not stated"),
    ("ambiguous", "Ambiguous"),
]

INJURY_OPTIONS = [
    ("no_injury_marker", "No injury indicated"),
    ("reported_injury", "Injury indicated"),
    ("reported_fatality", "Fatality indicated"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

RESPONSIBILITY_OPTIONS = [
    ("AV_primary", "AV primarily responsible"),
    ("other_road_user", "Other road user primarily responsible"),
    ("environment_or_conditions", "Environment / road conditions"),
    ("unclear", "Cannot determine from the report"),
]

PRESENCE_OPTIONS = [
    ("present", "Clearly stated / recoverable"),
    ("not_stated", "Not stated in the report"),
    ("ambiguous", "Present but ambiguous"),
]

ANNOTATION_FIELDS = [
    "road_user_type_human",
    "av_mode_human",
    "v1_move_narrative_human",
    "v2_move_narrative_human",
    "move_v1_checkbox_human",
    "move_v2_checkbox_human",
    "v1_intersection_human",
    "v2_intersection_human",
    "collision_v1_human",
    "collision_v2_human",
    "parked_or_curbside_cue_human",
    "obstruction_yield_blocked_cue_human",
    "v1_injury_human",
    "v2_injury_human",
    "av_responsibility_human",
    "responsibility_explanation",
    "v1_lane_presence",
    "v1_lane_value",
    "v2_lane_presence",
    "v2_lane_value",
    "v1_speed_presence",
    "v1_speed_value",
    "v2_speed_presence",
    "v2_speed_value",
    "direction_presence",
    "direction_value",
    "general_notes",
]

MULTI_VALUE_FIELDS = {
    "collision_v1_human",
    "collision_v2_human",
}

REQUIRED_ON_SUBMIT = [
    "road_user_type_human",
    "av_mode_human",
    "v1_move_narrative_human",
    "v2_move_narrative_human",
    "move_v1_checkbox_human",
    "move_v2_checkbox_human",
    "v1_intersection_human",
    "v2_intersection_human",
    "collision_v1_human",
    "collision_v2_human",
    "parked_or_curbside_cue_human",
    "obstruction_yield_blocked_cue_human",
    "v1_injury_human",
    "v2_injury_human",
    "av_responsibility_human",
    "v1_lane_presence",
    "v2_lane_presence",
    "v1_speed_presence",
    "v2_speed_presence",
    "direction_presence",
]

FIELD_LABELS = {
    "road_user_type_human": "Other road user type",
    "av_mode_human": "AV operating mode",
    "v1_move_narrative_human": "AV movement from narrative/form text",
    "v2_move_narrative_human": "Other party movement from narrative/form text",
    "move_v1_checkbox_human": "AV movement from page 3 checkbox section",
    "move_v2_checkbox_human": "Other party movement from page 3 checkbox section",
    "v1_intersection_human": "AV at intersection",
    "v2_intersection_human": "Other party at intersection",
    "collision_v1_human": "Collision type marked for AV",
    "collision_v2_human": "Collision type marked for other party",
    "parked_or_curbside_cue_human": "Evidence of parked/curbside/object conflict",
    "obstruction_yield_blocked_cue_human": "Evidence of obstruction/yield/blocked stop",
    "v1_injury_human": "AV occupant injury status",
    "v2_injury_human": "Other party injury status",
    "av_responsibility_human": "Responsibility assessment",
    "v1_lane_presence": "AV lane information presence",
    "v2_lane_presence": "Other party lane information presence",
    "v1_speed_presence": "AV pre-collision speed information presence",
    "v2_speed_presence": "Other party pre-collision speed information presence",
    "direction_presence": "Direction information presence",
}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

BASE_STYLE = """
<style>
:root {
    --border: #d8dde6;
    --bg: #f5f7fa;
    --panel: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --primary: #185adb;
    --primary-dark: #1248b0;
    --success: #16794f;
    --danger: #b42318;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    color: var(--text);
    background: var(--bg);
}
a { color: var(--primary); }
.topbar {
    height: 58px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 18px;
    position: sticky;
    top: 0;
    z-index: 10;
}
.brand { font-weight: 700; }
.muted { color: var(--muted); }
.review-layout {
    height: calc(100vh - 58px);
    display: grid;
    grid-template-columns: minmax(520px, 60%) minmax(390px, 40%);
}
.pdf-panel {
    background: #e8ebf0;
    border-right: 1px solid var(--border);
    min-width: 0;
}
.pdf-panel iframe {
    width: 100%;
    height: 100%;
    border: 0;
    background: white;
}
.form-panel {
    overflow-y: auto;
    padding: 18px 20px 90px;
}
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
}
.card h2 {
    font-size: 17px;
    margin: 0 0 12px;
}
.field { margin-bottom: 13px; }
.field:last-child { margin-bottom: 0; }
label.title {
    display: block;
    font-weight: 600;
    margin-bottom: 6px;
    line-height: 1.3;
}
.help {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin-top: 4px;
}
select, input[type=text], textarea {
    width: 100%;
    padding: 9px 10px;
    border: 1px solid #b8c0cc;
    border-radius: 6px;
    background: white;
    font-size: 14px;
}
textarea { min-height: 82px; resize: vertical; }
.checkbox-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 10px;
    border: 1px solid #b8c0cc;
    border-radius: 6px;
    padding: 10px;
}
.checkbox-grid label { font-size: 13px; }
.fine-grid {
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 8px;
    align-items: center;
}
.actions {
    position: fixed;
    bottom: 0;
    right: 0;
    width: 40%;
    min-width: 390px;
    background: rgba(255,255,255,.97);
    border-top: 1px solid var(--border);
    padding: 11px 20px;
    display: flex;
    gap: 8px;
    z-index: 20;
}
button, .button {
    border: 0;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 14px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
}
button.primary, .button.primary { background: var(--primary); color: white; }
button.primary:hover, .button.primary:hover { background: var(--primary-dark); }
button.secondary, .button.secondary { background: #e9edf4; color: var(--text); }
.save-status { margin-left: auto; font-size: 12px; color: var(--muted); align-self: center; }
.progress-wrap {
    width: 180px;
    height: 7px;
    background: #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
}
.progress-bar { height: 100%; background: var(--success); }
.error-box {
    background: #fff0ef;
    border: 1px solid #f0a7a0;
    color: var(--danger);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 12px;
}
.home {
    max-width: 950px;
    margin: 35px auto;
    padding: 0 18px;
}
.home-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
}
.stat { font-size: 30px; font-weight: 700; }
table { border-collapse: collapse; width: 100%; background: white; }
th, td { border: 1px solid var(--border); padding: 8px; text-align: left; }
th { background: #f0f3f7; }
@media (max-width: 900px) {
    .review-layout { grid-template-columns: 1fr; height: auto; }
    .pdf-panel { height: 62vh; }
    .form-panel { overflow: visible; }
    .actions { width: 100%; min-width: 0; }
}
</style>
"""

HOME_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human validation</title>
""" + BASE_STYLE + """
</head>
<body>
<div class="topbar"><div class="brand">LLM Events · Human validation</div><div class="muted">100 shared PDFs ·\
 independent review</div></div>
<div class="home">
    <div class="card">
        <h2>Validation setup</h2>
        <p><strong>Active reviewer:</strong> {{ reviewer_label }}</p>
        <p>The two reviewers see exactly the same 100 reports, but the order is independently randomised for each\
 reviewer. The active reviewer is selected in the project config, and LLM output or LLM-derived labels are never shown\
 on the review pages.</p>
        <p><strong>Data/PDF location:</strong> {{ pdf_dir }}</p>
        <p><strong>ChatGPT output CSV:</strong> {{ input_csv }}</p>
        <p><strong>Shared 100-report list:</strong> {{ validation_list }}</p>
        <p><strong>Validation set ID:</strong> <code>{{ validation_set_id }}</code></p>
        <p class="muted">Reviewer 1 and Reviewer 2 should see the same validation set ID. The CSV/TXT filters the full\
 Reports directory; the 100 PDFs do not need to be copied into a separate folder.</p>
        <p><strong>Validation output:</strong> {{ output_dir }}</p>
    </div>
    <div class="card">
        <h2>{{ reviewer_label }}</h2>
        <div class="stat">{{ reviewer_progress['submitted'] }}/{{ sample_size }}</div>
        <p class="muted">submitted · {{ reviewer_progress['saved'] }} with any saved data</p>
        <div class="progress-wrap"><div class="progress-bar" style="width: {{ reviewer_progress['pct'] }}%"></div></div>
        <p><a class="button primary" href="{{ url_for('review_report', position=reviewer_progress['next_position'])\
 }}">Continue validation</a></p>
    </div>
    <div class="card">
        <h2>Exports</h2>
        <p>Export can be run at any time. Inter-rater agreement becomes complete once annotations from both reviewers\
 are present in the same validation database.</p>
        <a class="button primary" href="{{ url_for('export_validation') }}">Generate and download validation exports</a>
    </div>
</div>
</body>
</html>
"""

REVIEW_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ reviewer_label }} · {{ position }}/{{ sample_size }}</title>
""" + BASE_STYLE + """
</head>
<body>
<div class="topbar">
    <div>
        <span class="brand">{{ reviewer_label }}</span>
        <span class="muted"> · Report {{ position }} / {{ sample_size }} · {{ report_name }}</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px">
        <span class="muted">{{ submitted_count }} submitted</span>
        <div class="progress-wrap"><div class="progress-bar" style="width: {{ progress_pct }}%"></div></div>
        <a href="{{ url_for('home') }}">Home</a>
    </div>
</div>
<div class="review-layout">
    <div class="pdf-panel">
        <iframe src="{{ url_for('serve_pdf', report_name=report_name) }}#view=FitH"></iframe>
    </div>
    <div class="form-panel">
        {% if errors %}
        <div class="error-box">
            <strong>Please complete the required fields before submitting:</strong>
            <ul>{% for error in errors %}<li>{{ error }}</li>{% endfor %}</ul>
        </div>
        {% endif %}
        <form id="annotation-form" method="post">
            <input type="hidden" name="action" id="action" value="save">
            <input type="hidden" name="started_at" value="{{ annotation.get('started_at', '') }}">

            <div class="card">
                <h2>A. Road user and AV mode</h2>
                {{ select_field('road_user_type_human', 'Other road user type', road_user_options, annotation) | safe }}
                {{ select_field('av_mode_human', 'Was the AV operating autonomously at the time?', av_mode_options,\
 annotation) | safe }}
            </div>

            <div class="card">
                <h2>B. Movement evidence</h2>
                <p class="help">Code the narrative/form-text movement separately from the page 3 movement checkbox\
 section. Do not try to reconcile a contradiction; record what each source says.</p>
                {{ select_field('v1_move_narrative_human', 'AV movement from narrative/form text', movement_options,\
 annotation) | safe }}
                {{ select_field('v2_move_narrative_human', 'Other party movement from narrative/form text',\
 movement_options, annotation) | safe }}
                {{ select_field('move_v1_checkbox_human', 'AV movement from page 3 checkbox section', movement_options,\
 annotation) | safe }}
                {{ select_field('move_v2_checkbox_human', 'Other party movement from page 3 checkbox section',\
 movement_options, annotation) | safe }}
            </div>

            <div class="card">
                <h2>C. Intersection and collision</h2>
                {{ select_field('v1_intersection_human', 'Does the report indicate that the AV was in an intersection?',\
 boolean_options, annotation) | safe }}
                {{ select_field('v2_intersection_human', 'Does the report indicate that the other party was in an\
 intersection?', boolean_options, annotation) | safe }}

                {{ checkbox_field('collision_v1_human', 'Collision type(s) marked for the AV', collision_options,\
 annotation) | safe }}
                {{ checkbox_field('collision_v2_human', 'Collision type(s) marked for the other party',\
 collision_options, annotation) | safe }}

                {{ select_field('parked_or_curbside_cue_human', 'Is there source evidence of a parked/curbside vehicle\
 or object conflict?', cue_options, annotation) | safe }}
                {{ select_field('obstruction_yield_blocked_cue_human', 'Is there source evidence that the AV stopped for\
 an obstruction, yielding, blockage, or uncertainty?', cue_options, annotation) | safe }}
            </div>

            <div class="card">
                <h2>D. Injury</h2>
                {{ select_field('v1_injury_human', 'AV occupant injury status', injury_options, annotation) | safe }}
                {{ select_field('v2_injury_human', 'Other party injury status', injury_options, annotation) | safe }}
            </div>

            <div class="card">
                <h2>E. Fine-context source availability</h2>
                <p class="help">This section is specifically for separating information absent from the source report\
 from information that an extraction system may have missed.</p>
                {{ presence_value_field('v1_lane', 'AV lane position', annotation) | safe }}
                {{ presence_value_field('v2_lane', 'Other party lane position', annotation) | safe }}
                {{ presence_value_field('v1_speed', 'AV pre-collision speed', annotation) | safe }}
                {{ presence_value_field('v2_speed', 'Other party pre-collision speed', annotation) | safe }}
                {{ presence_value_field('direction', 'Direction of travel / same-direction information', annotation) |\
 safe }}
            </div>

            <div class="card">
                <h2>F. Responsibility assessment</h2>
                <p class="help">This is an interpretation task, unlike the extraction questions above. Judge only from\
 the contents of the report.</p>
                {{ select_field('av_responsibility_human', 'Which responsibility category is most supported by the\
 report?', responsibility_options, annotation) | safe }}
                <div class="field">
                    <label class="title" for="responsibility_explanation">Brief evidence supporting this\
 judgement</label>
                    <textarea id="responsibility_explanation" name="responsibility_explanation">{{\
 annotation.get('responsibility_explanation', '') }}</textarea>
                </div>
            </div>

            <div class="card">
                <h2>G. Notes</h2>
                <div class="field">
                    <label class="title" for="general_notes">Optional notes about ambiguity, document quality, or\
 unusual cases</label>
                    <textarea id="general_notes" name="general_notes">{{ annotation.get('general_notes', '')\
 }}</textarea>
                </div>
            </div>
        </form>
    </div>
</div>
<div class="actions">
    {% if position > 1 %}
        <a class="button secondary" href="{{ url_for('review_report', position=position-1) }}">Previous</a>
    {% endif %}
    <button class="secondary" type="button" onclick="saveOnly()">Save</button>
    <button class="primary" type="button" onclick="submitAndNext()">Submit & next</button>
    <span class="save-status" id="save-status">{{ 'Submitted' if annotation.get('submitted_at') else 'Autosave on'\
 }}</span>
</div>
<script>
const form = document.getElementById('annotation-form');
const saveStatus = document.getElementById('save-status');
let saveTimer = null;

function collectFormData() {
    return new FormData(form);
}

async function autosave() {
    const data = collectFormData();
    data.set('action', 'autosave');
    saveStatus.textContent = 'Saving…';
    try {
        const response = await fetch('{{ url_for('autosave_report', position=position) }}', {
            method: 'POST',
            body: data
        });
        if (!response.ok) throw new Error('Autosave failed');
        saveStatus.textContent = 'Saved';
    } catch (error) {
        saveStatus.textContent = 'Save failed';
    }
}

form.addEventListener('change', () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(autosave, 250);
});
form.addEventListener('input', () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(autosave, 800);
});

function saveOnly() {
    document.getElementById('action').value = 'save';
    form.submit();
}

function submitAndNext() {
    document.getElementById('action').value = 'submit_next';
    form.submit();
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `_html_escape`
# ==========================================================================
# Purpose:
#   Escape free text before inserting it into generated HTML.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Rendered controls must never expose hidden LLM outputs or automated scenario labels to reviewers.
#   Free text values are escaped before insertion into HTML.
#   Control names must remain aligned with the SQLite schema and export field names.
#   The browser interface is a data collection layer and must not reinterpret reviewer selections.
#   Source evidence categories remain explicit so ambiguous and not stated values are not conflated.
#   Changes to field labels should not change the stored machine readable values.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _html_escape(value: Any) -> str:
    """Escape free text before inserting it into generated HTML.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ==========================================================================
# Developer notes for `select_field`
# ==========================================================================
# Purpose:
#   Render one single choice annotation control for the validation form.
#
# Interface:
#   Parameters: name, label, options, annotation.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Rendered controls must never expose hidden LLM outputs or automated scenario labels to reviewers.
#   Free text values are escaped before insertion into HTML.
#   Control names must remain aligned with the SQLite schema and export field names.
#   The browser interface is a data collection layer and must not reinterpret reviewer selections.
#   Source evidence categories remain explicit so ambiguous and not stated values are not conflated.
#   Changes to field labels should not change the stored machine readable values.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def select_field(
    name: str,
    label: str,
    options: list[tuple[str, str]],
    annotation: dict[str, Any],
) -> str:
    """Render one single choice annotation control for the validation form.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    current = str(annotation.get(name, "") or "")
    option_html = ['<option value="">Select…</option>']
    for value, text in options:
        selected = " selected" if current == value else ""
        option_html.append(
            f'<option value="{_html_escape(value)}"{selected}>{_html_escape(text)}</option>'
        )
    return (
        '<div class="field">'
        f'<label class="title" for="{_html_escape(name)}">{_html_escape(label)}</label>'
        f'<select id="{_html_escape(name)}" name="{_html_escape(name)}">'
        + "".join(option_html)
        + "</select></div>"
    )


# ==========================================================================
# Developer notes for `checkbox_field`
# ==========================================================================
# Purpose:
#   Render one multiple choice checkbox control for the validation form.
#
# Interface:
#   Parameters: name, label, options, annotation.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Rendered controls must never expose hidden LLM outputs or automated scenario labels to reviewers.
#   Free text values are escaped before insertion into HTML.
#   Control names must remain aligned with the SQLite schema and export field names.
#   The browser interface is a data collection layer and must not reinterpret reviewer selections.
#   Source evidence categories remain explicit so ambiguous and not stated values are not conflated.
#   Changes to field labels should not change the stored machine readable values.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def checkbox_field(
    name: str,
    label: str,
    options: list[tuple[str, str]],
    annotation: dict[str, Any],
) -> str:
    """Render one multiple choice checkbox control for the validation form.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    current_raw = str(annotation.get(name, "") or "")
    current = {value for value in current_raw.split("|") if value}
    parts = [
        '<div class="field">',
        f'<label class="title">{_html_escape(label)}</label>',
        '<div class="checkbox-grid">',
    ]
    for value, text in options:
        checked = " checked" if value in current else ""
        parts.append(
            '<label>'
            f'<input type="checkbox" name="{_html_escape(name)}" '
            f'value="{_html_escape(value)}"{checked}> {_html_escape(text)}'
            "</label>"
        )
    parts.extend(["</div></div>"])
    return "".join(parts)


# ==========================================================================
# Developer notes for `presence_value_field`
# ==========================================================================
# Purpose:
#   Render a source presence selector together with an optional recovered value.
#
# Interface:
#   Parameters: prefix, label, annotation.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Rendered controls must never expose hidden LLM outputs or automated scenario labels to reviewers.
#   Free text values are escaped before insertion into HTML.
#   Control names must remain aligned with the SQLite schema and export field names.
#   The browser interface is a data collection layer and must not reinterpret reviewer selections.
#   Source evidence categories remain explicit so ambiguous and not stated values are not conflated.
#   Changes to field labels should not change the stored machine readable values.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def presence_value_field(
    prefix: str,
    label: str,
    annotation: dict[str, Any],
) -> str:
    """Render a source presence selector together with an optional recovered value.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    presence_name = f"{prefix}_presence"
    value_name = f"{prefix}_value"
    current = str(annotation.get(presence_name, "") or "")
    value = str(annotation.get(value_name, "") or "")

    opts = ['<option value="">Select presence…</option>']
    for option_value, option_label in PRESENCE_OPTIONS:
        selected = " selected" if current == option_value else ""
        opts.append(
            f'<option value="{_html_escape(option_value)}"{selected}>{_html_escape(option_label)}</option>'
        )

    return (
        '<div class="field">'
        f'<label class="title">{_html_escape(label)}</label>'
        '<div class="fine-grid">'
        f'<select name="{_html_escape(presence_name)}">{"".join(opts)}</select>'
        f'<input type="text" name="{_html_escape(value_name)}" value="{_html_escape(value)}" '
        'placeholder="Value if clearly stated (optional)">'
        "</div></div>"
    )


app.jinja_env.globals.update(
    select_field=select_field,
    checkbox_field=checkbox_field,
    presence_value_field=presence_value_field,
)


# ---------------------------------------------------------------------------
# Data preparation and sampling
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `utc_now`
# ==========================================================================
# Purpose:
#   Return the current UTC timestamp in the format stored by the validation database.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def utc_now() -> str:
    """Return the current UTC timestamp in the format stored by the validation database.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ==========================================================================
# Developer notes for `_pdf_index`
# ==========================================================================
# Purpose:
#   Build a case insensitive lookup from PDF filename to absolute source path.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _pdf_index() -> dict[str, Path]:
    """Build a case insensitive lookup from PDF filename to absolute source path.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if not PDF_DIR.exists():
        return {}
    index: dict[str, Path] = {}
    for path in PDF_DIR.rglob("*.pdf"):
        index.setdefault(path.name.lower(), path.resolve())
    return index


# ==========================================================================
# Developer notes for `_normalise_validation_names`
# ==========================================================================
# Purpose:
#   Canonicalise validation filenames for stable set comparison.
#
# Interface:
#   Parameters: report_names.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_validation_names(report_names: list[str]) -> list[str]:
    """Return a stable filename-only representation of a validation set."""

    return sorted(
        {Path(str(name).strip()).name for name in report_names if str(name).strip()},
        key=str.lower,
    )


# ==========================================================================
# Developer notes for `_validation_set_id`
# ==========================================================================
# Purpose:
#   Create the short fingerprint used to identify the frozen validation set.
#
# Interface:
#   Parameters: report_names.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _validation_set_id(report_names: list[str]) -> str:
    """Create a short fingerprint for the exact 100-report set.

    The fingerprint ignores row order and filename case. Reviewer 1 and
    Reviewer 2 should therefore see the same ID when they use the same shared
    CSV/TXT file.
    """

    canonical = "\n".join(name.lower() for name in _normalise_validation_names(report_names))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ==========================================================================
# Developer notes for `_read_validation_pdf_list`
# ==========================================================================
# Purpose:
#   Read and validate the authoritative shared list of source PDFs.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _read_validation_pdf_list() -> list[str]:
    """Read the authoritative set of 100 PDF filenames from CSV or plain text.

    The file only contains names. The actual PDFs remain in ``data/Reports``
    (resolved from ``common.get_configs("data")``). The list therefore acts
    as a filter and can be shared instead of copying a separate 100-PDF folder.
    """

    if not VALIDATION_LIST_PATH.exists():
        return []

    suffix = VALIDATION_LIST_PATH.suffix.lower()
    names: list[str] = []
    if suffix == ".csv":
        table = pd.read_csv(VALIDATION_LIST_PATH)
        if table.empty and len(table.columns) == 0:
            return []
        preferred = ["pdf_name", "source_report", "report_pdf", "Report"]
        column = next((c for c in preferred if c in table.columns), None)
        if column is None:
            if len(table.columns) != 1:
                raise RuntimeError(
                    f"{VALIDATION_LIST_PATH} must contain a pdf_name column "
                    "or exactly one filename column."
                )
            column = str(table.columns[0])
        names = table[column].dropna().astype(str).tolist()
    else:
        names = [
            line.strip()
            for line in VALIDATION_LIST_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    cleaned = [Path(name.strip()).name for name in names if name.strip()]
    lowered = [name.lower() for name in cleaned]
    duplicate_keys = sorted({name for name in lowered if lowered.count(name) > 1})
    if duplicate_keys:
        raise RuntimeError(
            "The shared validation list contains duplicate PDF names: "
            + ", ".join(duplicate_keys)
        )
    if len(cleaned) != SAMPLE_SIZE:
        raise RuntimeError(
            f"{VALIDATION_LIST_PATH} contains {len(cleaned)} unique PDF names; "
            f"exactly {SAMPLE_SIZE} are required."
        )
    return cleaned


# ==========================================================================
# Developer notes for `_write_validation_pdf_list`
# ==========================================================================
# Purpose:
#   Create the shared validation list once without overwriting an existing study record.
#
# Interface:
#   Parameters: report_names.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _write_validation_pdf_list(report_names: list[str]) -> None:
    """Create the shared 100-report list once and never overwrite it.

    If the configured CSV/TXT already exists, it is treated as authoritative.
    An existing file is never regenerated or replaced. If the caller attempts
    to create a different set at the same path, execution stops instead.
    """

    names = _normalise_validation_names(report_names)
    if len(names) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Refusing to write validation list with {len(names)} names; "
            f"expected {SAMPLE_SIZE}."
        )

    if VALIDATION_LIST_PATH.exists():
        existing = _normalise_validation_names(_read_validation_pdf_list())
        if existing != names:
            raise RuntimeError(
                f"Validation list already exists at {VALIDATION_LIST_PATH} and "
                "contains a different 100-report set. It will not be overwritten."
            )
        return

    VALIDATION_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VALIDATION_LIST_PATH.suffix.lower() == ".csv":
        pd.DataFrame({"pdf_name": names}).to_csv(VALIDATION_LIST_PATH, index=False)
    else:
        VALIDATION_LIST_PATH.write_text(
            "\n".join(names) + "\n",
            encoding="utf-8",
        )


# ==========================================================================
# Developer notes for `_clean_output_input`
# ==========================================================================
# Purpose:
#   Validate and prepare the LLM output table before research derivation.
#
# Interface:
#   Parameters: df.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _clean_output_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the LLM output table before research derivation.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    required = {"Report", "Output"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )
    working = df.copy()
    working = working.loc[working["Output"].notna()].copy().reset_index(drop=True)
    working["row_id"] = range(len(working))
    working["selected_text_column"] = "Output"
    working["selected_text_score"] = "NA"
    return working


# ==========================================================================
# Developer notes for `_build_research_dataframe`
# ==========================================================================
# Purpose:
#   Parse the stored LLM responses and derive the research variables used for sampling.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _build_research_dataframe() -> pd.DataFrame:
    """Parse the stored LLM responses and derive the research variables used for sampling.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    raw = pd.read_csv(INPUT_CSV)
    raw = _clean_output_input(raw)
    parsed = parse_events_dataframe(raw, text_column="Output")
    return derive_research_columns(
        parsed,
        blind_spot_fields=DEFAULT_BLIND_SPOT_FIELDS,
    )


# ==========================================================================
# Developer notes for `_allocate_by_sqrt_population`
# ==========================================================================
# Purpose:
#   Allocate a target sample across scenario strata using square root population weights.
#
# Interface:
#   Parameters: counts, target.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _allocate_by_sqrt_population(
    counts: pd.Series,
    target: int,
) -> dict[str, int]:
    """Allocate a quota across non-empty scenario strata.

    Every non-empty scenario receives one sample where possible, then remaining
    slots are distributed approximately in proportion to sqrt(population).
    This preserves representation of rare classes without making the sample
    entirely uniform across classes.
    """

    counts = counts[counts > 0].astype(int)
    if counts.empty or target <= 0:
        return {}

    target = min(int(target), int(counts.sum()))
    labels = list(counts.index.astype(str))
    allocation = {label: 0 for label in labels}

    if target >= len(labels):
        for label in labels:
            allocation[label] = 1
        remaining = target - len(labels)
    else:
        ranked = counts.sort_values(ascending=False).index.astype(str).tolist()
        for label in ranked[:target]:
            allocation[label] = 1
        return allocation

    while remaining > 0:
        eligible = [
            label
            for label in labels
            if allocation[label] < int(counts.loc[label])
        ]
        if not eligible:
            break

        weights = {
            label: math.sqrt(float(counts.loc[label]))
            for label in eligible
        }
        total_weight = sum(weights.values()) or 1.0

        raw_add = {
            label: remaining * weights[label] / total_weight
            for label in eligible
        }
        added_any = False
        for label in eligible:
            add = min(
                int(math.floor(raw_add[label])),
                int(counts.loc[label]) - allocation[label],
            )
            if add > 0:
                allocation[label] += add
                remaining -= add
                added_any = True
                if remaining == 0:
                    break

        if remaining == 0:
            break

        # Largest fractional remainder, respecting capacity.
        ranked = sorted(
            eligible,
            key=lambda label: (
                raw_add[label] - math.floor(raw_add[label]),
                counts.loc[label],
            ),
            reverse=True,
        )
        for label in ranked:
            if remaining == 0:
                break
            if allocation[label] < int(counts.loc[label]):
                allocation[label] += 1
                remaining -= 1
                added_any = True

        if not added_any:
            break

    return allocation


# ==========================================================================
# Developer notes for `_normalise_bool_for_reference`
# ==========================================================================
# Purpose:
#   Convert a boolean like LLM value to the validation reference vocabulary.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_bool_for_reference(value: Any) -> str:
    """Convert a boolean like LLM value to the validation reference vocabulary.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    normalised = normalise_boolish(value)
    if normalised == "True":
        return "true"
    if normalised == "False":
        return "false"
    return "not_stated"


# ==========================================================================
# Developer notes for `_normalise_collision_set`
# ==========================================================================
# Purpose:
#   Convert collision labels into a stable pipe separated reference representation.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_collision_set(value: Any) -> str:
    """Convert collision labels into a stable pipe separated reference representation.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if is_missing(value):
        return "not_stated"
    raw = str(value)
    tokens = []
    for part in raw.replace("/", ",").replace("|", ",").split(","):
        part = part.strip()
        if not part:
            continue
        token = normalise_collision(part)
        if token and token not in {"NA", "na", "unknown"}:
            tokens.append(token)
    if not tokens:
        return "not_stated"
    return "|".join(sorted(set(tokens)))


# ==========================================================================
# Developer notes for `_normalise_movement_reference`
# ==========================================================================
# Purpose:
#   Map movement output to the controlled movement vocabulary used for comparison.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_movement_reference(value: Any) -> str:
    """Map movement output to the controlled movement vocabulary used for comparison.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    token = normalise_movement(value)
    if token in {"NA", "na", "unknown", ""}:
        return "not_stated"
    known = {value for value, _ in MOVEMENT_OPTIONS}
    return token if token in known else "other"


# ==========================================================================
# Developer notes for `_normalise_road_user_reference`
# ==========================================================================
# Purpose:
#   Map road user output to the controlled validation vocabulary.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_road_user_reference(value: Any) -> str:
    """Map road user output to the controlled validation vocabulary.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    token = normalise_road_user(value)
    if token in {"NA", "na", "unknown", ""}:
        return "not_stated"
    known = {value for value, _ in ROAD_USER_OPTIONS}
    return token if token in known else "other"


# ==========================================================================
# Developer notes for `_normalise_mode_reference`
# ==========================================================================
# Purpose:
#   Map AV operating mode output to the controlled validation vocabulary.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _normalise_mode_reference(value: Any) -> str:
    """Map AV operating mode output to the controlled validation vocabulary.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    token = normalise_mode(value)
    if token == "unknown":
        return "not_stated"
    return token


# ==========================================================================
# Developer notes for `_injury_reference`
# ==========================================================================
# Purpose:
#   Convert extracted injury information to the validation injury categories.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _injury_reference(value: Any) -> str:
    """Convert extracted injury information to the validation injury categories.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if is_missing(value):
        return "not_stated"
    text = str(value).strip().lower()
    if text in {"no_injury_marker", "none"} or "no injur" in text:
        return "no_injury_marker"
    if any(token in text for token in ["fatal", "death", "deceased"]):
        return "reported_fatality"
    if any(
        token in text
        for token in ["injur", "hospital", "medical", "transported"]
    ):
        return "reported_injury"
    return "ambiguous"


# ==========================================================================
# Developer notes for `_llm_available`
# ==========================================================================
# Purpose:
#   Return whether an LLM extracted value is available after normalisation.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _llm_available(value: Any) -> bool:
    """Return whether an LLM extracted value is available after normalisation.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return not is_missing(value)


# ==========================================================================
# Developer notes for `_manifest_row`
# ==========================================================================
# Purpose:
#   Create one hidden manifest row containing sampling metadata and LLM comparison fields.
#
# Interface:
#   Parameters: row, validation_id.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _manifest_row(row: pd.Series, validation_id: str) -> dict[str, Any]:
    """Create one hidden manifest row containing sampling metadata and LLM comparison fields.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return {
        "validation_id": validation_id,
        "row_id": row.get("row_id"),
        "source_report": row.get("source_report"),
        "report_pdf": row.get("report_pdf"),
        "manufacturer_group": row.get("manufacturer_group"),
        "report_period": row.get("report_period"),
        "sampling_support_group": row.get("scenario_rule_support_group"),
        "sampling_scenario_class": row.get("scenario_class"),
        "movement_field_agreement_llm": row.get("movement_field_agreement"),
        "scenario_candidate_count_llm": row.get("scenario_candidate_count"),
        "scenario_rule_support_score_llm": row.get("scenario_rule_support_score"),
        "scenario_class_llm": row.get("scenario_class"),
        "blame_group_llm": row.get("blame_group"),
        # Principal field-level LLM outputs. These are kept in the hidden
        # manifest and never rendered on review pages.
        "road_user_type_llm": _normalise_road_user_reference(row.get("v2_id")),
        "av_mode_llm": _normalise_mode_reference(row.get("v1_av")),
        "v1_move_narrative_llm": _normalise_movement_reference(row.get("v1_move")),
        "v2_move_narrative_llm": _normalise_movement_reference(row.get("v2_move")),
        "move_v1_checkbox_llm": _normalise_movement_reference(row.get("move_v1")),
        "move_v2_checkbox_llm": _normalise_movement_reference(row.get("move_v2")),
        "v1_intersection_llm": _normalise_bool_for_reference(row.get("v1_intersection")),
        "v2_intersection_llm": _normalise_bool_for_reference(row.get("v2_intersection")),
        "collision_v1_llm": _normalise_collision_set(row.get("collision_v1")),
        "collision_v2_llm": _normalise_collision_set(row.get("collision_v2")),
        "v1_injury_llm": _injury_reference(row.get("v1_injury")),
        "v2_injury_llm": _injury_reference(row.get("v2_injury")),
        "v1_lane_llm_available": _llm_available(row.get("v1_lane")),
        "v2_lane_llm_available": _llm_available(row.get("v2_lane")),
        "v1_speed_llm_available": _llm_available(row.get("v1_speed")),
        "v2_speed_llm_available": _llm_available(row.get("v2_speed")),
        "direction_llm_available": _llm_available(row.get("direction")),
    }


# ==========================================================================
# Developer notes for `_manifest_from_shared_list`
# ==========================================================================
# Purpose:
#   Reconstruct the hidden manifest from the frozen shared PDF list.
#
# Interface:
#   Parameters: research_df, report_names.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _manifest_from_shared_list(research_df: pd.DataFrame, report_names: list[str]) -> pd.DataFrame:
    """Build the hidden LLM manifest from the shared fixed PDF-name list."""

    pdf_lookup = _pdf_index()
    missing_pdf = [name for name in report_names if name.lower() not in pdf_lookup]
    if missing_pdf:
        raise RuntimeError(
            "PDFs listed in the shared validation file were not found under "
            f"{PDF_DIR}: " + ", ".join(missing_pdf[:20])
        )

    working = research_df.copy()
    working["_report_key"] = working["source_report"].astype(str).map(
        lambda value: Path(value).name.lower()
    )
    wanted = {Path(name).name.lower() for name in report_names}
    selected = working.loc[working["_report_key"].isin(wanted)].copy()

    present = set(selected["_report_key"].tolist())
    missing_output = sorted(wanted.difference(present))
    if missing_output:
        raise RuntimeError(
            "PDFs listed for validation have no matching Report row in Output.csv: "
            + ", ".join(missing_output[:20])
        )

    duplicate_rows = selected["_report_key"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicates = sorted(selected.loc[duplicate_rows, "_report_key"].unique())
        raise RuntimeError(
            "Output.csv contains multiple rows for validation PDF names: "
            + ", ".join(duplicates[:20])
        )

    # Stable alphabetical order means validation IDs are identical even if the
    # shared CSV rows are accidentally reordered. Reviewer presentation order
    # is randomised separately below.
    selected = selected.sort_values("_report_key").reset_index(drop=True)

    population_counts = (
        working.groupby(["scenario_rule_support_group", "scenario_class"])
        .size()
        .to_dict()
    )
    sample_counts = (
        selected.groupby(["scenario_rule_support_group", "scenario_class"])
        .size()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        validation_id = f"V{index + 1:03d}"
        out = _manifest_row(row, validation_id)
        key = (
            str(row.get("scenario_rule_support_group")),
            str(row.get("scenario_class")),
        )
        population_n = int(population_counts.get(key, 0))
        sample_n = int(sample_counts.get(key, 0))
        out["population_stratum_n"] = population_n
        out["sample_stratum_n"] = sample_n
        out["inclusion_probability"] = (
            sample_n / population_n if population_n else 0.0
        )
        out["sampling_weight"] = population_n / sample_n if sample_n else 0.0
        rows.append(out)

    manifest = pd.DataFrame(rows)
    if len(manifest) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Shared validation list resolved to {len(manifest)} reports; "
            f"expected {SAMPLE_SIZE}."
        )
    return manifest


# ==========================================================================
# Developer notes for `_sample_validation_reports`
# ==========================================================================
# Purpose:
#   Create the stratified 100 report validation sample when no frozen list exists.
#
# Interface:
#   Parameters: research_df.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _sample_validation_reports(research_df: pd.DataFrame) -> pd.DataFrame:
    """Create the stratified 100 report validation sample when no frozen list exists.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    pdf_lookup = _pdf_index()
    if not pdf_lookup:
        raise RuntimeError(
            f"No PDFs found under {PDF_DIR}. The PDF location is resolved from "
            "common.get_configs(\"data\")."
        )

    working = research_df.copy()
    working["pdf_found"] = working["source_report"].astype(str).str.lower().isin(pdf_lookup)

    missing_pdf_rows = working.loc[
        ~working["pdf_found"], ["row_id", "source_report", "scenario_class"]
    ].copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_pdf_rows.to_csv(MISSING_PDFS_PATH, index=False)

    working = working.loc[working["pdf_found"]].copy()
    if len(working) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(working)} reports have matching PDFs under {PDF_DIR}; "
            f"at least {SAMPLE_SIZE} are required."
        )

    rng = random.Random(SAMPLE_SEED)
    sampled_parts: list[pd.DataFrame] = []
    sampled_ids: set[Any] = set()

    for support_group, target in SUPPORT_TARGETS.items():
        support_df = working.loc[
            working["scenario_rule_support_group"].eq(support_group)
        ].copy()
        if support_df.empty:
            continue

        target = min(target, len(support_df))
        counts = support_df["scenario_class"].value_counts()
        allocation = _allocate_by_sqrt_population(counts, target)

        for scenario_class, n_take in allocation.items():
            if n_take <= 0:
                continue
            stratum = support_df.loc[
                support_df["scenario_class"].astype(str).eq(str(scenario_class))
            ]
            # A deterministic but different seed for each stratum.
            stratum_seed = rng.randint(1, 2_000_000_000)
            chosen = stratum.sample(
                n=min(n_take, len(stratum)),
                random_state=stratum_seed,
            )
            sampled_parts.append(chosen)
            sampled_ids.update(chosen["row_id"].tolist())

    sample = (
        pd.concat(sampled_parts, ignore_index=True)
        if sampled_parts
        else pd.DataFrame(columns=working.columns)
    )

    if len(sample) < SAMPLE_SIZE:
        remaining = working.loc[~working["row_id"].isin(sampled_ids)].copy()
        top_up = remaining.sample(
            n=min(SAMPLE_SIZE - len(sample), len(remaining)),
            random_state=SAMPLE_SEED + 99,
        )
        sample = pd.concat([sample, top_up], ignore_index=True)

    if len(sample) > SAMPLE_SIZE:
        sample = sample.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED + 199)

    sample = sample.drop_duplicates(subset=["row_id"]).reset_index(drop=True)
    if len(sample) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Validation sampling produced {len(sample)} unique reports instead of {SAMPLE_SIZE}."
        )

    # Save population/sample stratum sizes for transparent sampling fractions.
    population_counts = (
        working.groupby(["scenario_rule_support_group", "scenario_class"])
        .size()
        .to_dict()
    )
    sample_counts = (
        sample.groupby(["scenario_rule_support_group", "scenario_class"])
        .size()
        .to_dict()
    )

    manifest_rows = []
    for index, row in sample.iterrows():
        validation_id = f"V{index + 1:03d}"
        out = _manifest_row(row, validation_id)
        key = (
            str(row.get("scenario_rule_support_group")),
            str(row.get("scenario_class")),
        )
        population_n = int(population_counts.get(key, 0))
        sample_n = int(sample_counts.get(key, 0))
        out["population_stratum_n"] = population_n
        out["sample_stratum_n"] = sample_n
        out["inclusion_probability"] = (
            sample_n / population_n if population_n else 0.0
        )
        out["sampling_weight"] = (
            population_n / sample_n if sample_n else 0.0
        )
        manifest_rows.append(out)

    manifest = pd.DataFrame(manifest_rows)
    return manifest.sort_values("validation_id").reset_index(drop=True)


# ==========================================================================
# Developer notes for `_build_reviewer_orders`
# ==========================================================================
# Purpose:
#   Create deterministic presentation orders for the configured reviewer identifiers.
#
# Interface:
#   Parameters: manifest.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _build_reviewer_orders(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic presentation orders for the configured reviewer identifiers.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    rows: list[dict[str, Any]] = []
    ids = manifest["validation_id"].tolist()
    for reviewer_id, seed in REVIEWER_ORDER_SEEDS.items():
        order = ids.copy()
        random.Random(seed).shuffle(order)
        for position, validation_id in enumerate(order, start=1):
            rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "position": position,
                    "validation_id": validation_id,
                }
            )
    orders = pd.DataFrame(rows)

    # Defensive check: exact same set, different sequence.
    first = orders.loc[orders["reviewer_id"].eq(REVIEWERS[0]), "validation_id"].tolist()
    second = orders.loc[orders["reviewer_id"].eq(REVIEWERS[1]), "validation_id"].tolist()
    if set(first) != set(second):
        raise RuntimeError("Reviewer orders do not contain the same report set.")
    if first == second:
        raise RuntimeError("Reviewer orders unexpectedly have the same sequence.")
    return orders


# ==========================================================================
# Developer notes for `_initialise_manifest_and_orders`
# ==========================================================================
# Purpose:
#   Initialise the frozen validation manifest, set identifier, and reviewer order files.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _initialise_manifest_and_orders() -> None:
    """Initialise the hidden manifest from the shared list and reviewer orders.

    The shared CSV/TXT file is authoritative. Reviewer 1 creates it once if it
    does not exist. Reviewer 2 must receive that exact file before starting.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_names = _read_validation_pdf_list()
    if not report_names:
        if CURRENT_REVIEWER_ID != "reviewer1":
            raise RuntimeError(
                f"Shared validation list not found: {VALIDATION_LIST_PATH}. "
                "Give Reviewer 2 the exact CSV/TXT created for Reviewer 1. "
                "Reviewer 2 will not generate a replacement list."
            )
        research_df = _build_research_dataframe()
        sampled_manifest = _sample_validation_reports(research_df)
        _write_validation_pdf_list(sampled_manifest["source_report"].astype(str).tolist())
        report_names = _read_validation_pdf_list()

    # Store a short auditable identifier. Both reviewer machines should show
    # the same value when the exact same shared CSV/TXT is being used.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_SET_ID_PATH.write_text(
        _validation_set_id(report_names) + "\n",
        encoding="utf-8",
    )

    research_df = _build_research_dataframe()
    expected_manifest = _manifest_from_shared_list(research_df, report_names)

    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        existing_set = {Path(str(v)).name.lower() for v in existing["source_report"]}
        expected_set = {Path(str(v)).name.lower() for v in expected_manifest["source_report"]}
        if existing_set != expected_set:
            raise RuntimeError(
                "The existing _validation manifest does not match the shared "
                f"validation list at {VALIDATION_LIST_PATH}. Delete _validation "
                "only if you intentionally want to restart this review."
            )
        manifest = existing
    else:
        expected_manifest.to_csv(MANIFEST_PATH, index=False)
        manifest = expected_manifest

    expected_orders = _build_reviewer_orders(manifest)
    if ORDERS_PATH.exists():
        # Once reviewer_orders.csv has been created and used for a study, it is
        # part of the frozen validation record. Do not reject it merely because
        # a later version of the code would generate a different deterministic
        # sequence. Instead, verify that it is internally valid and contains
        # exactly the same validation set as the manifest, then preserve it.
        orders = pd.read_csv(ORDERS_PATH)

        required_columns = {"reviewer_id", "position", "validation_id"}
        missing_columns = required_columns.difference(orders.columns)
        if missing_columns:
            raise RuntimeError(
                "Existing reviewer_orders.csv is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        manifest_ids = manifest["validation_id"].astype(str).tolist()
        manifest_id_set = set(manifest_ids)

        reviewer_sequences: dict[str, list[str]] = {}
        for reviewer in REVIEWERS:
            existing_order = (
                orders.loc[orders["reviewer_id"].astype(str).eq(reviewer)]
                .sort_values("position")
                .copy()
            )

            if len(existing_order) != len(manifest_ids):
                raise RuntimeError(
                    f"Existing reviewer order for {reviewer} contains "
                    f"{len(existing_order)} rows, but {len(manifest_ids)} are required."
                )

            positions = existing_order["position"].tolist()
            expected_positions = list(range(1, len(manifest_ids) + 1))
            if positions != expected_positions:
                raise RuntimeError(
                    f"Existing reviewer order for {reviewer} does not contain "
                    f"positions 1 through {len(manifest_ids)} exactly once."
                )

            sequence = existing_order["validation_id"].astype(str).tolist()
            if len(sequence) != len(set(sequence)):
                raise RuntimeError(
                    f"Existing reviewer order for {reviewer} contains duplicate validation IDs."
                )

            if set(sequence) != manifest_id_set:
                raise RuntimeError(
                    f"Existing reviewer order for {reviewer} does not contain "
                    "the same validation IDs as the validation manifest."
                )

            reviewer_sequences[reviewer] = sequence

        if len(REVIEWERS) >= 2:
            first_sequence = reviewer_sequences[REVIEWERS[0]]
            second_sequence = reviewer_sequences[REVIEWERS[1]]
            if first_sequence == second_sequence:
                # Preserve the study record exactly as it was used.
                #
                # A different presentation order was the intended design, but
                # an existing reviewer_orders.csv may come from a completed
                # study in which both reviewers happened to use the same
                # sequence. That does not invalidate the annotations and must
                # not be "fixed" retrospectively. The important integrity check
                # is that both reviewers coded the same frozen validation set.
                print(
                    "WARNING: Reviewer 1 and Reviewer 2 have the same "
                    "presentation order in the existing reviewer_orders.csv. "
                    "The order is being preserved because it is part of the "
                    "completed validation record."
                )

        # Keep the frozen order exactly as it is. The freshly generated
        # expected_orders variable is used only when no order file exists.
    else:
        expected_orders.to_csv(ORDERS_PATH, index=False)


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `_db_connection`
# ==========================================================================
# Purpose:
#   Open a SQLite connection configured for dictionary style row access.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _db_connection() -> sqlite3.Connection:
    """Open a SQLite connection configured for dictionary style row access.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


# ==========================================================================
# Developer notes for `_initialise_database`
# ==========================================================================
# Purpose:
#   Create the annotation database schema without deleting existing reviewer work.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _initialise_database() -> None:
    """Create the annotation database schema without deleting existing reviewer work.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns_sql = ",\n".join(f'"{field}" TEXT' for field in ANNOTATION_FIELDS)
    with _db_connection() as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS annotations (
                reviewer_id TEXT NOT NULL,
                validation_id TEXT NOT NULL,
                {columns_sql},
                started_at TEXT,
                updated_at TEXT,
                submitted_at TEXT,
                duration_seconds REAL,
                PRIMARY KEY (reviewer_id, validation_id)
            )
            """
        )
        connection.commit()


# ==========================================================================
# Developer notes for `_load_annotation`
# ==========================================================================
# Purpose:
#   Load one reviewer's saved annotation for a validation report.
#
# Interface:
#   Parameters: reviewer_id, validation_id.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _load_annotation(reviewer_id: str, validation_id: str) -> dict[str, Any]:
    """Load one reviewer's saved annotation for a validation report.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    with _db_connection() as connection:
        row = connection.execute(
            "SELECT * FROM annotations WHERE reviewer_id=? AND validation_id=?",
            (reviewer_id, validation_id),
        ).fetchone()
    if row is None:
        return {"started_at": utc_now()}
    return dict(row)


# ==========================================================================
# Developer notes for `_request_annotation_values`
# ==========================================================================
# Purpose:
#   Convert submitted form fields into the canonical annotation dictionary.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _request_annotation_values() -> dict[str, str]:
    """Convert submitted form fields into the canonical annotation dictionary.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    values: dict[str, str] = {}
    for field in ANNOTATION_FIELDS:
        if field in MULTI_VALUE_FIELDS:
            selected = [value for value in request.form.getlist(field) if value]
            values[field] = "|".join(sorted(set(selected)))
        else:
            values[field] = request.form.get(field, "").strip()
    return values


# ==========================================================================
# Developer notes for `_save_annotation`
# ==========================================================================
# Purpose:
#   Insert or update one reviewer annotation while preserving submission metadata.
#
# Interface:
#   Parameters: reviewer_id, validation_id, values, submit, started_at.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _save_annotation(
    reviewer_id: str,
    validation_id: str,
    values: dict[str, str],
    *,
    submit: bool,
    started_at: str | None,
) -> None:
    """Insert or update one reviewer annotation while preserving submission metadata.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    now = utc_now()
    existing = _load_annotation(reviewer_id, validation_id)
    started = (
        started_at
        or existing.get("started_at")
        or now
    )

    submitted_at = existing.get("submitted_at") or ""
    if submit:
        submitted_at = now

    duration_seconds = existing.get("duration_seconds")
    if submit:
        try:
            started_dt = datetime.fromisoformat(started)
            end_dt = datetime.fromisoformat(now)
            duration_seconds = max((end_dt - started_dt).total_seconds(), 0.0)
        except Exception:
            duration_seconds = None

    db_values = {
        "reviewer_id": reviewer_id,
        "validation_id": validation_id,
        **values,
        "started_at": started,
        "updated_at": now,
        "submitted_at": submitted_at,
        "duration_seconds": duration_seconds,
    }

    columns = list(db_values.keys())
    placeholders = ",".join("?" for _ in columns)
    quoted_columns = ",".join(f'"{column}"' for column in columns)
    update_clause = ",".join(
        f'"{column}"=excluded."{column}"'
        for column in columns
        if column not in {"reviewer_id", "validation_id"}
    )

    with _db_connection() as connection:
        connection.execute(
            f"""
            INSERT INTO annotations ({quoted_columns})
            VALUES ({placeholders})
            ON CONFLICT(reviewer_id, validation_id) DO UPDATE SET
            {update_clause}
            """,
            [db_values[column] for column in columns],
        )
        connection.commit()


# ==========================================================================
# Developer notes for `_validate_submission`
# ==========================================================================
# Purpose:
#   Check that all required annotation fields are complete before submission.
#
# Interface:
#   Parameters: values.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   SQLite stores reviewer work locally and must preserve previously submitted annotations.
#   Reviewer identity and validation identifier together form the logical annotation key.
#   Autosave and final submission use the same field schema but different completion semantics.
#   Database initialisation is idempotent so restarting the application does not erase progress.
#   Timestamps are stored in UTC to avoid reviewer machine timezone differences.
#   Any schema change should remain compatible with already collected reviewer annotations.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _validate_submission(values: dict[str, str]) -> list[str]:
    """Check that all required annotation fields are complete before submission.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    errors = []
    for field in REQUIRED_ON_SUBMIT:
        if not values.get(field, "").strip():
            errors.append(FIELD_LABELS.get(field, field))

    for field in MULTI_VALUE_FIELDS:
        selected = {token for token in values.get(field, "").split("|") if token}
        if len(selected) > 1 and selected.intersection({"not_stated", "ambiguous"}):
            errors.append(
                f"{FIELD_LABELS.get(field, field)}: choose either a collision type, "
                "Not stated, or Ambiguous, not a combination."
            )
    return errors


# ==========================================================================
# Developer notes for `_orders_dataframe`
# ==========================================================================
# Purpose:
#   Load the recorded reviewer order table from disk.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _orders_dataframe() -> pd.DataFrame:
    """Load the recorded reviewer order table from disk.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return pd.read_csv(ORDERS_PATH)


# ==========================================================================
# Developer notes for `_manifest_dataframe`
# ==========================================================================
# Purpose:
#   Load the hidden validation manifest from disk.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   The shared validation list is the source of truth once it has been created.
#   Sampling deliberately represents scenario classes and rule support levels rather than using simple random
#   sampling.
#   Population and sample stratum sizes are stored so inclusion probabilities remain auditable.
#   The same source report must not appear twice in the frozen validation set.
#   Sampling code must be deterministic for the same seed and analytical dataframe.
#   A later rerun must not overwrite the completed study sample without an explicit restart.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _manifest_dataframe() -> pd.DataFrame:
    """Load the hidden validation manifest from disk.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return pd.read_csv(MANIFEST_PATH)


# ==========================================================================
# Developer notes for `_validation_id_at`
# ==========================================================================
# Purpose:
#   Resolve the validation identifier shown at a reviewer position.
#
# Interface:
#   Parameters: reviewer_id, position.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _validation_id_at(reviewer_id: str, position: int) -> str:
    """Resolve the validation identifier shown at a reviewer position.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    orders = _orders_dataframe()
    row = orders.loc[
        orders["reviewer_id"].eq(reviewer_id)
        & orders["position"].eq(position)
    ]
    if row.empty:
        abort(404)
    return str(row.iloc[0]["validation_id"])


# ==========================================================================
# Developer notes for `_report_name_for_validation_id`
# ==========================================================================
# Purpose:
#   Resolve the source PDF filename for one validation identifier.
#
# Interface:
#   Parameters: validation_id.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _report_name_for_validation_id(validation_id: str) -> str:
    """Resolve the source PDF filename for one validation identifier.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    manifest = _manifest_dataframe()
    row = manifest.loc[manifest["validation_id"].eq(validation_id)]
    if row.empty:
        abort(404)
    source = str(row.iloc[0]["source_report"])
    return source


# ==========================================================================
# Developer notes for `_progress`
# ==========================================================================
# Purpose:
#   Summarise saved and submitted progress for the active reviewer.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _progress() -> dict[str, dict[str, Any]]:
    """Summarise saved and submitted progress for the active reviewer.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    orders = _orders_dataframe()
    result: dict[str, dict[str, Any]] = {}
    with _db_connection() as connection:
        for reviewer in REVIEWERS:
            rows = connection.execute(
                "SELECT validation_id, submitted_at, updated_at FROM annotations WHERE reviewer_id=?",
                (reviewer,),
            ).fetchall()
            saved_ids = {row["validation_id"] for row in rows if row["updated_at"]}
            submitted_ids = {
                row["validation_id"]
                for row in rows
                if row["submitted_at"]
            }
            order = orders.loc[
                orders["reviewer_id"].eq(reviewer)
            ].sort_values("position")
            next_position = 1
            for _, row in order.iterrows():
                if str(row["validation_id"]) not in submitted_ids:
                    next_position = int(row["position"])
                    break
            else:
                next_position = SAMPLE_SIZE

            result[reviewer] = {
                "saved": len(saved_ids),
                "submitted": len(submitted_ids),
                "pct": round(100 * len(submitted_ids) / SAMPLE_SIZE, 1),
                "next_position": next_position,
            }
    return result


# ---------------------------------------------------------------------------
# Agreement and export helpers
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `_cohen_kappa`
# ==========================================================================
# Purpose:
#   Calculate unweighted Cohen's kappa for two categorical coding series.
#
# Interface:
#   Parameters: left, right.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _cohen_kappa(left: pd.Series, right: pd.Series) -> float | None:
    """Calculate unweighted Cohen's kappa for two categorical coding series.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    paired = paired.loc[
        paired["left"].astype(str).ne("")
        & paired["right"].astype(str).ne("")
    ]
    if paired.empty:
        return None
    observed = float(paired["left"].eq(paired["right"]).mean())
    left_share = paired["left"].value_counts(normalize=True)
    right_share = paired["right"].value_counts(normalize=True)
    labels = set(left_share.index).union(right_share.index)
    expected = sum(
        float(left_share.get(label, 0.0))
        * float(right_share.get(label, 0.0))
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


# ==========================================================================
# Developer notes for `_human_annotations_dataframe`
# ==========================================================================
# Purpose:
#   Export submitted human annotations from SQLite as a dataframe.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _human_annotations_dataframe() -> pd.DataFrame:
    """Export submitted human annotations from SQLite as a dataframe.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    with _db_connection() as connection:
        df = pd.read_sql_query(
            "SELECT * FROM annotations ORDER BY reviewer_id, validation_id",
            connection,
        )
    return df


# ==========================================================================
# Developer notes for `_human_field_evidence`
# ==========================================================================
# Purpose:
#   Convert reviewer annotations to a long field evidence table.
#
# Interface:
#   Parameters: annotations.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _human_field_evidence(annotations: pd.DataFrame) -> pd.DataFrame:
    """Convert reviewer annotations to a long field evidence table.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    rows = []
    for _, row in annotations.iterrows():
        for field in ANNOTATION_FIELDS:
            rows.append(
                {
                    "reviewer_id": row.get("reviewer_id"),
                    "validation_id": row.get("validation_id"),
                    "field": field,
                    "human_value": row.get(field, ""),
                    "submitted_at": row.get("submitted_at", ""),
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# Developer notes for `_interrater_table`
# ==========================================================================
# Purpose:
#   Calculate human to human agreement and preserve field level disagreements.
#
# Interface:
#   Parameters: annotations.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _interrater_table(annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate human to human agreement and preserve field level disagreements.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    submitted = annotations.loc[
        annotations["submitted_at"].fillna("").astype(str).ne("")
    ].copy()
    if submitted.empty:
        return pd.DataFrame(), pd.DataFrame()

    left = submitted.loc[submitted["reviewer_id"].eq(REVIEWERS[0])].set_index("validation_id")
    right = submitted.loc[submitted["reviewer_id"].eq(REVIEWERS[1])].set_index("validation_id")
    common_ids = left.index.intersection(right.index)

    rows = []
    disagreement_rows = []
    fields_for_agreement = [
        field
        for field in ANNOTATION_FIELDS
        if field not in {
            "responsibility_explanation",
            "v1_lane_value",
            "v2_lane_value",
            "v1_speed_value",
            "v2_speed_value",
            "direction_value",
            "general_notes",
        }
    ]

    for field in fields_for_agreement:
        pair = pd.DataFrame(
            {
                REVIEWERS[0]: left.loc[common_ids, field],
                REVIEWERS[1]: right.loc[common_ids, field],
            }
        ).fillna("")
        complete = pair.loc[
            pair[REVIEWERS[0]].astype(str).ne("")
            & pair[REVIEWERS[1]].astype(str).ne("")
        ]
        n = len(complete)
        exact = (
            float(complete[REVIEWERS[0]].eq(complete[REVIEWERS[1]]).mean())
            if n
            else None
        )
        kappa = _cohen_kappa(
            complete[REVIEWERS[0]],
            complete[REVIEWERS[1]],
        ) if n else None
        rows.append(
            {
                "field": field,
                "n_both_coded": n,
                "exact_agreement": exact,
                "cohen_kappa": kappa,
            }
        )

        disagreements = complete.loc[
            ~complete[REVIEWERS[0]].eq(complete[REVIEWERS[1]])
        ]
        for validation_id, disagreement in disagreements.iterrows():
            disagreement_rows.append(
                {
                    "validation_id": validation_id,
                    "field": field,
                    REVIEWERS[0]: disagreement[REVIEWERS[0]],
                    REVIEWERS[1]: disagreement[REVIEWERS[1]],
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(disagreement_rows)


# ==========================================================================
# Developer notes for `_llm_vs_humans_table`
# ==========================================================================
# Purpose:
#   Compare the LLM output separately with each human reviewer.
#
# Interface:
#   Parameters: annotations, manifest.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _llm_vs_humans_table(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the LLM output separately with each human reviewer.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    mappings = {
        "road_user_type_human": "road_user_type_llm",
        "av_mode_human": "av_mode_llm",
        "v1_move_narrative_human": "v1_move_narrative_llm",
        "v2_move_narrative_human": "v2_move_narrative_llm",
        "move_v1_checkbox_human": "move_v1_checkbox_llm",
        "move_v2_checkbox_human": "move_v2_checkbox_llm",
        "v1_intersection_human": "v1_intersection_llm",
        "v2_intersection_human": "v2_intersection_llm",
        "collision_v1_human": "collision_v1_llm",
        "collision_v2_human": "collision_v2_llm",
        "v1_injury_human": "v1_injury_llm",
        "v2_injury_human": "v2_injury_llm",
        "av_responsibility_human": "blame_group_llm",
    }

    submitted = annotations.loc[
        annotations["submitted_at"].fillna("").astype(str).ne("")
    ].copy()
    merged = submitted.merge(manifest, on="validation_id", how="left")

    rows = []
    for reviewer in REVIEWERS:
        reviewer_df = merged.loc[merged["reviewer_id"].eq(reviewer)]
        for human_field, llm_field in mappings.items():
            pair = reviewer_df[[human_field, llm_field]].fillna("").copy()
            pair = pair.loc[
                pair[human_field].astype(str).ne("")
                & pair[llm_field].astype(str).ne("")
            ]
            n = len(pair)
            rows.append(
                {
                    "reviewer_id": reviewer,
                    "field": human_field,
                    "llm_field": llm_field,
                    "n": n,
                    "exact_agreement": (
                        float(pair[human_field].eq(pair[llm_field]).mean())
                        if n else None
                    ),
                    "cohen_kappa": (
                        _cohen_kappa(pair[human_field], pair[llm_field])
                        if n else None
                    ),
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# Developer notes for `_as_bool`
# ==========================================================================
# Purpose:
#   Normalise common truthy values used by source availability comparisons.
#
# Interface:
#   Parameters: value.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Normalisation maps LLM outputs to the human coding vocabulary only for later comparison.
#   The raw LLM response remains available elsewhere and is not modified by this helper.
#   Missing and unknown values remain explicit rather than being forced into a substantive category.
#   Controlled vocabularies should stay aligned with the options shown to human reviewers.
#   New source categories require coordinated updates to sampling, exports, and analysis code.
#   Normalisation must not add information that is absent from the extracted response.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _as_bool(value: Any) -> bool:
    """Normalise common truthy values used by source availability comparisons.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


# ==========================================================================
# Developer notes for `_source_presence_vs_llm`
# ==========================================================================
# Purpose:
#   Compare human source presence coding with LLM extraction availability.
#
# Interface:
#   Parameters: annotations, manifest.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Reviewer 1 and Reviewer 2 remain separate observations throughout the agreement analysis.
#   Exact agreement is reported together with Cohen's kappa because prevalence affects kappa.
#   LLM agreement is calculated separately for each human reviewer and should not be averaged into one accuracy
#   value.
#   Disagreement records are retained for audit and possible later adjudication.
#   A human disagreement is not automatically treated as an LLM error.
#   Any later adjudicated reference must be stored separately from the original independent coding.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _source_presence_vs_llm(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Compare human source presence coding with LLM extraction availability.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    mappings = {
        "v1_lane_presence": "v1_lane_llm_available",
        "v2_lane_presence": "v2_lane_llm_available",
        "v1_speed_presence": "v1_speed_llm_available",
        "v2_speed_presence": "v2_speed_llm_available",
        "direction_presence": "direction_llm_available",
    }

    submitted = annotations.loc[
        annotations["submitted_at"].fillna("").astype(str).ne("")
    ].copy()
    merged = submitted.merge(manifest, on="validation_id", how="left")
    rows = []

    for reviewer in REVIEWERS:
        reviewer_df = merged.loc[merged["reviewer_id"].eq(reviewer)]
        for presence_field, llm_available_field in mappings.items():
            counts: dict[str, int] = defaultdict(int)
            for _, row in reviewer_df.iterrows():
                source_state = str(row.get(presence_field, "") or "")
                if not source_state:
                    continue
                llm_available = _as_bool(row.get(llm_available_field, False))
                if source_state == "present" and llm_available:
                    outcome = "source_present_llm_recovered"
                elif source_state == "present" and not llm_available:
                    outcome = "source_present_llm_missed"
                elif source_state == "not_stated" and not llm_available:
                    outcome = "source_absent_llm_abstained"
                elif source_state == "not_stated" and llm_available:
                    outcome = "source_absent_llm_returned_value"
                else:
                    outcome = "source_ambiguous"
                counts[outcome] += 1

            total = sum(counts.values())
            for outcome, count in sorted(counts.items()):
                rows.append(
                    {
                        "reviewer_id": reviewer,
                        "field": presence_field,
                        "outcome": outcome,
                        "count": count,
                        "share": count / total if total else None,
                    }
                )
    return pd.DataFrame(rows)


# ==========================================================================
# Developer notes for `_generate_exports`
# ==========================================================================
# Purpose:
#   Generate the validation CSV outputs and package them into a reproducible ZIP archive.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Exports are derivative artefacts; the SQLite database remains the primary local record of reviewer input.
#   Human annotations, disagreements, agreement metrics, and LLM comparisons are exported separately.
#   Generated files must preserve reviewer identifiers and validation identifiers for auditability.
#   Source presence analyses distinguish source coding from extraction availability.
#   Packaging the outputs must not modify the reviewer annotations used to create them.
#   A later adjudication file should be additive rather than replacing these original exports.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def _generate_exports() -> list[Path]:
    """Generate the validation CSV outputs and package them into a reproducible ZIP archive.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    annotations = _human_annotations_dataframe()
    manifest = _manifest_dataframe()

    annotations.to_csv(ANNOTATIONS_PATH, index=False)
    _human_field_evidence(annotations).to_csv(FIELD_EVIDENCE_PATH, index=False)

    interrater, disagreements = _interrater_table(annotations)
    interrater.to_csv(INTER_RATER_PATH, index=False)
    disagreements.to_csv(DISAGREEMENTS_PATH, index=False)

    llm_vs_humans = _llm_vs_humans_table(annotations, manifest)
    llm_vs_humans.to_csv(LLM_VS_HUMANS_PATH, index=False)

    source_recovery = _source_presence_vs_llm(annotations, manifest)
    source_recovery.to_csv(SOURCE_RECOVERY_PATH, index=False)

    export_paths = [
        MANIFEST_PATH,
        ORDERS_PATH,
        ANNOTATIONS_PATH,
        FIELD_EVIDENCE_PATH,
        INTER_RATER_PATH,
        DISAGREEMENTS_PATH,
        LLM_VS_HUMANS_PATH,
        SOURCE_RECOVERY_PATH,
        VALIDATION_SET_ID_PATH,
        VALIDATION_LIST_PATH,
    ]

    with zipfile.ZipFile(EXPORT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in export_paths:
            if path.exists():
                archive.write(path, arcname=path.name)
    return export_paths


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `home`
# ==========================================================================
# Purpose:
#   Render the validation landing page and active reviewer progress.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/")
def home() -> str:
    """Render the validation landing page and active reviewer progress.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    reviewer_progress = _progress()[CURRENT_REVIEWER_ID]
    return render_template_string(
        HOME_TEMPLATE,
        reviewer_label=CURRENT_REVIEWER_LABEL,
        reviewer_id=CURRENT_REVIEWER_ID,
        reviewer_progress=reviewer_progress,
        sample_size=SAMPLE_SIZE,
        pdf_dir=PDF_DIR,
        input_csv=INPUT_CSV,
        validation_list=VALIDATION_LIST_PATH,
        validation_set_id=_validation_set_id(_read_validation_pdf_list()),
        output_dir=OUTPUT_DIR,
    )


# ==========================================================================
# Developer notes for `serve_pdf`
# ==========================================================================
# Purpose:
#   Serve an authorised source PDF from the configured report directory.
#
# Interface:
#   Parameters: report_name.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/pdf/<path:report_name>")
def serve_pdf(report_name: str) -> Response:
    """Serve an authorised source PDF from the configured report directory.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    pdf_lookup = _pdf_index()
    path = pdf_lookup.get(Path(report_name).name.lower())
    if path is None or not path.exists():
        abort(404)
    return send_from_directory(path.parent, path.name, mimetype="application/pdf")


# ==========================================================================
# Developer notes for `review_report`
# ==========================================================================
# Purpose:
#   Render and process the main source report annotation form.
#
# Interface:
#   Parameters: position.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/review/<int:position>", methods=["GET", "POST"])
def review_report(position: int) -> str | Response:
    """Render and process the main source report annotation form.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    reviewer_id = CURRENT_REVIEWER_ID
    if position < 1 or position > SAMPLE_SIZE:
        abort(404)

    validation_id = _validation_id_at(reviewer_id, position)
    report_name = _report_name_for_validation_id(validation_id)
    annotation = _load_annotation(reviewer_id, validation_id)
    errors: list[str] = []

    if request.method == "POST":
        values = _request_annotation_values()
        action = request.form.get("action", "save")
        submit = action == "submit_next"
        if submit:
            errors = _validate_submission(values)
        if not errors:
            _save_annotation(
                reviewer_id,
                validation_id,
                values,
                submit=submit,
                started_at=request.form.get("started_at"),
            )
            if submit and position < SAMPLE_SIZE:
                return redirect(url_for("review_report", position=position + 1))
            if submit and position == SAMPLE_SIZE:
                return redirect(url_for("home"))
            return redirect(url_for("review_report", position=position))
        annotation = {**annotation, **values}

    progress = _progress()[reviewer_id]
    return render_template_string(
        REVIEW_TEMPLATE,
        reviewer_id=reviewer_id,
        reviewer_label=CURRENT_REVIEWER_LABEL,
        position=position,
        sample_size=SAMPLE_SIZE,
        report_name=report_name,
        annotation=annotation,
        errors=errors,
        submitted_count=progress["submitted"],
        progress_pct=progress["pct"],
        road_user_options=ROAD_USER_OPTIONS,
        av_mode_options=AV_MODE_OPTIONS,
        movement_options=MOVEMENT_OPTIONS,
        boolean_options=BOOLEAN_SOURCE_OPTIONS,
        cue_options=CUE_OPTIONS,
        collision_options=COLLISION_OPTIONS,
        injury_options=INJURY_OPTIONS,
        responsibility_options=RESPONSIBILITY_OPTIONS,
    )


# ==========================================================================
# Developer notes for `autosave_report`
# ==========================================================================
# Purpose:
#   Persist an in progress annotation without marking the report as submitted.
#
# Interface:
#   Parameters: position.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/autosave/<int:position>", methods=["POST"])
def autosave_report(position: int) -> Response:
    """Persist an in progress annotation without marking the report as submitted.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    reviewer_id = CURRENT_REVIEWER_ID
    if not (1 <= position <= SAMPLE_SIZE):
        abort(404)
    validation_id = _validation_id_at(reviewer_id, position)
    values = _request_annotation_values()
    _save_annotation(
        reviewer_id,
        validation_id,
        values,
        submit=False,
        started_at=request.form.get("started_at"),
    )
    return jsonify({"ok": True, "saved_at": utc_now()})


# ==========================================================================
# Developer notes for `export_validation`
# ==========================================================================
# Purpose:
#   Generate the current validation exports and return the ZIP archive.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/export")
def export_validation() -> Response:
    """Generate the current validation exports and return the ZIP archive.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    _generate_exports()
    return send_file(
        EXPORT_ZIP_PATH,
        as_attachment=True,
        download_name=EXPORT_ZIP_PATH.name,
    )


# ==========================================================================
# Developer notes for `status`
# ==========================================================================
# Purpose:
#   Return a compact JSON status summary for the validation interface.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   Flask routes expose only the information needed for the requested validation action.
#   The review route must keep the hidden manifest and LLM outputs out of the rendered page.
#   Report filenames are resolved through the frozen manifest rather than accepting arbitrary file paths.
#   Submitted annotations are validated before they are marked complete.
#   Autosave should never convert an incomplete form into a submitted report.
#   Export actions must preserve the underlying reviewer database and source records.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

@app.route("/status")
def status() -> Response:
    """Return a compact JSON status summary for the validation interface.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    return jsonify(
        {
            "sample_size": SAMPLE_SIZE,
            "reviewer": CURRENT_REVIEWER_LABEL,
            "reviewer_id": CURRENT_REVIEWER_ID,
            "progress": _progress()[CURRENT_REVIEWER_ID],
            "data": str(DATA_PATH),
            "pdf_directory": str(PDF_DIR),
            "input_csv": str(INPUT_CSV),
            "validation_pdf_list": str(VALIDATION_LIST_PATH),
            "validation_set_id": _validation_set_id(_read_validation_pdf_list()),
            "manifest": str(MANIFEST_PATH),
            "orders": str(ORDERS_PATH),
            "database": str(DB_PATH),
        }
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


# ==========================================================================
# Developer notes for `initialise_validation`
# ==========================================================================
# Purpose:
#   Initialise all validation artefacts required before starting the Flask application.
#
# Interface:
#   Parameters: none.
#   The documented return value is the contract used by downstream validation code.
#
# Implementation and research safeguards:
#   This helper isolates one repeated operation so the validation workflow remains easier to audit.
#   Return types should stay stable because several later steps depend on exact field names and values.
#   Missing information is preserved explicitly rather than silently imputed.
#   Deterministic behaviour is preferred because the application supports reproducible research.
#   Errors should fail visibly instead of silently generating a different study state.
#   Changes here should be reflected in the README when they alter the reviewer workflow.
#
# Maintenance guidance:
#   Keep stored field names stable unless the database schema, exports, and analysis code are updated together.
#   Do not add hidden heuristics that change reviewer answers, validation membership, or comparison categories.
#   If this helper changes study behaviour, update the README and preserve the previous frozen study artefacts.
# ==========================================================================

def initialise_validation() -> None:
    """Initialise all validation artefacts required before starting the Flask application.

    The helper is intentionally deterministic where study state or comparison
    outputs depend on it. Missing information is preserved explicitly rather
    than being inferred from unrelated fields.
    """

    if not INPUT_CSV.exists():
        raise RuntimeError(
            f"Input CSV not found: {INPUT_CSV}. If common.get_configs(\"data\") "
            "points to a PDF directory, place Output.csv in that directory, "
            "in its _output subdirectory, or in the repository root/_output."
        )
    _initialise_manifest_and_orders()
    _initialise_database()


if __name__ == "__main__":
    initialise_validation()
    host = os.environ.get("VALIDATION_HOST", "127.0.0.1")
    port = int(os.environ.get("VALIDATION_PORT", "5000"))
    print("Human validation interface ready")
    print(f"Active reviewer: {CURRENT_REVIEWER_LABEL}")
    print(f"Review URL:      http://{host}:{port}/review/1")
    print(f"Outputs:    {OUTPUT_DIR}")
    app.run(host=host, port=port, debug=False)
