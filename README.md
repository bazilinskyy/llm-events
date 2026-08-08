# California autonomous vehicle collision report analysis

This project parses LLM responses produced from California DMV autonomous
vehicle collision reports and applies deterministic research rules to create
descriptive tables, robustness checks, audit files, and figures.

The revised analysis deliberately distinguishes three concepts:

1. post extraction unavailability, where a value is unavailable in the parsed
   dataset;
2. report context, which covers information extracted from the regulatory form
   and narrative; and
3. external context, which covers online enrichment.

Post extraction unavailability must not be interpreted as proof that a field
was absent from the original report. Establishing whether a value was absent
from the form, missed during extraction, or genuinely unknown requires
comparison with source reports or an independently coded reference sample.

## Citation and usage of code
If you use this work for academic work please cite the following paper:

> Alam, M. S., Zhang, L., Li, J., Dou, F., Bazilinskyy, P. (2026). Collision Patterns and Reporting Blind Spots in 970 California Autonomous Vehicle Crash Reports.

The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code. For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.12.13-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.12.13** and the [`uv`](https://docs.astral.sh/uv/) package manager.
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/bazilinskyy/llm-events.git
cd llm-events
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.12.13 installed, let `uv` fetch it:
```command line
uv python install 3.12.13
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Set the `data` value in `config` to the input CSV. The input must
contain a `Report` column and an `Output` column. `Report` identifies the
source document and `Output` contains the complete LLM response.


**Step 9:** Run the code:
```command line
python3 analysis.py
```

### Configuration of project
Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
- **`data`**: Path to the input CSV file containing the LLM generated output to be parsed and analysed.
- **`output_dir`**: Directory where processed CSV files, summary files, and generated plots are saved. All generated figures are saved directly into this directory.
- **`figures_dir`**: Directory where final copies of figures are saved when `save_final` is enabled.
- **`logger_level`**: Logging level used during execution, such as `INFO`, `DEBUG`, `WARNING`, or `ERROR`.
- **`auto_open_html`**: Automatically opens generated HTML plots in the default browser after they are created.
- **`save_final`**: Saves final copies of figures into `figures_dir` in addition to the standard output location.
- **`filter_rows_with_na`**: Filters out rows from the plotting dataset when important parsed fields contain missing values.
- **`na_filter_fields`**: List of parsed fields considered critical for filtering. If any of these fields are missing and `filter_rows_with_na` is enabled, the corresponding row is excluded from the plots.
- **`include_plot_fields`**: Ordered list of parsed fields to include in the overview plots such as the Sankey diagram, sunburst diagram, and transition graph.
- **`exclude_plot_fields`**: List of parsed fields to remove from the configured plot fields, allowing quick experimentation with different figure layouts.
- **`histogram_fields`**: List of parsed fields for which histogram style summary plots are generated.
- **`blind_spot_fields`**: List of detailed context fields used to measure post extraction unavailability. The configuration name is retained for backwards compatibility.
- **`max_categories`**: Maximum number of categories retained per stage in the overview plots before less frequent values are grouped into `Other`.
- **`min_count`**: Minimum count threshold for links in the Sankey diagram. Links below this threshold are excluded from the figure.
- **`row_keep_policy`**: Controls which response rows are kept before parsing. Supported values are `output_only`, `best_available`, and `best_per_row`.
- **`validation_sample_size`**: Number of rows sampled for the validation output table.
- **`validation_seed`**: Random seed used when generating the validation sample.
- **`validation_include_text`**: Whether the validation sample should include the source text used for parsing.
- **`paper_plot_top_n`**: Number of top categories retained in selected paper style plots.


## Results
### Overview figures

#### Accident overview Sankey diagram
[![Accident overview Sankey diagram](figures/accident_overview_sankey.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accident_overview_sankey.html)
Accident overview Sankey diagram showing the flow of parsed accident attributes across the selected plot fields.

#### Accident 5W1H Sankey diagram
[![Accident 5W1H Sankey diagram](figures/accident_5w1h_sankey.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accident_5w1h_sankey.html)
Accident 5W1H Sankey diagram showing the flow across who, where, what, when, and why storyline dimensions.

#### Accident overview sunburst diagram
[![Accident overview sunburst diagram](figures/accident_overview_sunburst.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accident_overview_sunburst.html)
Accident overview sunburst diagram showing the hierarchical distribution of accident attributes across the selected plot fields.

#### Accident transition graph
[![Accident transition graph](figures/accident_transition_graph.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accident_transition_graph.html)
Accident transition graph showing connections between consecutive accident attributes across the selected plot fields.

#### Accident location map
[![Accident location map](figures/accident_location_map.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accident_location_map.html)
Accident location map showing geocoded accident report locations derived from questionnaire address fields.

### Histogram figures

#### Histogram of road user type
[![Histogram of road user type](figures/road_user_type.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/road_user_type.html)
Histogram of road user types extracted from the accident reports.


#### Histogram of collision group
[![Histogram of collision group](figures/collision_group.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/collision_group.html)
Histogram of collision groups extracted from the accident reports.

#### Histogram of blame group
[![Histogram of blame group](figures/blame_group.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/blame_group.html)
Histogram of blame groups extracted from the accident reports.

#### Histogram of scenario class
[![Histogram of scenario class](figures/scenario_class.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/scenario_class.html)
Histogram of scenario classes derived from the accident reports.

### Research figures

#### Taxonomy overview
[![Taxonomy overview](figures/taxonomy_overview.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/taxonomy_overview.html)
Bar chart showing the most frequent scenario classes in the empirical accident subset.

#### Post extraction unavailability
[![Blind spots missingness](figures/blind_spots_missingness.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/blind_spots_missingness.html)
Bar chart showing unavailable values after extraction across selected context
fields. The chart does not identify whether the source report, the form
design, or the extraction process accounts for an unavailable value.

#### Accountability by taxonomy
[![Accountability by taxonomy](figures/accountability_by_taxonomy.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/accountability_by_taxonomy.html)
Chart showing how blame assignments vary across the most common scenario classes.

#### Report completeness
[![Report completeness](figures/report_completeness.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/report_completeness.html)
Figure showing report completeness scores across the parsed accident reports.

#### Taxonomy by road user
[![Taxonomy by road user](figures/taxonomy_by_road_user.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/taxonomy_by_road_user.html)
Chart showing how scenario classes vary by road user type.

#### Provenance availability
[![Provenance availability](figures/provenance_availability.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/provenance_availability.html)
Figure showing mean field availability by provenance source.

#### Context gap
[![Context gap](figures/context_gap.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/context_gap.html)
Figure comparing coarse report context, fine report context, and external
online context availability.

#### Movement field agreement
[![Movement consistency](figures/movement_consistency.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/movement_consistency.html)
Figure showing exact, compatible, contradictory, and unavailable movement
comparisons across checkbox and narrative fields.

#### Scenario rule support
[![Scenario determinability](figures/scenario_determinability.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/scenario_determinability.html)
Figure showing internal rule support for deterministic scenario assignments.
This measure is not a validated classification accuracy score.

#### Environment profile
[![Environment profile](figures/environment_profile.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/environment_profile.html)
Figure showing the distribution of environmental friction profiles across the accident set.

#### Blame confidence alignment
[![Blame confidence alignment](figures/blame_confidence_alignment.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/blame_confidence_alignment.html)
Figure showing alignment between blame assignment and confidence related evidence.

#### Stopped AV subtype
[![Stopped AV subtype](figures/stopped_av_subtype.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/stopped_av_subtype.html)
Figure showing subtype patterns within stopped automated vehicle scenarios.

#### Intersection detail quality
[![Intersection detail quality](figures/intersection_detail_quality.png)](https://htmlpreview.github.io/?https://github.com/bazilinskyy/llm-events/blob/main/figures/intersection_detail_quality.html)
Figure showing the quality and completeness of intersection related detail in the parsed reports.

### Injury status interpretation

`reported_injury_status` is a conservative extraction based category.
`no_injury_marked` means that the extracted injury fields contained an
explicit `None` or no injury marker. It does not establish a clinically
verified absence of injury. `reported_injury` and `reported_fatality` likewise
refer to what was marked in the extracted fields, not an independently
validated medical outcome.
