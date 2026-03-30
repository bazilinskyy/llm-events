from __future__ import annotations

import logging

from utils.config import load_runtime_config
from utils.io import ensure_output_dirs, load_input_events, save_dataframe, save_json
from utils.logging_utils import setup_logging
from utils.parsing import parse_events_dataframe
from utils.sankey import apply_plot_filters, build_overview_summary, resolve_plot_fields
from utils.summary_plots import create_all_plots

logger = logging.getLogger(__name__)


def main() -> int:
    config = load_runtime_config()
    setup_logging(config.log_level)

    logger.info('Loaded configuration from %s', config.config_path)
    logger.info('Using input CSV: %s', config.input_csv)
    logger.info('Using row_keep_policy: %s', config.row_keep_policy)
    logger.info('Using save_final: %s', config.save_final)

    output_dirs = ensure_output_dirs(config.output_dir, config.figures_dir)

    raw_df, selected_text_column = load_input_events(
        input_csv=config.input_csv,
        preferred_text_column=config.text_column,
        row_keep_policy=config.row_keep_policy,
    )
    logger.info('Using text column: %s', selected_text_column)

    parsed_df = parse_events_dataframe(raw_df, text_column=selected_text_column)

    if parsed_df.empty:
        logger.warning('No rows remained after loading and parsing. Nothing to plot.')
        return 0

    plot_fields = resolve_plot_fields(parsed_df, config.include_plot_fields, config.exclude_plot_fields)
    filtered_df, filter_report = apply_plot_filters(
        parsed_df,
        plot_fields=plot_fields,
        filter_rows_with_na=config.filter_rows_with_na,
        na_filter_fields=config.na_filter_fields,
    )

    logger.info(
        'Row summary: loaded=%s, dropped_by_row_policy=%s, parsed=%s, dropped_for_plot_na=%s, used_for_plots=%s',
        len(raw_df) + filter_report.get('dropped_empty_output', 0),
        filter_report.get('dropped_empty_output', 0),
        len(parsed_df),
        filter_report.get('dropped_for_plot_na', 0),
        len(filtered_df),
    )

    save_dataframe(parsed_df, output_dirs.base / 'cleaned_events.csv')
    logger.info('Wrote cleaned events to %s', output_dirs.base / 'cleaned_events.csv')

    save_dataframe(filtered_df, output_dirs.base / 'accident_overview.csv')
    logger.info('Wrote accident overview to %s', output_dirs.base / 'accident_overview.csv')

    plot_input_columns = ['report_pdf'] + [field for field in plot_fields if field in filtered_df.columns]
    plot_input_columns = [col for col in plot_input_columns if col in filtered_df.columns]
    save_dataframe(filtered_df[plot_input_columns].copy(), output_dirs.base / 'plot_input_filtered.csv')
    logger.info('Wrote filtered plot input to %s', output_dirs.base / 'plot_input_filtered.csv')

    overview_summary = build_overview_summary(
        filtered_df=filtered_df,
        plot_fields=plot_fields,
        filter_report=filter_report,
    )
    save_json(overview_summary, output_dirs.base / 'accident_overview_summary.json')
    logger.info('Wrote overview summary to %s', output_dirs.base / 'accident_overview_summary.json')

    manifest = create_all_plots(
        parsed_df=parsed_df,
        filtered_df=filtered_df,
        plot_fields=plot_fields,
        output_dirs=output_dirs,
        auto_open_html=config.auto_open_html,
        histogram_fields=config.histogram_fields,
        min_count=config.min_count,
        max_categories=config.max_categories,
        save_final=config.save_final,
    )
    save_json(manifest, output_dirs.base / 'plot_manifest.json')
    logger.info('Wrote plot manifest to %s', output_dirs.base / 'plot_manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
