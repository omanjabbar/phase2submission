from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import compute_best_holdout_configs, compute_performance_summary, keep_key_metrics, save_outputs
from .discovery import discover_runs
from .parsers import parse_runs
from .plots import create_plots
from .report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple df-analyze runs and compare key performance metrics."
    )
    parser.add_argument("--input", required=True, help="Folder that contains df-analyze runs.")
    parser.add_argument("--output", required=True, help="Folder where outputs will be saved.")
    parser.add_argument(
        "--metric",
        default="auroc",
        help="Metric used for the two plots. Example: auroc, acc, f1",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    metric = str(args.metric)

    runs = discover_runs(input_dir)
    performance_df, warnings_df = parse_runs(runs)

    # Keep the full merged data, but focus the comparison on the main metrics.
    key_performance_df = keep_key_metrics(performance_df)

    performance_summary_df = compute_performance_summary(key_performance_df)
    best_holdout_df = compute_best_holdout_configs(key_performance_df)

    saved_csv_paths = save_outputs(
        output_dir=output_dir,
        performance_df=performance_df,
        warnings_df=warnings_df,
        performance_summary_df=performance_summary_df,
        best_holdout_df=best_holdout_df,
    )

    plot_paths = create_plots(key_performance_df, metric, output_dir)

    report_path = write_report(
        output_dir=output_dir,
        run_count=len(runs),
        performance_df=key_performance_df,
        warnings_df=warnings_df,
        performance_summary_df=performance_summary_df,
        best_holdout_df=best_holdout_df,
        plot_paths=plot_paths,
        saved_csv_paths=saved_csv_paths,
    )

    print(f"Runs discovered: {len(runs)}")
    print(f"Runs with parsed performance data: {key_performance_df['run_id'].nunique() if not key_performance_df.empty else 0}")
    print(f"Chosen plot metric: {metric}")
    print(f"Output folder: {output_dir}")
    print(f"Report written to: {report_path}")

    return 0 if not performance_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
