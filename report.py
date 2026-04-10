from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


KEY_METRICS = ["acc", "auroc", "f1"]


def write_report(
    output_dir: Path,
    run_count: int,
    performance_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    performance_summary_df: pd.DataFrame,
    best_holdout_df: pd.DataFrame,
    plot_paths: Dict[str, Path],
    saved_csv_paths: Dict[str, Path],
) -> Path:
    """Create a short markdown report for."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    metrics_available = sorted(performance_df["metric"].dropna().astype(str).unique().tolist()) if not performance_df.empty else []
    runs_with_data = int(performance_df["run_id"].nunique()) if not performance_df.empty else 0

    key_summary_df = performance_summary_df.loc[performance_summary_df["metric"].isin(KEY_METRICS)].copy()
    if key_summary_df.empty:
        key_summary_df = performance_summary_df.copy()

    key_best_df = best_holdout_df.loc[best_holdout_df["metric"].isin(KEY_METRICS)].copy()
    if key_best_df.empty:
        key_best_df = best_holdout_df.copy()

    lines: list[str] = []
    lines.append("# dfa_compare report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Runs discovered: {run_count}")
    lines.append(f"- Runs with parsed performance data: {runs_with_data}")
    lines.append(f"- Total performance rows: {len(performance_df)}")
    lines.append(f"- Metrics available: {', '.join(metrics_available) if metrics_available else 'None'}")
    lines.append(f"- Warning count: {len(warnings_df)}")
    lines.append("")

    lines.append("## Best holdout configurations")
    lines.append("")
    if key_best_df.empty:
        lines.append("No best holdout configurations could be computed.")
    else:
        display_df = key_best_df[["metric", "model", "selection", "embed_selector", "mean", "std", "count"]].copy()
        lines.extend(_df_to_markdown(display_df))
    lines.append("")

    lines.append("## Performance summary for key metrics")
    lines.append("")
    if key_summary_df.empty:
        lines.append("No summary table could be computed.")
    else:
        display_df = key_summary_df[["metric", "split", "model", "selection", "embed_selector", "mean", "std", "min", "max", "count"]].copy()
        lines.extend(_df_to_markdown(display_df.head(25)))
    lines.append("")

    lines.append("## Generated plots")
    lines.append("")
    for name, path in plot_paths.items():
        lines.append(f"- {name}: `{path.name}`")
    lines.append("")

    lines.append("## Generated CSV files")
    lines.append("")
    for name, path in saved_csv_paths.items():
        lines.append(f"- {name}: `{path.name}`")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if warnings_df.empty:
        lines.append("No warnings.")
    else:
        lines.extend(_df_to_markdown(warnings_df[["run_id", "stage", "message"]].head(50)))
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _df_to_markdown(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows."]

    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(_format_value)

    lines: list[str] = []
    headers = [str(col) for col in safe_df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in safe_df.iterrows():
        cells = [str(row[col]) for col in safe_df.columns]
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\n", " ").replace("|", "/")
