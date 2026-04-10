from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .schemas import RunArtifacts


REQUIRED_COLUMNS = {
    "metric",
    "trainset",
    "holdout",
    "5-fold",
    "model",
    "selection",
    "embed_selector",
}

CANONICAL_COLUMNS = [
    "run_id",
    "run_dir",
    "source_file",
    "metric",
    "split",
    "value",
    "model",
    "selection",
    "embed_selector",
]


def parse_runs(runs: List[RunArtifacts]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """we parse all discovered runs into one merged performance dataframe"""
    performance_frames: list[pd.DataFrame] = []
    warning_rows: list[dict[str, str]] = []

    for run in runs:
        for warning in run.warnings:
            warning_rows.append(
                {
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "stage": "discovery",
                    "message": warning,
                }
            )

        performance_df, parse_warnings = parse_performance_for_run(run)
        for message in parse_warnings:
            warning_rows.append(
                {
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "stage": "performance",
                    "message": message,
                }
            )

        if not performance_df.empty:
            performance_frames.append(performance_df)

    merged_df = (
        pd.concat(performance_frames, ignore_index=True)
        if performance_frames
        else pd.DataFrame(columns=CANONICAL_COLUMNS)
    )
    warnings_df = pd.DataFrame(warning_rows, columns=["run_id", "run_dir", "stage", "message"])
    return merged_df, warnings_df


def parse_performance_for_run(run: RunArtifacts) -> Tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if run.performance_file is not None:
        try:
            raw_df = pd.read_csv(run.performance_file)
            return normalize_performance_csv(raw_df, run, run.performance_file), warnings
        except Exception as exc:
            warnings.append(f"Failed to parse performance CSV '{run.performance_file.name}': {exc}")

    if run.report_file is not None:
        try:
            report_df = parse_results_report(run.report_file, run)
            if report_df.empty:
                warnings.append("results_report.md was found but no performance rows were extracted.")
            return report_df, warnings
        except Exception as exc:
            warnings.append(f"Failed to parse results_report.md: {exc}")

    warnings.append("Run skipped because no parseable performance data was found .")
    return pd.DataFrame(columns=CANONICAL_COLUMNS), warnings


def normalize_performance_csv(df: pd.DataFrame, run: RunArtifacts, source_file: Path) -> pd.DataFrame:
    """this turn df-analyze performance CSV into one standard long dataframe."""
    df = df.copy()

    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if "selection" not in df.columns:
        df["selection"] = "none"
    if "embed_selector" not in df.columns:
        df["embed_selector"] = "none"

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["metric"] = df["metric"].astype(str)
    df["model"] = df["model"].astype(str)
    df["selection"] = df["selection"].fillna("none").astype(str)
    df["embed_selector"] = df["embed_selector"].fillna("none").astype(str)

    long_df = df[["metric", "model", "selection", "embed_selector", "trainset", "holdout", "5-fold"]].melt(
        id_vars=["metric", "model", "selection", "embed_selector"],
        value_vars=["trainset", "holdout", "5-fold"],
        var_name="split",
        value_name="value",
    )

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["run_id"] = run.run_id
    long_df["run_dir"] = str(run.run_dir)
    long_df["source_file"] = str(source_file)

    long_df = long_df[CANONICAL_COLUMNS].sort_values(
        ["run_id", "metric", "split", "model", "selection", "embed_selector"]
    )
    return long_df.reset_index(drop=True)


def parse_results_report(report_file: Path, run: RunArtifacts) -> pd.DataFrame:
    """markdown-table fallback parser for results_report.md"""
    split_map = {
        "training set performance": "trainset",
        "holdout set performance": "holdout",
        "5-fold performance on holdout set": "5-fold",
    }

    lines = report_file.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, object]] = []
    current_split = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("## "):
            current_split = split_map.get(line[3:].strip().lower())
            i += 1
            continue

        if current_split is not None and line.startswith("model"):
            header = line.split()
            i += 1

            if i < len(lines):
                divider = lines[i].strip().replace(" ", "")
                if divider and set(divider) == {"-"}:
                    i += 1

            while i < len(lines):
                row_line = lines[i].strip()
                if not row_line or row_line.startswith("## ") or row_line.startswith("# "):
                    break

                tokens = row_line.split()
                if len(tokens) != len(header):
                    i += 1
                    continue

                model = tokens[0]
                selection = tokens[1]
                embed_selector = tokens[2]

                for j in range(3, len(header)):
                    rows.append(
                        {
                            "run_id": run.run_id,
                            "run_dir": str(run.run_dir),
                            "source_file": str(report_file),
                            "metric": header[j],
                            "split": current_split,
                            "value": _safe_float(tokens[j]),
                            "model": model,
                            "selection": selection,
                            "embed_selector": embed_selector,
                        }
                    )
                i += 1
            continue

        i += 1

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _safe_float(value: str) -> float:
    value_text = str(value).strip().lower()
    if value_text in {"", "nan", "none"}:
        return float("nan")
    return float(value)
