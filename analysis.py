from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


KEY_METRICS = {"acc", "auroc", "f1"}


def compute_performance_summary(performance_df: pd.DataFrame) -> pd.DataFrame:
    """compute simple summary stats across runs."""
    if performance_df.empty:
        return pd.DataFrame(
            columns=["metric", "split", "model", "selection", "embed_selector", "count", "mean", "std", "min", "max"]
        )

    summary_df = (
        performance_df.groupby(["metric", "split", "model", "selection", "embed_selector"], dropna=False)["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values(["metric", "split", "mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return summary_df


def compute_best_holdout_configs(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Pick the best holdout configuration for each metric using mean value."""
    if performance_df.empty:
        return pd.DataFrame(
            columns=["metric", "model", "selection", "embed_selector", "count", "mean", "std", "min", "max"]
        )

    holdout_df = performance_df.loc[performance_df["split"] == "holdout"].copy()
    holdout_df = holdout_df.dropna(subset=["value"])

    if holdout_df.empty:
        return pd.DataFrame(
            columns=["metric", "model", "selection", "embed_selector", "count", "mean", "std", "min", "max"]
        )

    grouped_df = (
        holdout_df.groupby(["metric", "model", "selection", "embed_selector"], dropna=False)["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )

    grouped_df = grouped_df.sort_values(
        ["metric", "mean", "count", "model", "selection", "embed_selector"],
        ascending=[True, False, False, True, True, True],
    )

    best_df = grouped_df.groupby("metric", as_index=False).first()
    return best_df.reset_index(drop=True)


def keep_key_metrics(performance_df: pd.DataFrame) -> pd.DataFrame:
    if performance_df.empty:
        return performance_df.copy()

    key_df = performance_df.loc[performance_df["metric"].isin(KEY_METRICS)].copy()
    if key_df.empty:
        return performance_df.copy()
    return key_df


def save_outputs(
    output_dir: Path,
    performance_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    performance_summary_df: pd.DataFrame,
    best_holdout_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Write all final CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}

    output_paths["merged_performance_csv"] = output_dir / "merged_performance_long.csv"
    performance_df.to_csv(output_paths["merged_performance_csv"], index=False)

    output_paths["performance_summary_csv"] = output_dir / "performance_summary.csv"
    performance_summary_df.to_csv(output_paths["performance_summary_csv"], index=False)

    output_paths["best_holdout_csv"] = output_dir / "best_holdout_configs.csv"
    best_holdout_df.to_csv(output_paths["best_holdout_csv"], index=False)

    output_paths["warnings_csv"] = output_dir / "warnings.csv"
    warnings_df.to_csv(output_paths["warnings_csv"], index=False)

    return output_paths
