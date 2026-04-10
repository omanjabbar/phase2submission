from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def create_plots(performance_df: pd.DataFrame, metric: str, output_dir: Path) -> Dict[str, Path]:
    """create two plots for cli chosen metric"""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: dict[str, Path] = {}
    plot_paths["avg_holdout_plot"] = output_dir / f"avg_holdout_{metric}.png"
    plot_paths["holdout_variability_plot"] = output_dir / f"holdout_variability_{metric}.png"

    _plot_average_holdout(performance_df, metric, plot_paths["avg_holdout_plot"])
    _plot_holdout_boxplot(performance_df, metric, plot_paths["holdout_variability_plot"])
    return plot_paths


def _plot_average_holdout(performance_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    df = performance_df.copy()
    df = df[(df["split"] == "holdout") & (df["metric"] == metric)].dropna(subset=["value"])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)

    if df.empty:
        _make_placeholder(ax, f"No holdout data found for metric '{metric}'.")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    summary_df = (
        df.groupby(["model", "selection", "embed_selector"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    summary_df["config"] = summary_df.apply(_make_config_label, axis=1)
    summary_df = summary_df.sort_values("value", ascending=False).reset_index(drop=True)

    ax.bar(summary_df["config"], summary_df["value"])
    ax.set_title(f"Average holdout {metric} by configuration")
    ax.set_xlabel("Configuration")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_holdout_boxplot(performance_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    df = performance_df.copy()
    df = df[(df["split"] == "holdout") & (df["metric"] == metric)].dropna(subset=["value"])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)

    if df.empty:
        _make_placeholder(ax, f"No variability data found for metric '{metric}'.")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    grouped_df = (
        df.groupby(["model", "selection", "embed_selector"], dropna=False)
        .agg(mean_value=("value", "mean"))
        .reset_index()
        .sort_values("mean_value", ascending=False)
    )

    labels: list[str] = []
    value_lists: list[list[float]] = []

    for _, row in grouped_df.iterrows():
        row_mask = (
            (df["model"] == row["model"])
            & (df["selection"] == row["selection"])
            & (df["embed_selector"] == row["embed_selector"])
        )
        row_values = df.loc[row_mask, "value"].dropna().tolist()
        if not row_values:
            continue
        labels.append(_make_config_label(row))
        value_lists.append(row_values)

    if not value_lists:
        _make_placeholder(ax, f"No variability data found for metric '{metric}'.")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    ax.boxplot(value_lists, labels=labels, showfliers=True)
    ax.set_title(f"Holdout {metric} variability across runs")
    ax.set_xlabel("Configuration")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_placeholder(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def _make_config_label(row: pd.Series) -> str:
    return f"{row['model']} | {row['selection']} | {row['embed_selector']}"
