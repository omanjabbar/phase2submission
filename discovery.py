from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .schemas import RunArtifacts


RESULTS_DIR_NAME = "results"
PERFORMANCE_CANDIDATES = ("performance_long_table.csv", "final_performances.csv")
REPORT_CANDIDATE = "results_report.md"


def discover_runs(input_dir: Path) -> List[RunArtifacts]:
    """Find hashed df-analyze run roots under the given folder."""
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.exists():
        return []

    runs: list[RunArtifacts] = []

    for current_dir, dirnames, _ in os.walk(input_dir, topdown=True):
        current_path = Path(current_dir)
        results_dir = current_path / RESULTS_DIR_NAME

        #a real run root has a results/ folder.
        if not results_dir.is_dir():
            continue

        performance_file = None
        for file_name in PERFORMANCE_CANDIDATES:
            candidate = results_dir / file_name
            if candidate.is_file():
                performance_file = candidate
                break

        report_file = results_dir / REPORT_CANDIDATE
        if not report_file.is_file():
            report_file = None

        warnings: list[str] = []
        if performance_file is None and report_file is None:
            warnings.append("No performance file or results report found in results/.")

        runs.append(
            RunArtifacts(
                run_id=current_path.name,
                run_dir=current_path,
                performance_file=performance_file,
                report_file=report_file,
                warnings=warnings,
            )
        )

        #stop walking deeper inside this run so nested folders are not treated as runs.
        dirnames[:] = []

    runs.sort(key=lambda item: item.run_id.lower())
    return runs
