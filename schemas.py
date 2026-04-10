from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunArtifacts:
    """Small container for one df-analyze run."""

    run_id: str
    run_dir: Path
    performance_file: Optional[Path] = None
    report_file: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)
