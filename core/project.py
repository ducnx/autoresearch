"""Project-level contract for autonomous research runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_RUN_COMMAND = ["uv", "run", "python", "{entrypoint}"]


@dataclass
class ProjectSpec:
    """Configuration loaded from an optional ``project.json`` file."""

    name: str
    project_dir: Path
    description: str = ""
    entrypoint: str = "train.py"
    run_command: list[str] = field(default_factory=lambda: DEFAULT_RUN_COMMAND.copy())
    editable_files: list[str] = field(default_factory=lambda: ["train.py"])
    context_files: list[str] = field(default_factory=list)
    package_paths: list[str] = field(default_factory=list)
    data_paths: list[str] = field(default_factory=list)
    writing_dir: str | None = None
    result_section: str = "result.lyx"
    metric_name: str = "val_bpb"
    metric_direction: str = "minimize"

    @classmethod
    def load(cls, project: str, project_dir: Path) -> "ProjectSpec":
        spec_path = project_dir / "project.json"
        if not spec_path.exists():
            return cls(name=project, project_dir=project_dir)

        raw: dict[str, Any] = json.loads(spec_path.read_text())
        metric = raw.get("metric", {})
        writing = raw.get("writing", {})
        return cls(
            name=project,
            project_dir=project_dir,
            description=raw.get("description", ""),
            entrypoint=raw.get("entrypoint", "train.py"),
            run_command=raw.get("run_command", DEFAULT_RUN_COMMAND.copy()),
            editable_files=raw.get("editable_files", ["train.py"]),
            context_files=raw.get("context_files", []),
            package_paths=raw.get("package_paths", []),
            data_paths=raw.get("data_paths", []),
            writing_dir=writing.get("dir"),
            result_section=writing.get("result_section", "result.lyx"),
            metric_name=metric.get("name", "val_bpb"),
            metric_direction=metric.get("direction", "minimize"),
        )

    @property
    def primary_file(self) -> str:
        return self.editable_files[0] if self.editable_files else self.entrypoint

    @property
    def higher_is_better(self) -> bool:
        return self.metric_direction.lower() in {"maximize", "max", "higher"}

    def resolved_run_command(self) -> list[str]:
        substitutions = {
            "entrypoint": self.entrypoint,
            "project": self.name,
            "metric_name": self.metric_name,
        }
        return [part.format(**substitutions) for part in self.run_command]

    def resolved_package_paths(self) -> list[str]:
        return [str((self.project_dir / path).resolve()) for path in self.package_paths]

    def existing_context_files(self) -> list[str]:
        paths = []
        seen = set()
        for path in [*self.editable_files, *self.context_files]:
            if path in seen:
                continue
            seen.add(path)
            if (self.project_dir / path).exists():
                paths.append(path)
        return paths
