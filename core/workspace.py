"""
Shared workspace for inter-agent communication.

Inspired by PaperOrchestra's decoupled pipeline — agents communicate
through structured JSON files in a shared workspace rather than direct
message passing. This enables parallel execution and easy debugging.
"""

import json
import fcntl
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    experiment_id: int
    commit_hash: str
    hypothesis_id: str
    description: str
    val_bpb: float
    peak_vram_mb: float
    training_seconds: float
    total_seconds: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int
    status: str  # "keep", "discard", "crash"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    code_diff: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def crash(cls, experiment_id: int, commit_hash: str, hypothesis_id: str,
              description: str, error_message: str) -> "ExperimentResult":
        """Create a crash result."""
        return cls(
            experiment_id=experiment_id,
            commit_hash=commit_hash,
            hypothesis_id=hypothesis_id,
            description=description,
            val_bpb=0.0,
            peak_vram_mb=0.0,
            training_seconds=0.0,
            total_seconds=0.0,
            mfu_percent=0.0,
            total_tokens_m=0.0,
            num_steps=0,
            num_params_m=0.0,
            depth=0,
            status="crash",
            error_message=error_message,
        )


@dataclass
class Hypothesis:
    """A research hypothesis to test."""
    id: str
    description: str
    predicted_impact: str  # "high", "medium", "low"
    complexity: str  # "simple", "moderate", "complex"
    risk: str  # "low", "medium", "high"
    category: str  # "architecture", "optimizer", "hyperparameter", "training"
    rationale: str
    code_changes: Optional[str] = None  # suggested code changes
    source: str = "hypothesis_agent"  # "hypothesis_agent" or "literature_agent"
    status: str = "pending"  # "pending", "selected", "tested", "discarded"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LiteratureEntry:
    """A technique or finding from literature search."""
    title: str
    source: str  # URL or paper reference
    summary: str
    technique: str
    applicability: str  # "direct", "needs_adaptation", "inspirational"
    relevance_score: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LiteratureEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Workspace:
    """
    Manages the shared workspace for inter-agent communication.

    File structure:
        workspace/
        ├── state.json          — Current research state
        ├── results.json        — All experiment results
        ├── hypotheses.json     — Hypothesis queue
        ├── literature.json     — Collected literature/techniques
        ├── logs/               — Agent activity logs
        └── reports/            — Generated reports and plots
    """

    def __init__(self, workspace_dir: Path):
        self.root = Path(workspace_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        (self.root / "reports").mkdir(exist_ok=True)
        (self.root / "run_logs").mkdir(exist_ok=True)
        self._init_files()

    def _init_files(self):
        """Initialize workspace files if they don't exist."""
        defaults = {
            "state.json": {
                "run_tag": "",
                "branch": "",
                "baseline_bpb": None,
                "best_bpb": None,
                "best_commit": None,
                "experiment_count": 0,
                "current_phase": "setup",
                "started_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "results.json": [],
            "hypotheses.json": [],
            "literature.json": [],
        }
        for filename, default in defaults.items():
            path = self.root / filename
            if not path.exists():
                self._write_json(path, default)

    # --- Thread-safe JSON I/O ---

    def _read_json(self, path: Path) -> Any:
        """Read JSON file with shared lock."""
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write_json(self, path: Path, data: Any):
        """Write JSON file with exclusive lock."""
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2, default=str)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # --- State Management ---

    def get_state(self) -> dict:
        """Get current research state."""
        return self._read_json(self.root / "state.json")

    def update_state(self, **kwargs):
        """Update research state fields."""
        state = self.get_state()
        state.update(kwargs)
        state["last_updated"] = datetime.now().isoformat()
        self._write_json(self.root / "state.json", state)

    # --- Experiment Results ---

    def get_results(self) -> list[ExperimentResult]:
        """Get all experiment results."""
        data = self._read_json(self.root / "results.json")
        return [ExperimentResult.from_dict(d) for d in data]

    def add_result(self, result: ExperimentResult):
        """Add an experiment result."""
        data = self._read_json(self.root / "results.json")
        data.append(result.to_dict())
        self._write_json(self.root / "results.json", data)

        # Update state
        state = self.get_state()
        state["experiment_count"] = len(data)
        if result.status == "keep":
            if state["best_bpb"] is None or result.val_bpb < state["best_bpb"]:
                state["best_bpb"] = result.val_bpb
                state["best_commit"] = result.commit_hash
        if state["baseline_bpb"] is None and result.experiment_id == 0:
            state["baseline_bpb"] = result.val_bpb
        state["last_updated"] = datetime.now().isoformat()
        self._write_json(self.root / "state.json", state)

    def get_best_result(self) -> Optional[ExperimentResult]:
        """Get the best experiment result so far."""
        results = self.get_results()
        kept = [r for r in results if r.status == "keep"]
        if not kept:
            return None
        return min(kept, key=lambda r: r.val_bpb)

    # --- Hypotheses ---

    def get_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses."""
        data = self._read_json(self.root / "hypotheses.json")
        return [Hypothesis.from_dict(d) for d in data]

    def get_pending_hypotheses(self) -> list[Hypothesis]:
        """Get pending (untested) hypotheses."""
        return [h for h in self.get_hypotheses() if h.status == "pending"]

    def add_hypotheses(self, hypotheses: list[Hypothesis]):
        """Add hypotheses to the queue."""
        data = self._read_json(self.root / "hypotheses.json")
        data.extend([h.to_dict() for h in hypotheses])
        self._write_json(self.root / "hypotheses.json", data)

    def update_hypothesis_status(self, hypothesis_id: str, status: str):
        """Update the status of a hypothesis."""
        data = self._read_json(self.root / "hypotheses.json")
        for h in data:
            if h["id"] == hypothesis_id:
                h["status"] = status
                break
        self._write_json(self.root / "hypotheses.json", data)

    # --- Literature ---

    def get_literature(self) -> list[LiteratureEntry]:
        """Get all literature entries."""
        data = self._read_json(self.root / "literature.json")
        return [LiteratureEntry.from_dict(d) for d in data]

    def add_literature(self, entries: list[LiteratureEntry]):
        """Add literature entries."""
        data = self._read_json(self.root / "literature.json")
        data.extend([e.to_dict() for e in entries])
        self._write_json(self.root / "literature.json", data)

    # --- Logging ---

    def log_agent_activity(self, agent_name: str, message: str, data: Optional[dict] = None):
        """Log agent activity to a timestamped log file."""
        log_file = self.root / "logs" / f"{agent_name}.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "message": message,
        }
        if data:
            entry["data"] = data
        with open(log_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, default=str) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_agent_logs(self, agent_name: str, last_n: int = 50) -> list[dict]:
        """Get recent agent logs."""
        log_file = self.root / "logs" / f"{agent_name}.jsonl"
        if not log_file.exists():
            return []
        with open(log_file, "r") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-last_n:]]

    # --- Convenience ---

    @property
    def experiment_count(self) -> int:
        return self.get_state()["experiment_count"]

    def get_results_summary(self) -> str:
        """Get a text summary of all results for agent context."""
        results = self.get_results()
        if not results:
            return "No experiments have been run yet."

        lines = ["Experiment History:"]
        lines.append(f"{'ID':>4} | {'Status':<8} | {'val_bpb':>10} | {'VRAM_GB':>8} | Description")
        lines.append("-" * 80)
        for r in results:
            vram_gb = f"{r.peak_vram_mb / 1024:.1f}" if r.peak_vram_mb > 0 else "N/A"
            lines.append(
                f"{r.experiment_id:>4} | {r.status:<8} | {r.val_bpb:>10.6f} | {vram_gb:>8} | {r.description}"
            )

        state = self.get_state()
        if state["best_bpb"] is not None:
            lines.append(f"\nBest val_bpb: {state['best_bpb']:.6f} (commit: {state['best_commit']})")
        if state["baseline_bpb"] is not None:
            lines.append(f"Baseline val_bpb: {state['baseline_bpb']:.6f}")

        return "\n".join(lines)
