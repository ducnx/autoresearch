"""
Experiment runner — manages subprocess execution of configured project commands.

Handles:
  - Launching training as a subprocess
  - Timeout management (kills after 10 minutes)
  - Output parsing (primary metric, peak_vram_mb, etc.)
  - Crash detection and log capture
  - Dry-run mode for testing without GPU
"""

import os
import re
import random
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    """Raw output from a training run."""
    success: bool
    val_bpb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    total_tokens_m: float = 0.0
    num_steps: int = 0
    num_params_m: float = 0.0
    depth: int = 0
    error_message: Optional[str] = None
    log_path: Optional[str] = None
    metric_name: str = "val_bpb"
    metric_value: float = 0.0
    metric_direction: str = "minimize"
    extra_metrics: dict[str, float] = field(default_factory=dict)


def _parse_summary(log_content: str) -> dict:
    """Parse the training summary block from log output."""
    metrics: dict[str, float | int] = {}
    # Look for the summary block after "---"
    summary_match = re.search(r"^---\s*$(.+)", log_content, re.MULTILINE | re.DOTALL)
    if not summary_match:
        return metrics

    summary_text = summary_match.group(1)
    patterns = {
        "val_bpb": r"val_bpb:\s+([\d.]+)",
        "training_seconds": r"training_seconds:\s+([\d.]+)",
        "total_seconds": r"total_seconds:\s+([\d.]+)",
        "peak_vram_mb": r"peak_vram_mb:\s+([\d.]+)",
        "mfu_percent": r"mfu_percent:\s+([\d.]+)",
        "total_tokens_m": r"total_tokens_M:\s+([\d.]+)",
        "num_steps": r"num_steps:\s+(\d+)",
        "num_params_m": r"num_params_M:\s+([\d.]+)",
        "depth": r"depth:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, summary_text)
        if match:
            val = match.group(1)
            if key in ("num_steps", "depth"):
                metrics[key] = int(val)
            else:
                metrics[key] = float(val)

    for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", summary_text, re.MULTILINE):
        key, raw_val = match.groups()
        if key not in metrics:
            metrics[key] = float(raw_val)
    return metrics


def run_experiment(
    project_dir: Path,
    workspace_dir: Path,
    experiment_id: int,
    timeout: int = 600,
    dry_run: bool = False,
    command: Optional[list[str]] = None,
    metric_name: str = "val_bpb",
    metric_direction: str = "minimize",
    package_paths: Optional[list[str]] = None,
) -> RunResult:
    """
    Run a training experiment.

    Args:
        project_dir: Path to the project directory
        workspace_dir: Path to workspace (for log storage)
        experiment_id: Unique experiment identifier
        timeout: Maximum wall-clock seconds before kill
        dry_run: If True, simulate results without running the project command

    Returns:
        RunResult with parsed metrics or error information
    """
    metric_direction = "maximize" if metric_direction.lower() in {"maximize", "max", "higher"} else "minimize"

    if dry_run:
        return _dry_run_experiment(experiment_id, metric_name=metric_name, metric_direction=metric_direction)

    log_dir = workspace_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"experiment_{experiment_id:04d}.log"

    try:
        # Launch training subprocess
        run_command = command or ["uv", "run", "python", "train.py"]
        env = os.environ.copy()
        if package_paths:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join([*package_paths, existing]) if existing else os.pathsep.join(package_paths)
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                run_command,
                cwd=str(project_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # Create process group for clean kill
            )

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                return RunResult(
                    success=False,
                    error_message=f"Experiment timed out after {timeout}s",
                    log_path=str(log_path),
                )

        # Check return code
        if process.returncode != 0:
            # Read tail of log for error info
            log_content = log_path.read_text()
            tail = "\n".join(log_content.split("\n")[-50:])
            return RunResult(
                success=False,
                error_message=f"Process exited with code {process.returncode}:\n{tail}",
                log_path=str(log_path),
            )

        # Parse results
        log_content = log_path.read_text()
        metrics = _parse_summary(log_content)

        if not metrics or metric_name not in metrics:
            tail = "\n".join(log_content.split("\n")[-50:])
            return RunResult(
                success=False,
                error_message=f"Could not parse metric '{metric_name}' from output:\n{tail}",
                log_path=str(log_path),
                metric_name=metric_name,
                metric_direction=metric_direction,
            )

        metric_value = float(metrics[metric_name])
        optimization_value = metric_value if metric_direction == "minimize" else -metric_value
        metrics.setdefault("val_bpb", optimization_value)

        known_metric_fields = {
            key: metrics[key]
            for key in [
                "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
                "mfu_percent", "total_tokens_m", "num_steps", "num_params_m", "depth",
            ]
            if key in metrics
        }

        return RunResult(
            success=True,
            log_path=str(log_path),
            metric_name=metric_name,
            metric_value=metric_value,
            metric_direction=metric_direction,
            extra_metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            **known_metric_fields,
        )

    except Exception as e:
        return RunResult(
            success=False,
            error_message=f"Unexpected error: {e}",
            log_path=str(log_path) if log_path.exists() else None,
            metric_name=metric_name,
            metric_direction=metric_direction,
        )


def _dry_run_experiment(experiment_id: int, metric_name: str = "val_bpb", metric_direction: str = "minimize") -> RunResult:
    """Simulate an experiment result for testing without GPU."""
    # Simulate realistic-ish results with some variance
    base_bpb = 1.05 - experiment_id * 0.003  # gradual improvement
    noise = random.gauss(0, 0.01)
    val_bpb = max(0.85, base_bpb + noise)

    # Small chance of a "crash"
    if random.random() < 0.05:
        return RunResult(
            success=False,
            error_message="[dry-run] Simulated OOM crash",
            metric_name=metric_name,
            metric_direction=metric_direction,
        )

    time.sleep(2)  # Brief pause to simulate work

    metric_value = val_bpb if metric_direction == "minimize" else max(0.0, 1.0 - val_bpb)
    return RunResult(
        success=True,
        val_bpb=round(val_bpb, 6),
        metric_name=metric_name,
        metric_value=round(metric_value, 6),
        metric_direction=metric_direction,
        extra_metrics={metric_name: round(metric_value, 6), "val_bpb": round(val_bpb, 6)},
        training_seconds=300.0,
        total_seconds=round(300 + random.uniform(10, 30), 1),
        peak_vram_mb=round(44000 + random.uniform(-2000, 2000), 1),
        mfu_percent=round(35 + random.uniform(-5, 5), 2),
        total_tokens_m=round(450 + random.uniform(-50, 50), 1),
        num_steps=random.randint(800, 1100),
        num_params_m=round(50 + random.uniform(-5, 5), 1),
        depth=8,
    )
