"""
Experiment runner — manages subprocess execution of train.py.

Handles:
  - Launching training as a subprocess
  - Timeout management (kills after 10 minutes)
  - Output parsing (val_bpb, peak_vram_mb, etc.)
  - Crash detection and log capture
  - Dry-run mode for testing without GPU
"""

import os
import re
import random
import signal
import subprocess
import time
from dataclasses import dataclass
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


def _parse_summary(log_content: str) -> dict:
    """Parse the training summary block from log output."""
    metrics = {}
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
    return metrics


def run_experiment(
    project_root: Path,
    workspace_dir: Path,
    experiment_id: int,
    timeout: int = 600,
    dry_run: bool = False,
) -> RunResult:
    """
    Run a training experiment.

    Args:
        project_root: Path to the project root (where train.py lives)
        workspace_dir: Path to workspace (for log storage)
        experiment_id: Unique experiment identifier
        timeout: Maximum wall-clock seconds before kill
        dry_run: If True, simulate results without running train.py

    Returns:
        RunResult with parsed metrics or error information
    """
    if dry_run:
        return _dry_run_experiment(experiment_id)

    log_dir = workspace_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"experiment_{experiment_id:04d}.log"

    try:
        # Launch training subprocess
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                ["uv", "run", "train.py"],
                cwd=str(project_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
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

        if not metrics or "val_bpb" not in metrics:
            tail = "\n".join(log_content.split("\n")[-50:])
            return RunResult(
                success=False,
                error_message=f"Could not parse val_bpb from output:\n{tail}",
                log_path=str(log_path),
            )

        return RunResult(
            success=True,
            log_path=str(log_path),
            **metrics,
        )

    except Exception as e:
        return RunResult(
            success=False,
            error_message=f"Unexpected error: {e}",
            log_path=str(log_path) if log_path.exists() else None,
        )


def _dry_run_experiment(experiment_id: int) -> RunResult:
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
        )

    time.sleep(2)  # Brief pause to simulate work

    return RunResult(
        success=True,
        val_bpb=round(val_bpb, 6),
        training_seconds=300.0,
        total_seconds=round(300 + random.uniform(10, 30), 1),
        peak_vram_mb=round(44000 + random.uniform(-2000, 2000), 1),
        mfu_percent=round(35 + random.uniform(-5, 5), 2),
        total_tokens_m=round(450 + random.uniform(-50, 50), 1),
        num_steps=random.randint(800, 1100),
        num_params_m=round(50 + random.uniform(-5, 5), 1),
        depth=8,
    )
