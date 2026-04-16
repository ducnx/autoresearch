"""
Experiment Agent — modifies code and runs experiments.

Translates hypotheses into concrete train.py changes,
manages git operations, and invokes the experiment runner.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Optional

from agents.base import BaseAgent
from core.workspace import Hypothesis, ExperimentResult
from core.runner import run_experiment


class ExperimentAgent(BaseAgent):
    name = "experiment"
    role = "Experiment Agent — implements and runs experiments"

    def _read_train_py(self) -> str:
        """Read the current train.py content."""
        train_path = self.config.project_dir / self.config.train_script
        return train_path.read_text()

    def _write_train_py(self, content: str):
        """Write modified train.py content."""
        train_path = self.config.project_dir / self.config.train_script
        train_path.write_text(content)

    def _git_commit(self, message: str) -> str:
        """Commit current changes and return the short hash."""
        cwd = str(self.config.project_dir)
        subprocess.run(
            ["git", "add", self.config.train_script],
            cwd=cwd, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=cwd, capture_output=True, check=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    def _git_revert(self):
        """Revert the last commit (soft reset)."""
        cwd = str(self.config.project_dir)
        subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=cwd, capture_output=True, check=True,
        )

    def _git_diff(self) -> str:
        """Get the current diff."""
        cwd = str(self.config.project_dir)
        result = subprocess.run(
            ["git", "diff", self.config.train_script],
            cwd=cwd, capture_output=True, text=True,
        )
        return result.stdout

    def implement_hypothesis(self, hypothesis: Hypothesis, train_code: str) -> str:
        """
        Use LLM to implement the hypothesis as code changes.

        Args:
            hypothesis: The hypothesis to implement
            train_code: Current train.py content

        Returns:
            Modified train.py content
        """
        self._log(f"Implementing hypothesis: {hypothesis.description}")

        prompt = (
            f"## Hypothesis to Implement\n"
            f"**{hypothesis.description}**\n"
            f"Rationale: {hypothesis.rationale}\n"
            f"Category: {hypothesis.category}\n"
            f"Complexity: {hypothesis.complexity}\n\n"
            f"## Current train.py\n```python\n{train_code}\n```\n\n"
            f"Implement this hypothesis by modifying the code above. "
            f"Return the COMPLETE modified train.py file content.\n\n"
            f"Rules:\n"
            f"- Only modify train.py content — do not change imports from prepare.py\n"
            f"- Keep changes minimal and focused on the hypothesis\n"
            f"- Ensure the code is syntactically valid Python\n"
            f"- Preserve all existing functionality unless explicitly changing it\n\n"
            f"Respond with JSON: {{\"modified_code\": \"...full file content...\", \"changes_summary\": \"brief description\"}}"
        )

        messages = self._build_messages(prompt)

        try:
            result = self._call_llm_json(messages, temperature=0.3, max_tokens=8192)
            modified_code = result.get("modified_code", "")
            changes_summary = result.get("changes_summary", hypothesis.description)

            if not modified_code or len(modified_code) < 100:
                raise ValueError("LLM returned empty or too-short modified code")

            self._log(f"Code changes: {changes_summary}")
            return modified_code

        except Exception as e:
            self._log(f"Error implementing hypothesis: {e}")
            raise

    def run_experiment(
        self,
        hypothesis: Hypothesis,
        experiment_id: int,
    ) -> ExperimentResult:
        """
        Full experiment cycle: implement → commit → run → collect results.

        Args:
            hypothesis: The hypothesis to test
            experiment_id: Unique experiment identifier

        Returns:
            ExperimentResult with all metrics
        """
        self._log(f"Starting experiment {experiment_id}: {hypothesis.description}")

        # Read current code
        original_code = self._read_train_py()

        try:
            # Implement the hypothesis (skip for baseline run)
            if experiment_id > 0:
                modified_code = self.implement_hypothesis(hypothesis, original_code)
                self._write_train_py(modified_code)
            else:
                self._log("Running baseline (no code changes)")

            # Git commit
            commit_msg = f"experiment {experiment_id}: {hypothesis.description}"
            commit_hash = self._git_commit(commit_msg)
            self._log(f"Committed: {commit_hash}")

            # Get diff for logging
            code_diff = None
            if experiment_id > 0:
                # The diff is between HEAD~1 and HEAD
                cwd = str(self.config.project_dir)
                diff_result = subprocess.run(
                    ["git", "diff", "HEAD~1", "HEAD", "--", self.config.train_script],
                    cwd=cwd, capture_output=True, text=True,
                )
                code_diff = diff_result.stdout

            # Run the experiment
            run_result = run_experiment(
                project_dir=self.config.project_dir,
                workspace_dir=self.config.workspace_dir,
                experiment_id=experiment_id,
                timeout=self.config.experiment_timeout,
                dry_run=self.config.dry_run,
            )

            if run_result.success:
                result = ExperimentResult(
                    experiment_id=experiment_id,
                    commit_hash=commit_hash,
                    hypothesis_id=hypothesis.id,
                    description=hypothesis.description,
                    val_bpb=run_result.val_bpb,
                    peak_vram_mb=run_result.peak_vram_mb,
                    training_seconds=run_result.training_seconds,
                    total_seconds=run_result.total_seconds,
                    mfu_percent=run_result.mfu_percent,
                    total_tokens_m=run_result.total_tokens_m,
                    num_steps=run_result.num_steps,
                    num_params_m=run_result.num_params_m,
                    depth=run_result.depth,
                    status="pending",  # Director will set keep/discard
                    code_diff=code_diff,
                )
                self._log(f"Experiment {experiment_id} completed: val_bpb={run_result.val_bpb:.6f}")
            else:
                result = ExperimentResult.crash(
                    experiment_id=experiment_id,
                    commit_hash=commit_hash,
                    hypothesis_id=hypothesis.id,
                    description=hypothesis.description,
                    error_message=run_result.error_message or "Unknown error",
                )
                self._log(f"Experiment {experiment_id} crashed: {run_result.error_message}")

            return result

        except Exception as e:
            self._log(f"Error in experiment {experiment_id}: {e}")
            # Try to restore original code
            try:
                self._write_train_py(original_code)
            except Exception:
                pass

            return ExperimentResult.crash(
                experiment_id=experiment_id,
                commit_hash="unknown",
                hypothesis_id=hypothesis.id,
                description=hypothesis.description,
                error_message=str(e),
            )

    def revert_experiment(self):
        """Revert the last experiment (git reset)."""
        try:
            self._git_revert()
            self._log("Reverted last experiment")
        except Exception as e:
            self._log(f"Error reverting: {e}")

    def run(self, hypothesis: Hypothesis, experiment_id: int, **kwargs) -> ExperimentResult:
        """Main entry point."""
        return self.run_experiment(hypothesis, experiment_id)
