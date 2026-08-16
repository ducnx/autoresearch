"""
Experiment Agent — modifies code and runs experiments.

Translates hypotheses into concrete project-file changes,
manages git operations, and invokes the experiment runner.
"""

import subprocess

from agents.base import BaseAgent
from core.workspace import Hypothesis, ExperimentResult
from core.runner import run_experiment


class ExperimentAgent(BaseAgent):
    name = "experiment"
    role = "Experiment Agent — implements and runs experiments"

    def _editable_files(self) -> list[str]:
        return self.config.project_spec.editable_files

    def read_project_context(self) -> str:
        """Read configured project files for hypothesis and implementation context."""
        sections = []
        for rel_path in self.config.project_spec.existing_context_files():
            path = self.config.project_dir / rel_path
            sections.append(f"## {rel_path}\n```python\n{path.read_text()}\n```")
        return "\n\n".join(sections)

    def _read_editable_files(self) -> dict[str, str]:
        contents = {}
        for rel_path in self._editable_files():
            path = self.config.project_dir / rel_path
            if path.exists():
                contents[rel_path] = path.read_text()
        return contents

    def _write_project_files(self, files: dict[str, str]):
        """Write modified project files under the configured project directory."""
        for rel_path, content in files.items():
            if rel_path not in self._editable_files():
                raise ValueError(f"Refusing to edit non-configured file: {rel_path}")
            path = self.config.project_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def _git_commit(self, message: str) -> str:
        """Commit current changes and return the short hash."""
        cwd = str(self.config.project_dir)
        subprocess.run(
            ["git", "add", *self._editable_files()],
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
        """Revert the last experiment commit without deleting unrelated work."""
        cwd = str(self.config.project_dir)
        subprocess.run(
            ["git", "revert", "--no-edit", "HEAD"],
            cwd=cwd, capture_output=True, check=True,
        )

    def _git_diff(self) -> str:
        """Get the current diff."""
        cwd = str(self.config.project_dir)
        result = subprocess.run(
            ["git", "diff", "--", *self._editable_files()],
            cwd=cwd, capture_output=True, text=True,
        )
        return result.stdout

    def implement_hypothesis(self, hypothesis: Hypothesis, project_files: dict[str, str]) -> dict[str, str]:
        """
        Use LLM to implement the hypothesis as configured project-file changes.

        Args:
            hypothesis: The hypothesis to implement
            project_files: Current editable file contents

        Returns:
            Modified file contents keyed by project-relative path
        """
        self._log(f"Implementing hypothesis: {hypothesis.description}")

        file_blocks = "\n\n".join(
            f"## {rel_path}\n```python\n{content}\n```"
            for rel_path, content in project_files.items()
        )
        editable = ", ".join(self._editable_files())
        prompt = (
            f"## Hypothesis to Implement\n"
            f"**{hypothesis.description}**\n"
            f"Rationale: {hypothesis.rationale}\n"
            f"Category: {hypothesis.category}\n"
            f"Complexity: {hypothesis.complexity}\n\n"
            f"## Project\n{self.config.project_spec.description}\n\n"
            f"## Editable Files\n{editable}\n\n"
            f"{file_blocks}\n\n"
            f"Implement this hypothesis by modifying only the configured editable files. "
            f"Return COMPLETE modified file content for every file you change.\n\n"
            f"Rules:\n"
            f"- Only modify files listed under Editable Files\n"
            f"- Keep changes minimal and focused on the hypothesis\n"
            f"- Ensure the code is syntactically valid Python\n"
            f"- Preserve all existing functionality unless explicitly changing it\n\n"
            f"Respond with JSON: "
            f"{{\"files\": [{{\"path\": \"relative/file.py\", \"content\": \"...full file content...\"}}], "
            f"\"changes_summary\": \"brief description\"}}"
        )

        messages = self._build_messages(prompt)

        try:
            result = self._call_llm_json(messages, temperature=0.3, max_tokens=8192)
            modified_files = {}
            if "files" in result:
                for file_info in result.get("files", []):
                    rel_path = file_info.get("path")
                    content = file_info.get("content")
                    if rel_path and content:
                        modified_files[rel_path] = content
            elif result.get("modified_code"):
                modified_files[self.config.project_spec.primary_file] = result["modified_code"]
            changes_summary = result.get("changes_summary", hypothesis.description)

            if not modified_files:
                raise ValueError("LLM returned no modified files")

            self._log(f"Code changes: {changes_summary}")
            return modified_files

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
        original_files = self._read_editable_files()

        try:
            # Implement the hypothesis (skip for baseline run)
            if experiment_id > 0:
                modified_files = self.implement_hypothesis(hypothesis, original_files)
                self._write_project_files(modified_files)
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
                    ["git", "diff", "HEAD~1", "HEAD", "--", *self._editable_files()],
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
                command=self.config.project_spec.resolved_run_command(),
                metric_name=self.config.project_spec.metric_name,
                metric_direction=self.config.project_spec.metric_direction,
                package_paths=self.config.project_spec.resolved_package_paths(),
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
                    metric_name=run_result.metric_name,
                    metric_value=run_result.metric_value,
                    metric_direction=run_result.metric_direction,
                    extra_metrics=run_result.extra_metrics,
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
                self._write_project_files(original_files)
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
        """Revert the last experiment commit."""
        try:
            self._git_revert()
            self._log("Reverted last experiment")
        except Exception as e:
            self._log(f"Error reverting: {e}")

    def run(self, hypothesis: Hypothesis, experiment_id: int, **kwargs) -> ExperimentResult:
        """Main entry point."""
        return self.run_experiment(hypothesis, experiment_id)
