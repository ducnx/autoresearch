"""
Writer Agent — drafts and refines academic paper sections in LyX format.

Analogous to PaperOrchestra's Paper Writing Agent.
Writes academic text, structures paragraphs, and formats for LyX.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
from agents.base import BaseAgent
from core.workspace import ExperimentResult, Hypothesis


class WriterAgent(BaseAgent):
    name = "writer"
    role = "Writer Agent — drafts and refines academic paper sections in LyX format"

    def _target_path(self, target_file: str) -> Optional[Path]:
        writing_dir = self.config.project_spec.writing_dir
        if not writing_dir:
            return None
        return self.config.project_dir / writing_dir / target_file

    @staticmethod
    def _lyx_layout(text: str) -> str:
        return f"\\begin_layout Standard\n{text}\n\\end_layout"

    def update_results_log(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis,
        analysis: Optional[dict] = None,
    ) -> Optional[Path]:
        """Append a factual experiment record to the configured LyX results file."""
        target = self._target_path(self.config.project_spec.result_section)
        if target is None or not target.exists():
            return None

        metric_value = result.metric_value if result.metric_value is not None else result.val_bpb
        extra = result.extra_metrics or {}
        extra_bits = []
        for key in sorted(extra):
            if key in {"val_bpb", result.metric_name}:
                continue
            extra_bits.append(f"{key}={extra[key]:.6g}")
        extra_text = "; ".join(extra_bits[:8]) if extra_bits else "no auxiliary metrics recorded"
        assessment = ""
        if analysis:
            analysis_block = analysis.get("analysis", analysis)
            if isinstance(analysis_block, dict):
                assessment = analysis_block.get("result_assessment") or analysis_block.get("reasoning") or ""
            elif isinstance(analysis_block, str):
                assessment = analysis_block

        text = target.read_text()
        entry_lines = []
        if "## Autoresearch Results Log" not in text:
            entry_lines.append(self._lyx_layout("## Autoresearch Results Log"))
        entry_lines.append(
            self._lyx_layout(
                f"- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"Experiment {result.experiment_id} [{result.status}] tested "
                f"{hypothesis.description}. Primary metric "
                f"{result.metric_name}={metric_value:.6f} "
                f"({result.metric_direction}); normalized objective={result.val_bpb:.6f}. "
                f"Auxiliary metrics: {extra_text}."
            )
        )
        if assessment:
            entry_lines.append(self._lyx_layout(f"  Analysis note: {assessment}"))

        block = "\n\n".join(entry_lines) + "\n\n"
        marker = "\\end_body"
        if marker not in text:
            self._log(f"Cannot update {target}: missing LyX end_body marker")
            return None
        target.write_text(text.replace(marker, block + marker, 1))
        self._log(f"Updated results writing: {target}")
        return target

    def write_section(self, section_name: str, additional_instructions: Optional[str] = None) -> dict:
        """
        Draft or refine a specific academic paper section in LyX format.

        Args:
            section_name: Name of the section (e.g., intro, related, method, result, conclusion)
            additional_instructions: Optional specific focus or constraints for writing

        Returns:
            Dictionary containing the written LyX text and metadata
        """
        self._log(f"Drafting paper section '{section_name}'...")

        # Map section name to target file in the configured writing directory.
        file_mapping = {
            "intro": "intro.lyx",
            "related": "related.lyx",
            "literature": "related.lyx",
            "method": "method.lyx",
            "methodology": "method.lyx",
            "result": "result.lyx",
            "results": "result.lyx",
            "conclusion": "conclusion.lyx",
        }
        target_file = file_mapping.get(section_name.lower(), f"{section_name}.lyx")

        context = self.get_context()
        state = self.workspace.get_state()
        results = self.workspace.get_results()

        # Build detailed results text to feed to LLM
        results_text = "\n".join([
            f"- Exp {r.experiment_id} [{r.status}]: {r.metric_name}="
            f"{(r.metric_value if r.metric_value is not None else r.val_bpb):.6f}, "
            f"VRAM={r.peak_vram_mb:.0f}MB, Params={r.num_params_m:.1f}M — {r.description}"
            for r in results
        ])

        prompt = (
            f"## Target Section\n"
            f"- Section to write: {section_name}\n"
            f"- Target child file: {target_file}\n"
            f"- Format: LyX document serialized block layout (using \\begin_layout Standard and \\end_layout)\n\n"
            f"## All Experiment Results\n{results_text}\n\n"
            f"## Research State\n"
            f"- Baseline normalized objective: {state.get('baseline_bpb')}\n"
            f"- Best normalized objective: {state.get('best_bpb')}\n"
            f"- Total experiments run: {state.get('experiment_count')}\n\n"
        )

        if additional_instructions:
            prompt += f"## Additional Instructions\n{additional_instructions}\n\n"

        prompt += (
            f"Draft a high-quality academic section for '{section_name}' targeted for '{target_file}'. "
            f"Ensure all text is formatted with proper LyX paragraph layouts. "
            f"Describe methodology, experimental results, and key improvements factually "
            f"and professionally.\n\n"
            f"Respond with JSON containing: section, title, target_file, summary, lyx_content, key_results_used, recommendations_for_user"
        )

        messages = self._build_messages(prompt, context=context)

        try:
            writer_data = self._call_llm_json(messages, temperature=0.3, max_tokens=4096)
            self._log(f"Section '{section_name}' drafted successfully for target file '{target_file}'")
            return writer_data
        except Exception as e:
            self._log(f"Error drafting section '{section_name}': {e}")
            return {
                "section": section_name,
                "title": section_name.capitalize(),
                "target_file": target_file,
                "summary": f"Fallback draft for {section_name}",
                "lyx_content": f"\\begin_layout Standard\n[Failed to generate content: {e}]\n\\end_layout",
                "key_results_used": [],
                "recommendations_for_user": ["Try re-generating or running with a different model."]
            }

    def run(self, section: str = "intro", additional_instructions: Optional[str] = None, **kwargs) -> dict:
        """Main entry point."""
        return self.write_section(section, additional_instructions)
