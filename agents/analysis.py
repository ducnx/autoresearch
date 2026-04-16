"""
Analysis Agent — interprets experiment results.

Analogous to PaperOrchestra's Content Refinement Agent.
Analyzes results, identifies patterns, and provides
insights to guide future experiments.
"""

import json
from typing import Optional

from agents.base import BaseAgent
from core.workspace import ExperimentResult


class AnalysisAgent(BaseAgent):
    name = "analysis"
    role = "Analysis Agent — interprets results and identifies patterns"

    def analyze(self, result: ExperimentResult) -> dict:
        """
        Analyze the latest experiment result in context of all prior results.

        Args:
            result: The latest experiment result

        Returns:
            Analysis dictionary with assessment, patterns, and suggestions
        """
        self._log(f"Analyzing experiment {result.experiment_id}...")

        context = self.get_context()
        state = self.workspace.get_state()
        all_results = self.workspace.get_results()

        # Build detailed prompt
        baseline_bpb = state.get("baseline_bpb", "unknown")
        best_bpb = state.get("best_bpb", "unknown")

        results_detail = "\n".join([
            f"- Exp {r.experiment_id} [{r.status}]: val_bpb={r.val_bpb:.6f}, "
            f"VRAM={r.peak_vram_mb:.0f}MB, {r.description}"
            for r in all_results[-10:]  # Last 10 experiments
        ])

        prompt = (
            f"## Latest Result\n"
            f"- Experiment {result.experiment_id}: {result.description}\n"
            f"- val_bpb: {result.val_bpb:.6f}\n"
            f"- Status: {result.status}\n"
            f"- peak_vram_mb: {result.peak_vram_mb:.1f}\n"
            f"- training_seconds: {result.training_seconds:.1f}\n"
            f"- mfu_percent: {result.mfu_percent:.2f}\n"
            f"- num_params_M: {result.num_params_m:.1f}\n"
            f"- depth: {result.depth}\n\n"
            f"## Baseline val_bpb: {baseline_bpb}\n"
            f"## Current best val_bpb: {best_bpb}\n\n"
            f"## Recent Experiment History\n{results_detail}\n\n"
            f"Analyze this result. Consider:\n"
            f"1. Is the improvement (if any) meaningful or noise?\n"
            f"2. What patterns do you see across experiments?\n"
            f"3. Are we hitting diminishing returns?\n"
            f"4. What should we try next?\n\n"
            f"Respond with JSON containing: analysis, patterns, suggestions, meta"
        )

        messages = self._build_messages(prompt, context=context)

        try:
            analysis = self._call_llm_json(messages, temperature=0.3, max_tokens=2048)

            self._log(f"Analysis complete", {
                "assessment": analysis.get("analysis", {}).get("result_assessment", "unknown"),
                "keep_recommendation": analysis.get("analysis", {}).get("keep_recommendation"),
                "num_suggestions": len(analysis.get("suggestions", [])),
            })

            return analysis

        except Exception as e:
            self._log(f"Error analyzing results: {e}")
            # Return a basic analysis
            improved = (
                best_bpb is not None
                and isinstance(best_bpb, (int, float))
                and result.val_bpb < best_bpb
            )
            return {
                "analysis": {
                    "result_assessment": "improved" if improved else "no improvement",
                    "improvement_magnitude": "unknown",
                    "keep_recommendation": improved,
                    "reasoning": f"Automated fallback: val_bpb {'improved' if improved else 'did not improve'}",
                },
                "patterns": [],
                "suggestions": [],
                "meta": {"error": str(e)},
            }

    def run(self, result: ExperimentResult, **kwargs) -> dict:
        """Main entry point."""
        return self.analyze(result)
