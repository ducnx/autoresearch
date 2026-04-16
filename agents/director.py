"""
Research Director Agent — the orchestrator.

Analogous to PaperOrchestra's Outline Agent.
Manages overall research strategy, selects hypotheses,
and makes keep/discard decisions.
"""

import json
from typing import Optional

from agents.base import BaseAgent
from core.workspace import Hypothesis, ExperimentResult


class DirectorAgent(BaseAgent):
    name = "director"
    role = "Research Director — orchestrates the multi-agent research loop"

    def select_hypothesis(
        self,
        hypotheses: list[Hypothesis],
        literature: Optional[list[dict]] = None,
    ) -> Optional[Hypothesis]:
        """
        Select the most promising hypothesis from candidates.

        Args:
            hypotheses: List of hypothesis candidates
            literature: Optional literature findings to inform selection

        Returns:
            Selected hypothesis, or None if no good candidates
        """
        if not hypotheses:
            self._log("No hypotheses to select from")
            return None

        context = self.get_context()

        # Build hypothesis descriptions
        hyp_text = "\n".join([
            f"- **{h.id}** [{h.predicted_impact} impact, {h.complexity} complexity, {h.risk} risk]: "
            f"{h.description}\n  Rationale: {h.rationale}\n  Category: {h.category}"
            for h in hypotheses
        ])

        # Add literature context if available
        lit_text = ""
        if literature:
            lit_text = "\n\n## Relevant Literature Findings\n" + "\n".join([
                f"- {entry.get('title', 'Unknown')}: {entry.get('technique', 'N/A')} "
                f"(relevance: {entry.get('relevance_score', 0):.2f})"
                for entry in literature[:5]
            ])

        prompt = (
            f"## Hypothesis Candidates\n{hyp_text}"
            f"{lit_text}\n\n"
            f"Select the best hypothesis to test next. Consider the experiment history, "
            f"current best result, and balance between exploration and exploitation.\n\n"
            f"Respond with JSON containing: selected_hypothesis_id, reasoning, strategy_note"
        )

        messages = self._build_messages(prompt, context=context)

        try:
            result = self._call_llm_json(messages, temperature=0.3)
            selected_id = result.get("selected_hypothesis_id")
            reasoning = result.get("reasoning", "No reasoning provided")
            strategy = result.get("strategy_note", "")

            self._log(f"Selected hypothesis: {selected_id}", {
                "reasoning": reasoning,
                "strategy": strategy,
            })

            # Find and return the selected hypothesis
            for h in hypotheses:
                if h.id == selected_id:
                    return h

            # If ID didn't match, return the first one
            self._log(f"Warning: Selected ID '{selected_id}' not found, using first hypothesis")
            return hypotheses[0]

        except Exception as e:
            self._log(f"Error selecting hypothesis: {e}")
            # Fallback: return the first hypothesis
            return hypotheses[0] if hypotheses else None

    def decide(self, result: ExperimentResult, hypothesis: Hypothesis) -> str:
        """
        Decide whether to keep or discard an experiment result.

        Args:
            result: The experiment result to evaluate
            hypothesis: The hypothesis that was tested

        Returns:
            "keep" or "discard"
        """
        if result.status == "crash":
            return "crash"

        context = self.get_context()
        state = self.workspace.get_state()

        current_best = state.get("best_bpb")
        baseline = state.get("baseline_bpb")

        prompt = (
            f"## Latest Experiment Result\n"
            f"- Hypothesis: {hypothesis.description}\n"
            f"- val_bpb: {result.val_bpb:.6f}\n"
            f"- peak_vram_mb: {result.peak_vram_mb:.1f}\n"
            f"- Current best val_bpb: {current_best}\n"
            f"- Baseline val_bpb: {baseline}\n\n"
            f"Should we KEEP this result (advance the branch) or DISCARD it (revert)?\n"
            f"Consider: improvement magnitude, complexity cost, VRAM impact.\n\n"
            f"Respond with JSON containing: decision (keep/discard), reasoning, next_direction"
        )

        messages = self._build_messages(prompt, context=context)

        try:
            response = self._call_llm_json(messages, temperature=0.2)
            decision = response.get("decision", "discard").lower()
            reasoning = response.get("reasoning", "No reasoning provided")

            self._log(f"Decision: {decision} for experiment {result.experiment_id}", {
                "val_bpb": result.val_bpb,
                "reasoning": reasoning,
                "next_direction": response.get("next_direction", ""),
            })

            # Sanity check: if val_bpb improved, lean toward keeping
            if current_best is not None and result.val_bpb < current_best:
                if decision == "discard":
                    self._log("Override: keeping despite LLM saying discard (val_bpb improved)")
                    decision = "keep"

            return decision if decision in ("keep", "discard") else "discard"

        except Exception as e:
            self._log(f"Error in decision making: {e}")
            # Fallback: keep if improved, discard otherwise
            if current_best is not None and result.val_bpb < current_best:
                return "keep"
            return "discard"

    def get_research_brief(self) -> str:
        """
        Generate a research brief for the Hypothesis and Literature agents.

        Returns:
            String summarizing current state and what to focus on
        """
        state = self.workspace.get_state()
        results = self.workspace.get_results()

        brief_parts = [
            f"## Research Brief",
            f"- Experiments completed: {state['experiment_count']}",
            f"- Baseline val_bpb: {state.get('baseline_bpb', 'not yet established')}",
            f"- Best val_bpb: {state.get('best_bpb', 'not yet established')}",
        ]

        if results:
            recent = results[-5:]
            brief_parts.append("\n## Recent Experiments")
            for r in recent:
                brief_parts.append(
                    f"- [{r.status}] val_bpb={r.val_bpb:.6f}: {r.description}"
                )

            # Identify what worked and what didn't
            kept = [r for r in results if r.status == "keep"]
            discarded = [r for r in results if r.status == "discard"]
            crashed = [r for r in results if r.status == "crash"]

            if kept:
                brief_parts.append(f"\n## What Worked ({len(kept)} experiments)")
                for r in kept[-3:]:
                    brief_parts.append(f"- {r.description}")
            if discarded:
                brief_parts.append(f"\n## What Didn't Work ({len(discarded)} experiments)")
                for r in discarded[-3:]:
                    brief_parts.append(f"- {r.description}")
            if crashed:
                brief_parts.append(f"\n## Crashes ({len(crashed)} experiments)")
                for r in crashed[-3:]:
                    brief_parts.append(f"- {r.description}: {r.error_message}")

        return "\n".join(brief_parts)

    def run(self, phase: str = "brief", **kwargs):
        """
        Run the director in the specified phase.

        Phases:
            - "brief": Generate research brief
            - "select": Select a hypothesis
            - "decide": Make keep/discard decision
        """
        if phase == "brief":
            return self.get_research_brief()
        elif phase == "select":
            return self.select_hypothesis(
                kwargs.get("hypotheses", []),
                kwargs.get("literature"),
            )
        elif phase == "decide":
            return self.decide(
                kwargs["result"],
                kwargs["hypothesis"],
            )
        else:
            raise ValueError(f"Unknown director phase: {phase}")
