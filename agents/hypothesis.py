"""
Hypothesis Agent — generates experiment ideas.

Generates ranked lists of hypotheses for the Research Director
to choose from. Uses the experiment history to avoid repeating
failures and build on successes.
"""

import json
import uuid
from typing import Optional

from agents.base import BaseAgent
from core.workspace import Hypothesis


class HypothesisAgent(BaseAgent):
    name = "hypothesis"
    role = "Hypothesis Agent — generates creative experiment ideas"

    def generate(self, research_brief: str, train_code: str, num_ideas: int = 4) -> list[Hypothesis]:
        """
        Generate experiment hypotheses based on the research brief and current code.

        Args:
            research_brief: Research brief from the Director
            train_code: Current contents of train.py
            num_ideas: Number of hypotheses to generate

        Returns:
            List of Hypothesis objects
        """
        self._log("Generating hypotheses...")

        context = self.get_context()

        prompt = (
            f"{research_brief}\n\n"
            f"## Current train.py\n```python\n{train_code}\n```\n\n"
            f"Generate {num_ideas} experiment hypotheses ranked by expected value. "
            f"Each hypothesis should describe a specific, implementable change to train.py.\n\n"
            f"Remember:\n"
            f"- Check experiment history to avoid repeating failed ideas\n"
            f"- Balance between safe incremental changes and bold explorations\n"
            f"- Consider VRAM constraints\n"
            f"- Simpler is better when expected impact is similar\n\n"
            f"Respond with JSON containing a 'hypotheses' array."
        )

        messages = self._build_messages(prompt, context=context)

        try:
            result = self._call_llm_json(messages, temperature=0.8, max_tokens=4096)
            raw_hypotheses = result.get("hypotheses", [])

            hypotheses = []
            for i, raw in enumerate(raw_hypotheses):
                hyp = Hypothesis(
                    id=raw.get("id", f"hyp_{uuid.uuid4().hex[:6]}"),
                    description=raw.get("description", "Unknown"),
                    predicted_impact=raw.get("predicted_impact", "medium"),
                    complexity=raw.get("complexity", "moderate"),
                    risk=raw.get("risk", "medium"),
                    category=raw.get("category", "hyperparameter"),
                    rationale=raw.get("rationale", "No rationale provided"),
                    code_changes=raw.get("code_changes"),
                    source="hypothesis_agent",
                )
                hypotheses.append(hyp)

            self._log(f"Generated {len(hypotheses)} hypotheses", {
                "ids": [h.id for h in hypotheses],
                "categories": [h.category for h in hypotheses],
            })

            # Store in workspace
            self.workspace.add_hypotheses(hypotheses)

            return hypotheses

        except Exception as e:
            self._log(f"Error generating hypotheses: {e}")
            # Fallback: return a simple default hypothesis
            default = Hypothesis(
                id=f"hyp_fallback_{uuid.uuid4().hex[:4]}",
                description="Increase model depth by 1 layer",
                predicted_impact="low",
                complexity="simple",
                risk="low",
                category="hyperparameter",
                rationale="Fallback hypothesis due to generation failure",
                source="hypothesis_agent",
            )
            self.workspace.add_hypotheses([default])
            return [default]

    def run(self, research_brief: str, train_code: str, **kwargs) -> list[Hypothesis]:
        """Main entry point."""
        return self.generate(
            research_brief=research_brief,
            train_code=train_code,
            num_ideas=kwargs.get("num_ideas", 4),
        )
