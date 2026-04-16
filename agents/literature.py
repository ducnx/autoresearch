"""
Literature Agent — searches for relevant techniques.

Analogous to PaperOrchestra's Literature Review Agent.
Uses cloud LLM (with internet access) to find applicable techniques
from recent research.
"""

import json
from typing import Optional

from agents.base import BaseAgent
from core.workspace import LiteratureEntry


class LiteratureAgent(BaseAgent):
    name = "literature"
    role = "Literature Agent — discovers relevant research techniques"

    def search(self, research_brief: str, focus_areas: Optional[list[str]] = None) -> list[LiteratureEntry]:
        """
        Search for relevant techniques based on the research brief.

        Uses the cloud LLM's knowledge to identify applicable techniques
        from recent ML research literature.

        Args:
            research_brief: Research brief from the Director
            focus_areas: Optional list of specific areas to focus on

        Returns:
            List of LiteratureEntry objects
        """
        self._log("Searching literature...")

        context = self.get_context()

        focus_text = ""
        if focus_areas:
            focus_text = f"\n\n## Focus Areas\n" + "\n".join(f"- {area}" for area in focus_areas)

        prompt = (
            f"{research_brief}\n"
            f"{focus_text}\n\n"
            f"Search your knowledge of ML research for techniques that could improve "
            f"this language model training setup. Focus on:\n"
            f"1. Techniques applicable to small/medium GPT models\n"
            f"2. Methods that work within a fixed compute budget\n"
            f"3. Recent advances in efficient training (2024-2026)\n"
            f"4. Optimizer improvements, architecture tweaks, training recipes\n\n"
            f"For each technique, assess how directly it can be applied to the "
            f"current single-file `train.py` setup.\n\n"
            f"Respond with JSON containing a 'findings' array."
        )

        messages = self._build_messages(prompt, context=context)

        try:
            result = self._call_llm_json(messages, temperature=0.5, max_tokens=4096)
            raw_findings = result.get("findings", [])

            entries = []
            for raw in raw_findings:
                entry = LiteratureEntry(
                    title=raw.get("title", "Unknown"),
                    source=raw.get("source", "LLM knowledge"),
                    summary=raw.get("summary", ""),
                    technique=raw.get("technique", ""),
                    applicability=raw.get("applicability", "needs_adaptation"),
                    relevance_score=float(raw.get("relevance_score", 0.5)),
                )
                entries.append(entry)

            # Sort by relevance
            entries.sort(key=lambda e: e.relevance_score, reverse=True)

            self._log(f"Found {len(entries)} relevant techniques", {
                "titles": [e.title for e in entries[:5]],
            })

            # Store in workspace
            self.workspace.add_literature(entries)

            return entries

        except Exception as e:
            self._log(f"Error searching literature: {e}")
            return []

    def get_literature_for_hypothesis(self, hypothesis_description: str) -> list[LiteratureEntry]:
        """
        Find literature relevant to a specific hypothesis.

        Args:
            hypothesis_description: Description of the hypothesis to research

        Returns:
            Filtered list of relevant literature entries
        """
        # Check existing literature first
        existing = self.workspace.get_literature()
        if existing:
            # Use LLM to score relevance to this specific hypothesis
            prompt = (
                f"## Hypothesis\n{hypothesis_description}\n\n"
                f"## Available Literature\n"
                + "\n".join([
                    f"- [{i}] {e.title}: {e.technique}"
                    for i, e in enumerate(existing)
                ])
                + "\n\nWhich entries are most relevant to this hypothesis? "
                f"Respond with JSON: {{\"relevant_indices\": [0, 2, 5]}}"
            )

            try:
                messages = self._build_messages(prompt)
                result = self._call_llm_json(messages, temperature=0.2)
                indices = result.get("relevant_indices", [])
                return [existing[i] for i in indices if i < len(existing)]
            except Exception:
                pass

        return existing[:3] if existing else []

    def run(self, research_brief: str, **kwargs) -> list[LiteratureEntry]:
        """Main entry point."""
        return self.search(
            research_brief=research_brief,
            focus_areas=kwargs.get("focus_areas"),
        )
