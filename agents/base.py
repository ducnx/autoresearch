"""
Base agent class for the multi-agent autoresearch framework.

Provides:
  - LLM interface via litellm (supports OpenAI, Anthropic, Gemini, Ollama)
  - Structured output parsing (JSON mode)
  - Conversation history management
  - Workspace access (read/write shared state)
  - Retry logic with exponential backoff
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import litellm

from core.config import Config, LLMConfig
from core.workspace import Workspace


# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class BaseAgent:
    """
    Base class for all research agents.

    Each agent has:
      - A name and role description
      - A system prompt loaded from .agents/<name>.md
      - Access to the shared workspace
      - An LLM interface for reasoning
    """

    name: str = "base"
    role: str = "Base research agent"

    def __init__(self, config: Config, workspace: Workspace):
        self.config = config
        self.workspace = workspace
        self.llm_config = self._get_llm_config()
        self.system_prompt = self._load_system_prompt()
        self.conversation_history: list[dict] = []

    def _get_llm_config(self) -> LLMConfig:
        """Get the LLM config for this agent from the global config."""
        return getattr(self.config.llm, self.name, LLMConfig.cloud())

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the prompts directory."""
        prompt_path = self.config.project_root / ".agents" / f"{self.name}.md"
        if prompt_path.exists():
            return prompt_path.read_text()
        return f"You are the {self.role}."

    def _log(self, message: str, data: Optional[dict] = None):
        """Log agent activity to workspace."""
        self.workspace.log_agent_activity(self.name, message, data)
        print(f"  [{self.name}] {message}")

    def _call_llm(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Call the LLM with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            json_mode: If True, request JSON output format
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            max_retries: Number of retries on failure

        Returns:
            The LLM's response text
        """
        last_error = None
        configs = [self.llm_config]
        fallback = self.config.fallback_llm(self.llm_config)
        if fallback and fallback.model != self.llm_config.model:
            configs.append(fallback)

        for cfg_index, llm_cfg in enumerate(configs):
            if cfg_index > 0:
                self._log(f"Switching to fallback LLM: {llm_cfg.model}")

            kwargs = {
                "model": llm_cfg.model,
                "messages": messages,
                "temperature": temperature or llm_cfg.temperature,
                "max_tokens": max_tokens or llm_cfg.max_tokens,
                "timeout": llm_cfg.timeout,
            }

            if llm_cfg.api_base:
                kwargs["api_base"] = llm_cfg.api_base

            if llm_cfg.api_key:
                kwargs["api_key"] = llm_cfg.api_key

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            for attempt in range(max_retries):
                try:
                    response = litellm.completion(**kwargs)
                    if cfg_index > 0:
                        self.llm_config = llm_cfg
                    return response.choices[0].message.content
                except Exception as e:
                    last_error = e
                    self._log(f"LLM call failed (attempt {attempt + 1}/{max_retries}) on {llm_cfg.model}: {e}")
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after trying configured and fallback models: {last_error}"
        )

    def _call_llm_json(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Call LLM and parse JSON response."""
        response = self._call_llm(
            messages, json_mode=True, temperature=temperature, max_tokens=max_tokens
        )

        # Try to extract JSON from the response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON within the response (e.g., wrapped in markdown code blocks)
            import re
            json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Last resort: try to find any JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from LLM response: {response[:500]}")

    def _build_messages(
        self,
        user_message: str,
        context: Optional[str] = None,
        include_history: bool = False,
    ) -> list[dict]:
        """Build message list with system prompt and optional context."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if include_history:
            messages.extend(self.conversation_history)

        if context:
            user_message = f"{context}\n\n---\n\n{user_message}"

        messages.append({"role": "user", "content": user_message})
        return messages

    def _add_to_history(self, role: str, content: str):
        """Add a message to conversation history (keep last 20 exchanges)."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep history manageable
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-40:]

    def get_context(self) -> str:
        """
        Build context string from workspace state.
        Subclasses can override to add agent-specific context.
        """
        state = self.workspace.get_state()
        results_summary = self.workspace.get_results_summary()
        return f"## Current State\n{json.dumps(state, indent=2)}\n\n## {results_summary}"

    def run(self, *args, **kwargs) -> Any:
        """
        Main entry point for the agent. Subclasses must implement this.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")
