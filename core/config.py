"""
Configuration for the multi-agent autoresearch framework.

Supports hybrid LLM strategy:
  - Cloud API (via litellm) for tasks requiring internet access
  - Local Ollama for cost-sensitive tasks

GPU support:
  - Auto-detects NVIDIA GPU availability
  - Supports --dry-run mode for testing without GPU
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _detect_gpu() -> bool:
    """Check if an NVIDIA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Fallback: check nvidia-smi
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_ollama() -> bool:
    """Check if Ollama is available locally."""
    try:
        subprocess.run(
            ["ollama", "list"], capture_output=True, check=True, timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    @classmethod
    def cloud(cls, model: str = "gemini/gemini-2.5-flash", **kwargs) -> "LLMConfig":
        """Create a cloud LLM config (for internet-access tasks)."""
        return cls(model=model, **kwargs)

    @classmethod
    def local(cls, model: str = "ollama/qwen3:8b", **kwargs) -> "LLMConfig":
        """Create a local Ollama config (for cost-sensitive tasks)."""
        return cls(
            model=model,
            api_base="http://localhost:11434",
            temperature=kwargs.pop("temperature", 0.7),
            max_tokens=kwargs.pop("max_tokens", 4096),
            timeout=kwargs.pop("timeout", 300),
            **kwargs,
        )


@dataclass
class AgentLLMConfig:
    """Per-agent LLM configuration following the hybrid strategy."""
    director: LLMConfig = field(default_factory=lambda: LLMConfig.cloud())
    hypothesis: LLMConfig = field(default_factory=lambda: LLMConfig.local())
    literature: LLMConfig = field(default_factory=lambda: LLMConfig.cloud())
    experiment: LLMConfig = field(default_factory=lambda: LLMConfig.local())
    analysis: LLMConfig = field(default_factory=lambda: LLMConfig.local())
    report: LLMConfig = field(default_factory=lambda: LLMConfig.local())


@dataclass
class Config:
    """Global configuration for the autoresearch framework."""

    # --- Paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    project: str = "default"
    project_dir: Path = field(default=None)
    workspace_dir: Path = field(default=None)
    train_script: str = "train.py"
    prepare_script: str = "prepare.py"

    # --- LLM ---
    llm: AgentLLMConfig = field(default_factory=AgentLLMConfig)

    # --- Experiment ---
    experiment_timeout: int = 600  # 10 minutes max per experiment
    time_budget: int = 300  # 5 minutes training time budget
    max_retries: int = 2  # retries for crashed experiments

    # --- Runtime ---
    dry_run: bool = False  # simulate experiments without GPU
    has_gpu: bool = field(default_factory=_detect_gpu)
    has_ollama: bool = field(default_factory=_detect_ollama)
    run_tag: str = "default"
    max_experiments: Optional[int] = None  # None = unlimited
    report_interval: int = 5  # generate report every N experiments

    # --- Dashboard ---
    dashboard_port: int = 8501

    def __post_init__(self):
        if self.project_dir is None:
            self.project_dir = self.project_root / "projects" / self.project
            
        if self.workspace_dir is None:
            self.workspace_dir = self.project_dir / "workspace"

        # If no GPU and not explicitly set to dry_run, auto-enable dry_run
        if not self.has_gpu and not self.dry_run:
            print("[config] No NVIDIA GPU detected — enabling dry-run mode")
            self.dry_run = True

        # If no Ollama, fall back all local models to cloud
        if not self.has_ollama:
            print("[config] Ollama not detected — falling back local agents to cloud API")
            cloud = LLMConfig.cloud()
            self.llm.hypothesis = cloud
            self.llm.experiment = cloud
            self.llm.analysis = cloud
            self.llm.report = cloud

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Create config from environment variables with overrides."""
        config = cls(**overrides)

        # Override cloud model from env
        cloud_model = os.environ.get("AUTORESEARCH_CLOUD_MODEL")
        if cloud_model:
            for name in ["director", "literature"]:
                getattr(config.llm, name).model = cloud_model

        # Override local model from env
        local_model = os.environ.get("AUTORESEARCH_LOCAL_MODEL")
        if local_model and config.has_ollama:
            for name in ["hypothesis", "experiment", "analysis", "report"]:
                getattr(config.llm, name).model = f"ollama/{local_model}"

        # API key from env
        api_key = os.environ.get("AUTORESEARCH_API_KEY")
        if api_key:
            for name in ["director", "literature"]:
                getattr(config.llm, name).api_key = api_key

        return config
