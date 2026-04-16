"""Multi-agent framework for autonomous research."""

from agents.base import BaseAgent
from agents.director import DirectorAgent
from agents.hypothesis import HypothesisAgent
from agents.literature import LiteratureAgent
from agents.experiment import ExperimentAgent
from agents.analysis import AnalysisAgent
from agents.report import ReportAgent

__all__ = [
    "BaseAgent",
    "DirectorAgent",
    "HypothesisAgent",
    "LiteratureAgent",
    "ExperimentAgent",
    "AnalysisAgent",
    "ReportAgent",
]
