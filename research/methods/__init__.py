"""Reference metadata for planned research-method integrations."""

from research.methods.hcapo import summarize_hindsight
from research.methods.klong import build_subtrajectories
from research.methods.memex import summarize_memory_usage
from research.methods.mira import score_milestone_frontier
from research.methods.oversight import summarize_oversight
from research.methods.salt import compute_step_advantages
from research.methods.registry import METHOD_REGISTRY

__all__ = [
    "METHOD_REGISTRY",
    "summarize_hindsight",
    "build_subtrajectories",
    "summarize_memory_usage",
    "score_milestone_frontier",
    "summarize_oversight",
    "compute_step_advantages",
]
