"""Typed Pydantic models for Adaptive Clinical Trial Recruitment environment."""

from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional


class Observation(BaseModel):
    """What the agent observes each step."""

    timestamp: int = Field(description="Simulated day since trial start (0-180)")
    budget_remaining: float = Field(description="Budget left in dollars")
    time_to_deadline_days: int = Field(
        description="Days remaining before trial deadline"
    )
    enrolled_so_far: int = Field(description="Total patients successfully enrolled")
    target_enrollment: int = Field(description="Target number of patients to enroll")
    current_funnel: Dict[str, int] = Field(
        default_factory=dict,
        description="Funnel counts: contacted, screened, eligible, consented, enrolled, dropped",
    )
    available_patients: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 5 candidate patients with id, age, eligibility_score, dropout_risk",
    )
    site_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-site metrics: conversion_rate, avg_wait_days, capacity_remaining",
    )
    recent_events: List[str] = Field(
        default_factory=list,
        description="Recent event strings (e.g. 'dropout', 'site_delay', 'regulatory_hold')",
    )
    uncertainty_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current uncertainty in patient pool quality (0=certain, 1=very uncertain)",
    )
    difficulty: int = Field(default=1, description="1=easy, 2=medium, 3=hard")
    dropout_rate_7d: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Rolling 7-day dropout rate"
    )
    screening_backlog: int = Field(
        default=0, ge=0, description="Number of patients awaiting screening results"
    )
    causal_insight: str = Field(
        default="",
        description="Environment-generated causal feedback explaining dominant dynamics",
    )
    hypothesis_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How close the agent's last hypothesis was to ground truth (0=wrong, 1=exact)",
    )
    world_type: str = Field(
        default="unknown",
        description="Ground truth dominant dynamic for the current task (noise, site_bias, dropout)",
    )


class Action(BaseModel):
    """Action the agent takes each step."""

    action_type: Literal[
        "screen_patient",
        "recontact",
        "allocate_to_site",
        "adjust_strategy",
        "stop_recruitment",
    ] = Field(description="Type of recruitment action")
    patient_id: Optional[str] = Field(
        default=None, description="Target patient ID (for screen/recontact/allocate)"
    )
    site_id: Optional[str] = Field(
        default=None, description="Target site ID (for allocate)"
    )
    strategy_change: Optional[str] = Field(
        default=None,
        description="Strategy adjustment: increase_outreach, relax_criteria, focus_site_A, focus_site_B, focus_site_C, tighten_criteria",
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="Agent's hypothesis about dominant trial dynamics: dropout_dominant, noise_dominant, site_bias, confounding, or unknown",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in its hypothesis (0.0-1.0)",
    )


class Reward(BaseModel):
    """Detailed reward breakdown per step."""

    total: float = Field(description="Total reward this step")
    screening_success: float = Field(
        default=0.0, description="Reward for successful screening"
    )
    enrollment_gain: float = Field(default=0.0, description="Reward for new enrollment")
    dropout_penalty: float = Field(
        default=0.0, description="Penalty for patient dropout"
    )
    budget_efficiency: float = Field(
        default=0.0, description="Reward for cost-efficient operations"
    )
    timeline_bonus: float = Field(
        default=0.0, description="Bonus for being ahead of schedule"
    )


class State(BaseModel):
    """Full environment state for checkpointing."""

    task: str = Field(default="")
    step: int = Field(default=0)
    max_steps: int = Field(default=180)
    done: bool = Field(default=True)
    enrolled_so_far: int = Field(default=0)
    target_enrollment: int = Field(default=100)
    budget_remaining: float = Field(default=0.0)
    total_reward: float = Field(default=0.0)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class StepResult(BaseModel):
    """Result returned by step() and reset()."""

    observation: Observation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)
