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
    recontact_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 5 screened or eligible patients worth recontacting for consent or enrollment",
    )
    allocation_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 5 consented patients eligible for site allocation and enrollment",
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
    milestones: Dict[str, bool] = Field(
        default_factory=dict,
        description="Enrollment progress milestones reached so far (25%, 50%, 75%, 100%)",
    )
    active_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Currently active operational constraints such as regulatory holds or sponsor pressure",
    )
    delayed_effects_pending: int = Field(
        default=0,
        ge=0,
        description="Number of scheduled delayed consequences that have not triggered yet",
    )
    uncertainty_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed uncertainty across patient pool, site operations, and policy dynamics",
    )
    patient_memory_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary of remembered patient cohorts accumulated across prior actions",
    )
    counterfactual_hint: str = Field(
        default="",
        description="Simple what-if suggestion showing an alternative next move worth testing",
    )
    current_plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current high-level phase plan for explicit Plan-and-Act execution",
    )
    indexed_memory_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary of indexed memory entries available to the agent",
    )
    retrieved_memory_context: str = Field(
        default="",
        description="Most recently retrieved memory context exposed back to the agent",
    )
    milestone_potential: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="MiRA-style milestone potential critic score for the current state",
    )
    active_milestone: str = Field(
        default="",
        description="Currently weakest milestone frontier that should be targeted next",
    )
    hindsight_available: bool = Field(
        default=False,
        description="Whether an end-of-episode hindsight credit summary is available",
    )
    token_budget_remaining: int = Field(
        default=0,
        ge=0,
        description="Remaining inference token budget available to the agent in this episode",
    )
    token_usage_so_far: int = Field(
        default=0,
        ge=0,
        description="Cumulative tokens consumed so far by the agent during the episode",
    )
    token_efficiency_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mercor-style efficiency score rewarding useful progress at lower token cost",
    )
    counterfactual_rollout: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated gain from alternative next moves (allocate vs recontact)",
    )


class Action(BaseModel):
    """Action the agent takes each step."""

    action_type: Literal[
        "screen_patient",
        "recontact",
        "allocate_to_site",
        "adjust_strategy",
        "plan_next_phase",
        "summarize_and_index",
        "retrieve_relevant_history",
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
        description="Strategy adjustment: increase_outreach, relax_criteria, focus_site_A/B/C, tighten_criteria, or negotiate_site_A/B/C",
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
    plan_id: Optional[str] = Field(
        default=None,
        description="Optional high-level plan identifier for explicit Plan-and-Act control",
    )
    plan_summary: Optional[str] = Field(
        default=None,
        description="Natural-language summary for a new or updated high-level plan",
    )
    target_phase: Optional[str] = Field(
        default=None,
        description="High-level phase target such as screening, conversion, allocation, retention, or recovery",
    )
    memory_key: Optional[str] = Field(
        default=None,
        description="Key used when writing or updating an indexed memory entry",
    )
    memory_query: Optional[str] = Field(
        default=None,
        description="Query string used to retrieve relevant indexed memory entries",
    )
    memory_payload: Optional[str] = Field(
        default=None,
        description="Summary text written into indexed memory during summarize_and_index",
    )
    token_cost: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional estimated token cost attached to the action for token-aware accounting",
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
    milestones: Dict[str, bool] = Field(default_factory=dict)
    active_constraints: Dict[str, Any] = Field(default_factory=dict)
    delayed_effects_pending: int = Field(default=0, ge=0)
    uncertainty_components: Dict[str, float] = Field(default_factory=dict)
    current_plan: Dict[str, Any] = Field(default_factory=dict)
    indexed_memory_summary: Dict[str, int] = Field(default_factory=dict)
    retrieved_memory_context: str = Field(default="")
    milestone_potential: float = Field(default=0.0, ge=0.0, le=1.0)
    active_milestone: str = Field(default="")
    hindsight_summary: Dict[str, Any] = Field(default_factory=dict)
    token_budget_remaining: int = Field(default=0, ge=0)
    token_usage_so_far: int = Field(default=0, ge=0)
    token_efficiency_score: float = Field(default=1.0, ge=0.0, le=1.0)


class StepResult(BaseModel):
    """Result returned by step() and reset()."""

    observation: Observation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)
