# Theme #2 Alignment: Long-Horizon Planning & Instruction Following

## Overview

The Clinical Trial Recruitment Environment is designed specifically for **Theme #2: (Super) Long-Horizon Planning & Instruction Following**. This document details how each requirement is addressed.

---

## Core Requirements Alignment

### 1. Deep, Multi-Step Reasoning with Sparse/Delayed Rewards

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Long episodes | 180-step episodes (180 simulated days) | `env.py:_max_steps = 180` |
| Sparse rewards | Enrollment rewards (+0.5) are rare | ~5-10 enrollments per episode |
| Delayed rewards | Milestone bonuses at 25/50/75/100% | `env.py:_check_milestone_bonus()` |
| Delayed consequences | Consent windows expire in 5-14 days | `env.py:_delayed_effects_queue` |

**Delayed Effects Queue**:
```python
# From env.py
self._delayed_effects_queue: List[Tuple[int, str, Dict[str, Any]]] = []
# Effects include:
# - consent_expiry (5-14 days)
# - site_negotiation_result (3-7 days)
# - outreach_wave_response (7-21 days)
# - regulatory_resolution (14-30 days)
```

### 2. Goal Decomposition

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Hierarchical goals | HCAPO agent with subgoal decomposition | `research/methods/hcapo_agent.py` |
| Milestone tracking | 25%, 50%, 75%, 100% enrollment checkpoints | `models.py:milestones` |
| Phase planning | Explicit plan_next_phase action | `models.py:Action.target_phase` |
| Subgoal execution | StrictSubgoalExecutor class | `research/replay.py` |

**Subgoal Structure**:
```python
# From research/replay.py
@dataclass
class Subgoal:
    name: str
    target_metric: str  # e.g., "enrolled_so_far"
    target_value: float  # e.g., 30 (25% of 120)
    deadline_step: Optional[int] = None
    completed: bool = False
```

### 3. State Tracking Over Extended Trajectories

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Patient memory | Persistent patient cohort tracking | `env.py:_patient_memory` |
| Indexed memory | Episodic memory with write/retrieve | `models.py:indexed_memory_summary` |
| History compression | MemexRL attention-based retrieval | `research/methods/memex_agent.py` |
| 37-dim state vector | Rich observation space | `training/neural_policy.py:STATE_DIM=37` |

**Memory System**:
```python
# From env.py
self._patient_memory: Dict[str, Dict[str, Any]] = {}  # Patient-level tracking
self._indexed_memory: Dict[str, Dict[str, Any]] = {}  # Episodic memory store
self._memory_index_counter: int = 0
```

### 4. Recovery from Early Mistakes

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Recovery curriculum | EarlyMistakeRecoveryCurriculum | `training/curriculum.py` |
| Hindsight relabeling | HER in HCAPO agent | `research/methods/hcapo_agent.py` |
| Counterfactual hints | What-if guidance in observation | `models.py:counterfactual_hint` |
| Recontact mechanism | Re-engage dropped patients | `Action.action_type="recontact"` |

**Recovery Scenarios**:
```python
# From training/curriculum.py
RECOVERY_SCENARIOS = [
    RecoveryScenario("budget_overrun", budget_threshold=0.3, recovery_actions=["adjust_strategy"]),
    RecoveryScenario("high_dropout", dropout_threshold=0.2, recovery_actions=["recontact"]),
    RecoveryScenario("site_bottleneck", site_utilization=0.9, recovery_actions=["negotiate_site_terms"]),
    RecoveryScenario("missed_milestone", milestone_delay=14, recovery_actions=["screen_patient"]),
    RecoveryScenario("regulatory_hold", constraint_type="regulatory", recovery_actions=["request_budget_extension"]),
]
```

### 5. Beyond Shallow Next-Token Reasoning

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Multi-scale abstraction | KLong with 1/5/20/60 step windows | `research/methods/klong_agent.py` |
| Potential-based shaping | MiRA learned potential function | `research/methods/mira_agent.py` |
| TD(λ) learning | Eligibility traces for credit assignment | `research/methods/klong_agent.py` |
| Trajectory segmentation | Overlapping context windows | `training/trajectory_splitter.py` |

### 6. Structured Planning and Durable Representations

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Explicit planning | Plan-and-Act separation | `models.py:current_plan` |
| Phase-based execution | Screening→Conversion→Completion | `env.py:_current_plan` |
| Skill library | EvolvingSkillLibrary | `research/advanced_features.py` |
| World models | SiteWorldModel, SkillWorldModel | `research/advanced_features.py` |

### 7. Sessions Beyond Context Memory Limits

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Episodic memory | MemexRL with learned write gate | `research/methods/memex_agent.py` |
| Memory retrieval | Attention-based read mechanism | `memex_agent.py:_read_memory()` |
| Importance scoring | Hindsight-weighted memory | `memex_agent.py:_compute_importance()` |
| Trajectory splitting | Overlapping segments | `training/trajectory_splitter.py` |

**MemexRL Memory System**:
```python
# From research/methods/memex_agent.py
class MemexRLAgent:
    def __init__(self):
        self.memory_size = 1000  # Large episodic buffer
        self.memory_dim = STATE_DIM + 1  # State + reward
        self.memory = np.zeros((self.memory_size, self.memory_dim))
        self.memory_importance = np.zeros(self.memory_size)
        self.write_gate = NeuralNetwork([STATE_DIM, 32, 1])  # Learned gating
```

---

## Sub-Theme Alignment

### Scale AI: Business Workflow Focus

Clinical trial recruitment **is** a business workflow in the healthcare/pharma sector:

| Business Domain | Mapping |
|-----------------|---------|
| **HR/Recruiting** | Patient screening ≈ Candidate screening |
| **Sales Funnel** | Patient funnel (contact→screen→consent→enroll) |
| **Project Management** | Trial timeline, milestones, resource allocation |
| **Resource Management** | Budget, site capacity, staff allocation |

**Key Business Metrics**:
- Cost per enrolled patient
- Time to enrollment target
- Site utilization efficiency
- Dropout/churn rate
- ROI on outreach campaigns

### Mercor: Token-Scaled Rewards

The environment implements **capped token-scaled rewards**:

| Feature | Implementation |
|---------|----------------|
| Token budget | 12,000 tokens per episode |
| Per-action costs | 50-500 tokens depending on complexity |
| Efficiency scoring | Rewards scale inversely with token usage |
| Budget throttling | Expensive actions penalized when budget low |

**Token Reward Scaling**:
```python
# From env.py
def _compute_reward(self, outcome: Dict[str, Any]) -> float:
    # Base reward
    r = outcome_reward
    
    # Token efficiency penalty
    token_penalty = min(0.05, token_cost / max(1, self._token_budget_total) * 1.2)
    r -= token_penalty
    
    # Progress bonus scaled by efficiency
    if self._token_efficiency_score > 0.5:
        progress_bonus = min(0.04, self._token_efficiency_score * 0.03)
        r += progress_bonus
    
    # Penalize efficiency degradation without progress
    if self._token_efficiency_score < efficiency_before and not outcome.get("enrolled"):
        r -= min(0.02, (efficiency_before - self._token_efficiency_score) * 0.1)
    
    return r
```

**Token Cost by Action**:
| Action | Token Cost | Rationale |
|--------|------------|-----------|
| `screen_patient` | 200-400 | Complex eligibility reasoning |
| `allocate_to_site` | 150-300 | Site matching logic |
| `plan_next_phase` | 50-100 | Strategic planning |
| `summarize_and_index` | 100-200 | Memory compression |
| `retrieve_relevant_history` | 50-100 | Memory query |
| `adjust_strategy` | 100-200 | Strategy evaluation |

---

## Benchmark Characteristics Summary

| Characteristic | Value | Theme #2 Relevance |
|----------------|-------|-------------------|
| Episode length | 180 steps | Long horizon |
| State dimension | 37 features | Rich state tracking |
| Action space | 10 discrete | Complex decision space |
| Reward sparsity | ~5-10 enrollments/episode | Sparse rewards |
| Delay range | 5-30 days | Delayed consequences |
| Memory capacity | 1000 entries (MemexRL) | Beyond context limits |
| Token budget | 12,000 tokens | Token-scaled rewards |

---

## Agent Performance on Theme #2 Tasks

| Agent | Mean Score | Key Capability |
|-------|------------|----------------|
| **HCAPO** | 0.234 | Hindsight credit assignment, goal decomposition |
| **MemexRL** | 0.226 | Episodic memory, beyond context limits |
| **MiRA** | 0.221 | Potential-based shaping, milestone tracking |
| **KLong** | 0.212 | Multi-scale temporal abstraction |

**Statistical Significance**: HCAPO > KLong (p=0.0075), suggesting hindsight credit assignment is more effective than pure temporal abstraction for this long-horizon task.

---

## Conclusion

The Clinical Trial Recruitment Environment directly addresses all Theme #2 requirements:

1. ✅ **Long-horizon planning**: 180-step episodes with sparse milestone rewards
2. ✅ **Goal decomposition**: Hierarchical subgoals with HCAPO
3. ✅ **State tracking**: 37-dim observations + episodic memory
4. ✅ **Error recovery**: Curriculum + hindsight relabeling + recontact
5. ✅ **Structured planning**: Plan-and-Act separation + skill library
6. ✅ **Beyond context limits**: MemexRL episodic memory (1000 entries)
7. ✅ **Business workflow**: Healthcare recruiting funnel (Scale AI)
8. ✅ **Token-scaled rewards**: 12K budget with efficiency shaping (Mercor)
