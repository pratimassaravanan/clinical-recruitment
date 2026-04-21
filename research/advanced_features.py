"""Advanced long-horizon features for clinical recruitment.

Implements remaining Tier 2-5 features:
- 13. Token-efficiency tracking (enhanced)
- 14. Multi-phase reward decomposition
- 15. Patient-level memory graph
- 16. Site performance world model (enhanced)
- 20. Multi-objective Pareto controller
- 21. SALT step-level advantage (enhanced)
- 22. Predictable skills + abstract skill world model
- 30. Async RL utilities (enhanced)
- 31. Realistic regulatory events
- 32. Patient engagement simulator
- 37. Curriculum injection logging
- 41. Multi-agent hierarchical oversight
- 43. Federated privacy simulation
- 45. Human-in-the-loop preference alignment
- 47. Skill library evolution
- 48. Long-horizon uncertainty quantification
- 42. Carbon-aware scaling
- 49. Cross-domain transfer
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 13. Token Efficiency Tracking
# ============================================================================

@dataclass
class TokenUsageTracker:
    """Track and optimize token usage for cost efficiency."""
    
    total_budget: int = 12000
    used_tokens: int = 0
    cost_per_1k_tokens: float = 0.002  # $0.002 per 1k tokens
    
    # Per-action token costs
    action_costs: Dict[str, int] = field(default_factory=lambda: {
        "screen_patient": 50,
        "recontact": 40,
        "allocate_to_site": 80,
        "adjust_strategy": 100,
        "plan_next_phase": 150,
        "summarize_and_index": 200,
        "retrieve_relevant_history": 120,
        "stop_recruitment": 10,
    })
    
    def record_action(self, action: str, actual_tokens: Optional[int] = None) -> int:
        """Record token usage for an action."""
        tokens = actual_tokens or self.action_costs.get(action, 50)
        self.used_tokens += tokens
        return tokens
    
    def get_efficiency_score(self) -> float:
        """Calculate token efficiency score (0-1)."""
        if self.total_budget <= 0:
            return 0.0
        remaining_ratio = max(0, self.total_budget - self.used_tokens) / self.total_budget
        return min(1.0, remaining_ratio * 1.2)  # Bonus for staying under budget
    
    def get_cost_usd(self) -> float:
        """Get current cost in USD."""
        return (self.used_tokens / 1000) * self.cost_per_1k_tokens
    
    def remaining_tokens(self) -> int:
        return max(0, self.total_budget - self.used_tokens)
    
    def should_throttle(self, threshold: float = 0.2) -> bool:
        """Check if we should throttle expensive actions."""
        return self.remaining_tokens() / max(1, self.total_budget) < threshold


# ============================================================================
# 14. Multi-Phase Reward Decomposition
# ============================================================================

@dataclass
class PhaseObjective:
    """An objective for a specific trial phase."""
    
    phase_name: str
    weight: float
    target_metric: str
    target_value: float
    current_value: float = 0.0
    
    def progress(self) -> float:
        if self.target_value <= 0:
            return 0.0
        return min(1.0, self.current_value / self.target_value)
    
    def weighted_reward(self) -> float:
        return self.progress() * self.weight


class MultiPhaseRewardDecomposer:
    """Decompose rewards into phase-specific objectives."""
    
    def __init__(self):
        self.phases = self._create_default_phases()
        self.current_phase_idx = 0
        self.phase_history: List[Dict[str, Any]] = []
    
    def _create_default_phases(self) -> List[List[PhaseObjective]]:
        """Create default phase objectives."""
        return [
            # Phase 1: Screening
            [
                PhaseObjective("screening", 0.4, "screened_count", 50),
                PhaseObjective("pipeline_build", 0.3, "consented_pending", 20),
                PhaseObjective("efficiency", 0.3, "budget_efficiency", 0.8),
            ],
            # Phase 2: Conversion
            [
                PhaseObjective("enrollment", 0.5, "enrollment_progress", 0.5),
                PhaseObjective("retention", 0.3, "retention_rate", 0.85),
                PhaseObjective("site_balance", 0.2, "site_utilization", 0.7),
            ],
            # Phase 3: Completion
            [
                PhaseObjective("target_completion", 0.6, "enrollment_progress", 1.0),
                PhaseObjective("final_retention", 0.25, "retention_rate", 0.9),
                PhaseObjective("budget_remaining", 0.15, "budget_ratio", 0.1),
            ],
        ]
    
    def update_objectives(self, state: Dict[str, Any]) -> None:
        """Update objective progress from state."""
        for phase in self.phases:
            for obj in phase:
                obj.current_value = state.get(obj.target_metric, 0.0)
    
    def compute_phase_reward(self, phase_idx: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        """Compute decomposed reward for a phase."""
        idx = phase_idx if phase_idx is not None else self.current_phase_idx
        if idx >= len(self.phases):
            idx = len(self.phases) - 1
        
        phase = self.phases[idx]
        breakdown = {}
        total = 0.0
        
        for obj in phase:
            r = obj.weighted_reward()
            breakdown[obj.phase_name] = r
            total += r
        
        return total, breakdown
    
    def check_phase_completion(self, threshold: float = 0.8) -> bool:
        """Check if current phase objectives are met."""
        _, breakdown = self.compute_phase_reward()
        avg_progress = sum(breakdown.values()) / max(1, len(breakdown))
        return avg_progress >= threshold
    
    def advance_phase(self) -> bool:
        """Advance to next phase."""
        if self.current_phase_idx < len(self.phases) - 1:
            self.phase_history.append({
                "phase": self.current_phase_idx,
                "rewards": self.compute_phase_reward()[1],
            })
            self.current_phase_idx += 1
            return True
        return False
    
    def get_current_phase_info(self) -> Dict[str, Any]:
        """Get current phase information."""
        return {
            "phase_index": self.current_phase_idx,
            "phase_count": len(self.phases),
            "objectives": [
                {
                    "name": obj.phase_name,
                    "weight": obj.weight,
                    "progress": obj.progress(),
                    "target": obj.target_value,
                    "current": obj.current_value,
                }
                for obj in self.phases[self.current_phase_idx]
            ],
        }


# ============================================================================
# 15. Patient-Level Memory Graph
# ============================================================================

@dataclass
class PatientNode:
    """A node in the patient memory graph."""
    
    patient_id: str
    attributes: Dict[str, Any]
    edges: Dict[str, List[str]] = field(default_factory=dict)  # edge_type -> [patient_ids]
    memory: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_edge(self, edge_type: str, target_id: str) -> None:
        if edge_type not in self.edges:
            self.edges[edge_type] = []
        if target_id not in self.edges[edge_type]:
            self.edges[edge_type].append(target_id)
    
    def add_memory(self, event: Dict[str, Any]) -> None:
        self.memory.append(event)


class PatientMemoryGraph:
    """Graph-based patient memory for relational reasoning."""
    
    def __init__(self):
        self.nodes: Dict[str, PatientNode] = {}
        self.edge_types = ["same_site", "similar_profile", "referred_by", "same_cohort"]
    
    def add_patient(self, patient_id: str, attributes: Dict[str, Any]) -> PatientNode:
        """Add a patient to the graph."""
        node = PatientNode(patient_id=patient_id, attributes=attributes)
        self.nodes[patient_id] = node
        
        # Auto-create edges based on attributes
        self._create_similarity_edges(node)
        
        return node
    
    def _create_similarity_edges(self, node: PatientNode) -> None:
        """Create edges based on patient similarity."""
        for other_id, other in self.nodes.items():
            if other_id == node.patient_id:
                continue
            
            # Same site
            if node.attributes.get("site_id") == other.attributes.get("site_id"):
                node.add_edge("same_site", other_id)
                other.add_edge("same_site", node.patient_id)
            
            # Similar profile (age within 10 years)
            age_diff = abs(
                node.attributes.get("age", 0) - other.attributes.get("age", 0)
            )
            if age_diff <= 10:
                node.add_edge("similar_profile", other_id)
    
    def record_event(self, patient_id: str, event: Dict[str, Any]) -> None:
        """Record an event in patient's memory."""
        if patient_id in self.nodes:
            self.nodes[patient_id].add_memory(event)
    
    def get_related_patients(
        self,
        patient_id: str,
        edge_type: Optional[str] = None,
        max_depth: int = 2,
    ) -> List[str]:
        """Get related patients via graph traversal."""
        if patient_id not in self.nodes:
            return []
        
        visited: Set[str] = {patient_id}
        current_level = [patient_id]
        related = []
        
        for _ in range(max_depth):
            next_level = []
            for pid in current_level:
                node = self.nodes.get(pid)
                if not node:
                    continue
                
                edges = node.edges.get(edge_type, []) if edge_type else []
                if not edge_type:
                    for et in self.edge_types:
                        edges.extend(node.edges.get(et, []))
                
                for neighbor in edges:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
                        related.append(neighbor)
            
            current_level = next_level
            if not current_level:
                break
        
        return related
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get summary of patient and relationships."""
        if patient_id not in self.nodes:
            return {}
        
        node = self.nodes[patient_id]
        return {
            "patient_id": patient_id,
            "attributes": node.attributes,
            "connections": {et: len(ids) for et, ids in node.edges.items()},
            "memory_events": len(node.memory),
            "recent_events": node.memory[-5:] if node.memory else [],
        }
    
    def get_cohort_insights(self, patient_ids: List[str]) -> Dict[str, Any]:
        """Get insights about a cohort of patients."""
        nodes = [self.nodes.get(pid) for pid in patient_ids if pid in self.nodes]
        
        if not nodes:
            return {}
        
        # Aggregate statistics
        ages = [n.attributes.get("age", 0) for n in nodes if n.attributes.get("age")]
        sites = [n.attributes.get("site_id") for n in nodes]
        
        return {
            "cohort_size": len(nodes),
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "site_distribution": {s: sites.count(s) for s in set(sites)},
            "total_connections": sum(
                sum(len(ids) for ids in n.edges.values())
                for n in nodes
            ),
        }


# ============================================================================
# 16. Site Performance World Model (Enhanced)
# ============================================================================

class SiteWorldModel:
    """Learned world model for site performance prediction."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.site_history: Dict[str, List[Dict[str, float]]] = {}
        self.model_params: Dict[str, Dict[str, float]] = {}
    
    def record_observation(self, site_id: str, metrics: Dict[str, float]) -> None:
        """Record site performance observation."""
        if site_id not in self.site_history:
            self.site_history[site_id] = []
        self.site_history[site_id].append(metrics)
        
        # Update simple moving average model
        self._update_model(site_id)
    
    def _update_model(self, site_id: str) -> None:
        """Update predictive model for site."""
        history = self.site_history.get(site_id, [])
        if len(history) < 2:
            return
        
        # Simple exponential moving average
        alpha = 0.3
        params = {}
        
        for key in history[-1].keys():
            values = [h.get(key, 0.0) for h in history]
            ema = values[0]
            for v in values[1:]:
                ema = alpha * v + (1 - alpha) * ema
            params[key] = ema
        
        self.model_params[site_id] = params
    
    def predict_performance(
        self,
        site_id: str,
        steps_ahead: int = 10,
    ) -> Dict[str, float]:
        """Predict future site performance."""
        if site_id not in self.model_params:
            return {"conversion_rate": 0.5, "retention_rate": 0.8, "capacity": 20}
        
        params = self.model_params[site_id]
        
        # Add some uncertainty for longer predictions
        noise_scale = 0.01 * steps_ahead
        
        return {
            key: max(0, min(1 if "rate" in key else 100, 
                          val + self.rng.gauss(0, noise_scale)))
            for key, val in params.items()
        }
    
    def rank_sites(self, objective: str = "conversion_rate") -> List[Tuple[str, float]]:
        """Rank sites by predicted performance."""
        rankings = []
        for site_id, params in self.model_params.items():
            score = params.get(objective, 0.0)
            rankings.append((site_id, score))
        
        return sorted(rankings, key=lambda x: -x[1])
    
    def get_site_recommendation(
        self,
        required_capacity: int,
        min_conversion: float = 0.5,
    ) -> Optional[str]:
        """Recommend best site meeting requirements."""
        for site_id, params in self.model_params.items():
            if (params.get("capacity", 0) >= required_capacity and
                params.get("conversion_rate", 0) >= min_conversion):
                return site_id
        return None


# ============================================================================
# 20. Multi-Objective Pareto Controller
# ============================================================================

@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    
    objectives: Dict[str, float]
    action_sequence: List[str]
    dominated: bool = False


class ParetoController:
    """Online multi-objective controller using Pareto optimization."""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ["enrollment", "budget", "retention"]
        self.frontier: List[ParetoPoint] = []
        self.history: List[Dict[str, Any]] = []
    
    def add_point(
        self,
        objectives: Dict[str, float],
        actions: List[str],
    ) -> bool:
        """Add a point and update Pareto frontier."""
        point = ParetoPoint(objectives=objectives, action_sequence=actions)
        
        # Check if dominated by existing points
        for existing in self.frontier:
            if self._dominates(existing.objectives, point.objectives):
                point.dominated = True
                return False
        
        # Remove points dominated by new point
        self.frontier = [
            p for p in self.frontier
            if not self._dominates(point.objectives, p.objectives)
        ]
        
        self.frontier.append(point)
        return True
    
    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Check if a dominates b (a is better in all objectives)."""
        dominated = True
        strictly_better = False
        
        for obj in self.objectives:
            if a.get(obj, 0) < b.get(obj, 0):
                dominated = False
                break
            if a.get(obj, 0) > b.get(obj, 0):
                strictly_better = True
        
        return dominated and strictly_better
    
    def get_recommendation(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[List[str]]:
        """Get recommended action sequence based on weighted objectives."""
        if not self.frontier:
            return None
        
        weights = weights or {obj: 1.0 for obj in self.objectives}
        
        best_point = None
        best_score = -float("inf")
        
        for point in self.frontier:
            score = sum(
                weights.get(obj, 1.0) * point.objectives.get(obj, 0.0)
                for obj in self.objectives
            )
            if score > best_score:
                best_score = score
                best_point = point
        
        return best_point.action_sequence if best_point else None
    
    def get_frontier_summary(self) -> Dict[str, Any]:
        """Get Pareto frontier summary."""
        if not self.frontier:
            return {"frontier_size": 0}
        
        return {
            "frontier_size": len(self.frontier),
            "objectives": self.objectives,
            "frontier_points": [
                {
                    "objectives": p.objectives,
                    "actions_preview": p.action_sequence[:3],
                }
                for p in self.frontier[:10]  # Top 10
            ],
        }


# ============================================================================
# 21. SALT Step-Level Advantage (Enhanced)
# ============================================================================

class SALTAdvantageComputer:
    """SALT-style step-level trajectory-graph advantage computation."""
    
    def __init__(self, gamma: float = 0.99, lambda_gae: float = 0.95):
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.trajectory_graph: Dict[str, List[str]] = {}  # state_hash -> [next_state_hashes]
        self.state_values: Dict[str, float] = {}
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash for state."""
        key_vals = sorted(
            (k, round(float(v), 2) if isinstance(v, (int, float)) else str(v))
            for k, v in state.items()
            if k in ["enrollment_progress", "budget_ratio", "step", "uncertainty"]
        )
        return hashlib.md5(str(key_vals).encode()).hexdigest()[:8]
    
    def add_transition(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Add transition to trajectory graph."""
        state_hash = self._hash_state(state)
        next_hash = self._hash_state(next_state)
        
        if state_hash not in self.trajectory_graph:
            self.trajectory_graph[state_hash] = []
        if next_hash not in self.trajectory_graph[state_hash]:
            self.trajectory_graph[state_hash].append(next_hash)
        
        # Update state value estimate
        if done:
            self.state_values[next_hash] = 0.0
        
        # Bellman update
        next_value = self.state_values.get(next_hash, 0.0)
        self.state_values[state_hash] = reward + self.gamma * next_value
    
    def compute_advantages(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> List[float]:
        """Compute GAE advantages for trajectory."""
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(trajectory))):
            t = trajectory[i]
            reward = t.get("reward", 0.0)
            
            state_hash = self._hash_state(t.get("state", {}))
            value = self.state_values.get(state_hash, 0.0)
            
            if i + 1 < len(trajectory):
                next_hash = self._hash_state(trajectory[i + 1].get("state", {}))
                next_value = self.state_values.get(next_hash, 0.0)
            else:
                next_value = 0.0
            
            done = t.get("done", False)
            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * self.lambda_gae * (1 - done) * gae
            advantages.append(gae)
        
        return list(reversed(advantages))
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get trajectory graph statistics."""
        return {
            "num_states": len(self.trajectory_graph),
            "num_values": len(self.state_values),
            "avg_branching": (
                sum(len(nexts) for nexts in self.trajectory_graph.values()) /
                max(1, len(self.trajectory_graph))
            ),
        }


# ============================================================================
# 22. Predictable Skills + Abstract Skill World Model
# ============================================================================

@dataclass
class Skill:
    """A learned skill with predictable outcomes."""
    
    skill_id: str
    name: str
    action_sequence: List[str]
    preconditions: Dict[str, float]  # metric -> min_value
    expected_effects: Dict[str, float]  # metric -> change
    success_rate: float = 0.8
    execution_count: int = 0
    
    def check_preconditions(self, state: Dict[str, Any]) -> bool:
        """Check if skill preconditions are met."""
        for metric, min_val in self.preconditions.items():
            if state.get(metric, 0.0) < min_val:
                return False
        return True
    
    def predict_outcome(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Predict state after skill execution."""
        result = dict(state)
        for metric, change in self.expected_effects.items():
            result[metric] = result.get(metric, 0.0) + change * self.success_rate
        return result


class SkillWorldModel:
    """Abstract world model using learned skills."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.skills = self._create_default_skills()
        self.skill_history: List[Dict[str, Any]] = []
    
    def _create_default_skills(self) -> Dict[str, Skill]:
        """Create default skill library."""
        return {
            "aggressive_screening": Skill(
                "aggressive_screening", "Aggressive Screening",
                action_sequence=["screen_patient"] * 5,
                preconditions={"budget_ratio": 0.3, "screening_backlog": 10},
                expected_effects={"screened_count": 4, "budget_ratio": -0.05},
            ),
            "conversion_push": Skill(
                "conversion_push", "Conversion Push",
                action_sequence=["allocate_to_site", "recontact", "allocate_to_site"],
                preconditions={"consented_pending": 5},
                expected_effects={"enrollment_progress": 0.1, "consented_pending": -3},
            ),
            "retention_focus": Skill(
                "retention_focus", "Retention Focus",
                action_sequence=["recontact", "adjust_strategy"],
                preconditions={"enrolled_count": 10, "dropout_risk": 0.2},
                expected_effects={"retention_rate": 0.05, "dropout_risk": -0.1},
            ),
            "site_optimization": Skill(
                "site_optimization", "Site Optimization",
                action_sequence=["adjust_strategy", "allocate_to_site"],
                preconditions={"site_bottleneck": True},
                expected_effects={"site_utilization": 0.1, "capacity_remaining": 5},
            ),
        }
    
    def get_applicable_skills(self, state: Dict[str, Any]) -> List[Skill]:
        """Get skills whose preconditions are met."""
        return [s for s in self.skills.values() if s.check_preconditions(state)]
    
    def plan_with_skills(
        self,
        state: Dict[str, Any],
        target: Dict[str, float],
        max_skills: int = 5,
    ) -> List[Skill]:
        """Plan sequence of skills to reach target."""
        plan = []
        current = dict(state)
        
        for _ in range(max_skills):
            applicable = self.get_applicable_skills(current)
            if not applicable:
                break
            
            # Choose skill that moves closest to target
            best_skill = None
            best_progress = -float("inf")
            
            for skill in applicable:
                predicted = skill.predict_outcome(current)
                progress = sum(
                    (predicted.get(m, 0) - current.get(m, 0)) * 
                    (1 if target.get(m, 0) > current.get(m, 0) else -1)
                    for m in target.keys()
                )
                if progress > best_progress:
                    best_progress = progress
                    best_skill = skill
            
            if best_skill and best_progress > 0:
                plan.append(best_skill)
                current = best_skill.predict_outcome(current)
            else:
                break
        
        return plan
    
    def execute_skill(
        self,
        skill_id: str,
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, float], bool]:
        """Execute skill and return (new_state, success)."""
        if skill_id not in self.skills:
            return state, False
        
        skill = self.skills[skill_id]
        
        if not skill.check_preconditions(state):
            return state, False
        
        success = self.rng.random() < skill.success_rate
        skill.execution_count += 1
        
        if success:
            new_state = skill.predict_outcome(state)
        else:
            # Partial effect on failure
            new_state = dict(state)
            for m, change in skill.expected_effects.items():
                new_state[m] = new_state.get(m, 0) + change * 0.3
        
        self.skill_history.append({
            "skill": skill_id,
            "success": success,
            "state_before": state,
            "state_after": new_state,
        })
        
        return new_state, success
    
    def evolve_skills(self) -> None:
        """Evolve skill library based on execution history."""
        # Update success rates based on history
        for skill_id, skill in self.skills.items():
            relevant = [h for h in self.skill_history if h["skill"] == skill_id]
            if len(relevant) >= 5:
                success_rate = sum(1 for h in relevant if h["success"]) / len(relevant)
                skill.success_rate = 0.7 * skill.success_rate + 0.3 * success_rate


# ============================================================================
# 30. Async RL Utilities (Enhanced)
# ============================================================================

class AsyncRLCoordinator:
    """Coordinator for async RL training across workers."""
    
    def __init__(self, num_workers: int = 4, seed: int = 42):
        self.num_workers = num_workers
        self.rng = random.Random(seed)
        self.worker_states: Dict[int, Dict[str, Any]] = {}
        self.global_buffer: List[Dict[str, Any]] = []
        self.global_step = 0
    
    def init_workers(self, task_ids: List[str]) -> None:
        """Initialize worker states."""
        for i in range(self.num_workers):
            self.worker_states[i] = {
                "task": self.rng.choice(task_ids),
                "step": 0,
                "episodes": 0,
                "total_reward": 0.0,
            }
    
    def collect_experience(
        self,
        worker_id: int,
        transitions: List[Dict[str, Any]],
    ) -> None:
        """Collect experience from a worker."""
        self.global_buffer.extend(transitions)
        self.worker_states[worker_id]["step"] += len(transitions)
        
        # Trim buffer
        if len(self.global_buffer) > 10000:
            self.global_buffer = self.global_buffer[-5000:]
    
    def sample_batch(self, batch_size: int = 64) -> List[Dict[str, Any]]:
        """Sample batch from global buffer."""
        if len(self.global_buffer) < batch_size:
            return self.global_buffer[:]
        return self.rng.sample(self.global_buffer, batch_size)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "num_workers": self.num_workers,
            "global_step": self.global_step,
            "buffer_size": len(self.global_buffer),
            "workers": self.worker_states,
        }


# ============================================================================
# 31. Realistic Regulatory Events
# ============================================================================

@dataclass
class RegulatoryEvent:
    """A regulatory event affecting the trial."""
    
    event_id: str
    event_type: str
    description: str
    duration_days: int
    effects: Dict[str, Any]
    resolved: bool = False
    start_step: Optional[int] = None


class RegulatoryEventSimulator:
    """Simulate realistic regulatory events."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.active_events: List[RegulatoryEvent] = []
        self.event_history: List[RegulatoryEvent] = []
        self.event_templates = self._create_event_templates()
    
    def _create_event_templates(self) -> List[RegulatoryEvent]:
        """Create regulatory event templates."""
        return [
            RegulatoryEvent(
                "irb_amendment", "IRB Amendment Required",
                "Protocol amendment requires IRB review",
                duration_days=14,
                effects={"enrollment_blocked": True, "new_screening_blocked": True},
            ),
            RegulatoryEvent(
                "safety_review", "Safety Review",
                "Safety signal requires DSMB review",
                duration_days=21,
                effects={"enrollment_paused": True, "reporting_required": True},
            ),
            RegulatoryEvent(
                "consent_revision", "Consent Form Revision",
                "Informed consent must be updated",
                duration_days=10,
                effects={"new_enrollments_blocked": True, "reconsent_required": True},
            ),
            RegulatoryEvent(
                "site_audit", "Regulatory Site Audit",
                "Sponsor audit at primary site",
                duration_days=5,
                effects={"site_capacity_reduced": 0.5, "documentation_required": True},
            ),
            RegulatoryEvent(
                "fda_inquiry", "FDA Information Request",
                "FDA requests additional safety data",
                duration_days=30,
                effects={"enrollment_paused": True, "priority_reporting": True},
            ),
        ]
    
    def check_for_events(self, step: int, state: Dict[str, Any]) -> Optional[RegulatoryEvent]:
        """Check if a regulatory event should occur."""
        # Base probability of event
        base_prob = 0.01
        
        # Higher probability if there are safety signals
        if state.get("dropout_rate", 0) > 0.2:
            base_prob += 0.02
        if state.get("adverse_events", 0) > 0:
            base_prob += 0.03
        
        if self.rng.random() < base_prob:
            template = self.rng.choice(self.event_templates)
            event = RegulatoryEvent(
                event_id=f"{template.event_id}_{step}",
                event_type=template.event_type,
                description=template.description,
                duration_days=template.duration_days,
                effects=dict(template.effects),
                start_step=step,
            )
            self.active_events.append(event)
            return event
        
        return None
    
    def process_step(self, step: int) -> List[RegulatoryEvent]:
        """Process active events and return newly resolved ones."""
        resolved = []
        still_active = []
        
        for event in self.active_events:
            if event.start_step and step - event.start_step >= event.duration_days:
                event.resolved = True
                resolved.append(event)
                self.event_history.append(event)
            else:
                still_active.append(event)
        
        self.active_events = still_active
        return resolved
    
    def get_active_effects(self) -> Dict[str, Any]:
        """Get combined effects of all active events."""
        effects = {}
        for event in self.active_events:
            effects.update(event.effects)
        return effects


# ============================================================================
# 32. Patient Engagement Simulator
# ============================================================================

class PatientEngagementSimulator:
    """Simulate patient engagement and willingness over time."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.patients: Dict[str, Dict[str, float]] = {}
    
    def init_patient(self, patient_id: str, initial_willingness: float = 0.7) -> None:
        """Initialize patient engagement state."""
        self.patients[patient_id] = {
            "willingness": initial_willingness,
            "contact_count": 0,
            "last_contact_step": 0,
            "engagement_score": 0.5,
            "fatigue": 0.0,
        }
    
    def simulate_contact(
        self,
        patient_id: str,
        current_step: int,
        contact_type: str = "standard",
    ) -> Dict[str, Any]:
        """Simulate a contact attempt with patient."""
        if patient_id not in self.patients:
            self.init_patient(patient_id)
        
        p = self.patients[patient_id]
        
        # Update contact tracking
        p["contact_count"] += 1
        steps_since_last = current_step - p["last_contact_step"]
        p["last_contact_step"] = current_step
        
        # Contact fatigue
        if steps_since_last < 3:
            p["fatigue"] += 0.1
        else:
            p["fatigue"] = max(0, p["fatigue"] - 0.05)
        
        # Willingness decay
        p["willingness"] *= (1 - p["fatigue"] * 0.1)
        
        # Engagement update based on contact type
        engagement_boost = {
            "standard": 0.0,
            "personalized": 0.05,
            "reminder": -0.02,
            "incentive": 0.1,
        }.get(contact_type, 0.0)
        
        p["engagement_score"] = min(1.0, max(0, p["engagement_score"] + engagement_boost))
        
        # Response probability
        response_prob = p["willingness"] * p["engagement_score"] * (1 - p["fatigue"])
        responded = self.rng.random() < response_prob
        
        return {
            "patient_id": patient_id,
            "responded": responded,
            "response_probability": response_prob,
            "willingness": p["willingness"],
            "engagement": p["engagement_score"],
            "fatigue": p["fatigue"],
        }
    
    def get_recontact_priority(self, patient_ids: List[str]) -> List[Tuple[str, float]]:
        """Rank patients by recontact priority."""
        priorities = []
        for pid in patient_ids:
            if pid in self.patients:
                p = self.patients[pid]
                # Priority based on willingness, low fatigue, high engagement
                priority = p["willingness"] * (1 - p["fatigue"]) * p["engagement_score"]
                priorities.append((pid, priority))
        
        return sorted(priorities, key=lambda x: -x[1])


# ============================================================================
# 37. Curriculum Injection Logging
# ============================================================================

class CurriculumLogger:
    """Comprehensive logging for curriculum events."""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.event_counts: Dict[str, int] = {}
    
    def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        step: int,
    ) -> None:
        """Log a curriculum event."""
        entry = {
            "timestamp": time.time(),
            "step": step,
            "event_type": event_type,
            "details": details,
        }
        self.logs.append(entry)
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
    
    def log_level_change(self, old_level: str, new_level: str, step: int) -> None:
        self.log_event("level_change", {
            "old_level": old_level,
            "new_level": new_level,
        }, step)
    
    def log_task_completion(
        self,
        task_id: str,
        score: float,
        step: int,
    ) -> None:
        self.log_event("task_completion", {
            "task_id": task_id,
            "score": score,
        }, step)
    
    def log_recovery_attempt(
        self,
        scenario: str,
        success: bool,
        step: int,
    ) -> None:
        self.log_event("recovery_attempt", {
            "scenario": scenario,
            "success": success,
        }, step)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get logging summary."""
        return {
            "total_events": len(self.logs),
            "event_counts": self.event_counts,
            "recent_events": self.logs[-10:] if self.logs else [],
        }
    
    def export_logs(self) -> List[Dict[str, Any]]:
        """Export all logs."""
        return self.logs[:]


# ============================================================================
# 41. Multi-Agent Hierarchical Oversight
# ============================================================================

@dataclass
class OversightAgent:
    """An agent in the oversight hierarchy."""
    
    agent_id: str
    role: str  # monitor, reviewer, approver
    level: int  # 0 = lowest, higher = more authority
    review_threshold: float  # Action score threshold for review
    approval_rate: float = 0.8


class HierarchicalOversightSystem:
    """Multi-agent hierarchical oversight for action review."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.agents = self._create_default_hierarchy()
        self.review_queue: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
    
    def _create_default_hierarchy(self) -> List[OversightAgent]:
        """Create default oversight hierarchy."""
        return [
            OversightAgent("monitor_1", "monitor", 0, review_threshold=0.3),
            OversightAgent("reviewer_1", "reviewer", 1, review_threshold=0.5, approval_rate=0.7),
            OversightAgent("approver_1", "approver", 2, review_threshold=0.7, approval_rate=0.9),
        ]
    
    def submit_action(
        self,
        action: str,
        risk_score: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit action for oversight review."""
        result = {
            "action": action,
            "risk_score": risk_score,
            "reviews": [],
            "approved": True,
            "escalated": False,
        }
        
        for agent in self.agents:
            if risk_score >= agent.review_threshold:
                # This level needs to review
                approved = self.rng.random() < agent.approval_rate
                
                result["reviews"].append({
                    "agent": agent.agent_id,
                    "role": agent.role,
                    "level": agent.level,
                    "approved": approved,
                })
                
                if not approved:
                    if agent.level < len(self.agents) - 1:
                        result["escalated"] = True
                    else:
                        result["approved"] = False
                        break
        
        self.decisions.append(result)
        return result
    
    def get_oversight_stats(self) -> Dict[str, Any]:
        """Get oversight statistics."""
        if not self.decisions:
            return {"total_reviews": 0}
        
        approved = sum(1 for d in self.decisions if d["approved"])
        escalated = sum(1 for d in self.decisions if d["escalated"])
        
        return {
            "total_reviews": len(self.decisions),
            "approval_rate": approved / len(self.decisions),
            "escalation_rate": escalated / len(self.decisions),
            "by_agent": {
                agent.agent_id: sum(
                    1 for d in self.decisions
                    for r in d["reviews"]
                    if r["agent"] == agent.agent_id
                )
                for agent in self.agents
            },
        }


# ============================================================================
# 43. Federated Privacy Simulation
# ============================================================================

class FederatedPrivacySimulator:
    """Simulate federated learning with privacy guarantees."""
    
    def __init__(
        self,
        num_sites: int = 5,
        epsilon: float = 1.0,  # Differential privacy parameter
        seed: int = 42,
    ):
        self.num_sites = num_sites
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.site_data: Dict[str, List[Dict[str, Any]]] = {
            f"site_{i}": [] for i in range(num_sites)
        }
        self.global_model: Dict[str, float] = {}
    
    def add_local_data(self, site_id: str, data: Dict[str, Any]) -> None:
        """Add data to a site's local store."""
        if site_id in self.site_data:
            self.site_data[site_id].append(data)
    
    def _add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = self.rng.gauss(0, scale)
        return value + noise
    
    def compute_local_gradients(self, site_id: str) -> Dict[str, float]:
        """Compute local gradients with privacy."""
        data = self.site_data.get(site_id, [])
        if not data:
            return {}
        
        # Simulate gradient computation
        gradients = {}
        for key in ["enrollment_rate", "retention_rate", "efficiency"]:
            values = [d.get(key, 0) for d in data if key in d]
            if values:
                mean = sum(values) / len(values)
                # Add noise for privacy
                gradients[key] = self._add_noise(mean)
        
        return gradients
    
    def federated_average(self) -> Dict[str, float]:
        """Compute federated average of all sites."""
        all_gradients: Dict[str, List[float]] = {}
        
        for site_id in self.site_data.keys():
            grads = self.compute_local_gradients(site_id)
            for key, val in grads.items():
                if key not in all_gradients:
                    all_gradients[key] = []
                all_gradients[key].append(val)
        
        # Average with weights
        self.global_model = {
            key: sum(vals) / len(vals)
            for key, vals in all_gradients.items()
            if vals
        }
        
        return self.global_model
    
    def get_privacy_budget_remaining(self) -> float:
        """Estimate remaining privacy budget."""
        queries = sum(len(data) for data in self.site_data.values())
        # Simple composition
        used = queries * 0.01
        return max(0, self.epsilon - used)


# ============================================================================
# 45. Human-in-the-Loop Preference Alignment
# ============================================================================

@dataclass
class PreferencePair:
    """A pair of trajectories with preference label."""
    
    trajectory_a: List[Dict[str, Any]]
    trajectory_b: List[Dict[str, Any]]
    preferred: str  # "a", "b", or "equal"
    confidence: float = 1.0
    annotator: str = "human"


class PreferenceAligner:
    """Human-in-the-loop preference alignment system."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.preferences: List[PreferencePair] = []
        self.reward_model: Dict[str, float] = {}
    
    def add_preference(
        self,
        traj_a: List[Dict[str, Any]],
        traj_b: List[Dict[str, Any]],
        preferred: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a preference annotation."""
        self.preferences.append(PreferencePair(
            trajectory_a=traj_a,
            trajectory_b=traj_b,
            preferred=preferred,
            confidence=confidence,
        ))
    
    def _extract_features(self, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract features from trajectory."""
        if not trajectory:
            return {}
        
        rewards = [t.get("reward", 0) for t in trajectory]
        enrollments = [t.get("enrolled", 0) for t in trajectory]
        
        return {
            "total_reward": sum(rewards),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "final_enrollment": enrollments[-1] if enrollments else 0,
            "trajectory_length": len(trajectory),
        }
    
    def update_reward_model(self) -> None:
        """Update reward model from preferences."""
        if not self.preferences:
            return
        
        feature_weights = {
            "total_reward": 0.3,
            "avg_reward": 0.2,
            "final_enrollment": 0.4,
            "trajectory_length": 0.1,
        }
        
        # Simple Bradley-Terry update
        for pref in self.preferences:
            feat_a = self._extract_features(pref.trajectory_a)
            feat_b = self._extract_features(pref.trajectory_b)
            
            if pref.preferred == "a":
                for key in feature_weights:
                    if feat_a.get(key, 0) > feat_b.get(key, 0):
                        feature_weights[key] *= 1.1
            elif pref.preferred == "b":
                for key in feature_weights:
                    if feat_b.get(key, 0) > feat_a.get(key, 0):
                        feature_weights[key] *= 1.1
        
        # Normalize
        total = sum(feature_weights.values())
        self.reward_model = {k: v / total for k, v in feature_weights.items()}
    
    def score_trajectory(self, trajectory: List[Dict[str, Any]]) -> float:
        """Score a trajectory using learned reward model."""
        features = self._extract_features(trajectory)
        score = 0.0
        
        for key, weight in self.reward_model.items():
            score += features.get(key, 0) * weight
        
        return score
    
    def get_preference_stats(self) -> Dict[str, Any]:
        """Get preference statistics."""
        if not self.preferences:
            return {"total_preferences": 0}
        
        a_preferred = sum(1 for p in self.preferences if p.preferred == "a")
        b_preferred = sum(1 for p in self.preferences if p.preferred == "b")
        
        return {
            "total_preferences": len(self.preferences),
            "a_preferred": a_preferred,
            "b_preferred": b_preferred,
            "equal": len(self.preferences) - a_preferred - b_preferred,
            "avg_confidence": sum(p.confidence for p in self.preferences) / len(self.preferences),
        }


# ============================================================================
# 47. Skill Library Evolution
# ============================================================================

class EvolvingSkillLibrary:
    """Skill library that evolves based on execution history."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.skills: Dict[str, Skill] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.generation = 0
    
    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the library."""
        self.skills[skill.skill_id] = skill
    
    def record_execution(
        self,
        skill_id: str,
        success: bool,
        actual_effects: Dict[str, float],
    ) -> None:
        """Record skill execution outcome."""
        self.execution_log.append({
            "skill_id": skill_id,
            "success": success,
            "actual_effects": actual_effects,
            "generation": self.generation,
        })
    
    def evolve(self) -> List[str]:
        """Evolve skills based on execution history."""
        changes = []
        self.generation += 1
        
        for skill_id, skill in list(self.skills.items()):
            executions = [
                e for e in self.execution_log
                if e["skill_id"] == skill_id
            ]
            
            if len(executions) < 3:
                continue
            
            # Update success rate
            recent = executions[-10:]
            new_success_rate = sum(1 for e in recent if e["success"]) / len(recent)
            
            if abs(new_success_rate - skill.success_rate) > 0.1:
                skill.success_rate = 0.7 * skill.success_rate + 0.3 * new_success_rate
                changes.append(f"Updated {skill_id} success rate to {skill.success_rate:.2f}")
            
            # Update expected effects based on actual
            for e in recent:
                for metric, actual in e["actual_effects"].items():
                    if metric in skill.expected_effects:
                        expected = skill.expected_effects[metric]
                        skill.expected_effects[metric] = 0.8 * expected + 0.2 * actual
            
            # Remove underperforming skills
            if skill.success_rate < 0.2 and skill.execution_count > 20:
                del self.skills[skill_id]
                changes.append(f"Removed underperforming skill {skill_id}")
        
        return changes
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get skill library statistics."""
        return {
            "num_skills": len(self.skills),
            "generation": self.generation,
            "total_executions": len(self.execution_log),
            "skills": {
                sid: {
                    "success_rate": s.success_rate,
                    "execution_count": s.execution_count,
                }
                for sid, s in self.skills.items()
            },
        }


# ============================================================================
# 48. Long-Horizon Uncertainty Quantification
# ============================================================================

@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate with epistemic/aleatoric decomposition."""
    
    total: float
    epistemic: float  # Model uncertainty (reducible)
    aleatoric: float  # Data/environment uncertainty (irreducible)
    sources: Dict[str, float]  # Per-source breakdown


class LongHorizonUncertaintyQuantifier:
    """Quantify uncertainty over long horizons."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.history: List[Dict[str, Any]] = []
        self.uncertainty_sources = [
            "patient_pool",
            "site_capacity",
            "dropout_rate",
            "regulatory",
            "market_conditions",
        ]
    
    def estimate_uncertainty(
        self,
        state: Dict[str, Any],
        horizon: int = 30,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for given state and horizon."""
        
        sources = {}
        
        # Patient pool uncertainty
        pool_size = state.get("screening_backlog", 50)
        sources["patient_pool"] = 0.5 / (1 + pool_size / 20)
        
        # Site capacity uncertainty
        capacity = state.get("available_capacity", 30)
        sources["site_capacity"] = 0.3 / (1 + capacity / 10)
        
        # Dropout rate uncertainty (increases with enrolled)
        enrolled = state.get("enrolled", 0)
        sources["dropout_rate"] = min(0.4, enrolled * 0.01)
        
        # Regulatory uncertainty (constant baseline)
        sources["regulatory"] = 0.1
        
        # Market/external uncertainty (increases with horizon)
        sources["market_conditions"] = min(0.3, horizon * 0.005)
        
        total = sum(sources.values())
        
        # Decompose into epistemic (model uncertainty) vs aleatoric (noise)
        epistemic = sources["patient_pool"] + sources["regulatory"]
        aleatoric = total - epistemic
        
        return UncertaintyEstimate(
            total=min(1.0, total),
            epistemic=min(1.0, epistemic),
            aleatoric=min(1.0, aleatoric),
            sources=sources,
        )
    
    def update_from_observation(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float],
    ) -> None:
        """Update uncertainty model from prediction error."""
        errors = {
            key: abs(predicted.get(key, 0) - actual.get(key, 0))
            for key in set(predicted.keys()) | set(actual.keys())
        }
        
        self.history.append({
            "predicted": predicted,
            "actual": actual,
            "errors": errors,
        })
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get uncertainty calibration statistics."""
        if not self.history:
            return {"calibration_error": 0.0}
        
        total_error = sum(
            sum(h["errors"].values())
            for h in self.history
        )
        
        return {
            "num_observations": len(self.history),
            "avg_error": total_error / len(self.history),
            "recent_errors": [h["errors"] for h in self.history[-5:]],
        }


# ============================================================================
# 42. Carbon-Aware Scaling
# ============================================================================

@dataclass
class CarbonMetrics:
    """Carbon footprint metrics."""
    
    compute_kwh: float = 0.0
    carbon_intensity_gco2_per_kwh: float = 400.0  # Global average
    total_gco2: float = 0.0


class CarbonAwareScaler:
    """Scale compute based on carbon footprint."""
    
    def __init__(
        self,
        carbon_budget_gco2: float = 1000.0,
        region_intensities: Optional[Dict[str, float]] = None,
    ):
        self.carbon_budget = carbon_budget_gco2
        self.metrics = CarbonMetrics()
        self.region_intensities = region_intensities or {
            "us-west": 200,
            "us-east": 400,
            "eu-west": 300,
            "asia": 600,
        }
        self.current_region = "us-west"
    
    def record_compute(
        self,
        operation: str,
        duration_seconds: float,
        gpu_power_watts: float = 200,
    ) -> float:
        """Record compute usage and return carbon cost."""
        kwh = (gpu_power_watts * duration_seconds / 3600) / 1000
        self.metrics.compute_kwh += kwh
        
        co2 = kwh * self.region_intensities[self.current_region]
        self.metrics.total_gco2 += co2
        
        return co2
    
    def should_scale_down(self) -> bool:
        """Check if compute should be scaled down due to carbon budget."""
        return self.metrics.total_gco2 > self.carbon_budget * 0.8
    
    def get_cheapest_region(self) -> str:
        """Get region with lowest carbon intensity."""
        return min(self.region_intensities, key=self.region_intensities.get)
    
    def get_carbon_stats(self) -> Dict[str, Any]:
        """Get carbon footprint statistics."""
        return {
            "total_kwh": self.metrics.compute_kwh,
            "total_gco2": self.metrics.total_gco2,
            "budget_remaining": max(0, self.carbon_budget - self.metrics.total_gco2),
            "budget_used_pct": self.metrics.total_gco2 / self.carbon_budget * 100,
            "current_region": self.current_region,
            "recommended_region": self.get_cheapest_region(),
        }


# ============================================================================
# 49. Cross-Domain Transfer
# ============================================================================

@dataclass
class DomainConfig:
    """Configuration for a domain."""
    
    domain_id: str
    name: str
    state_mapping: Dict[str, str]  # source_key -> target_key
    action_mapping: Dict[str, str]  # source_action -> target_action
    reward_scale: float = 1.0


class CrossDomainTransfer:
    """Transfer learning across domains."""
    
    def __init__(self):
        self.source_domain: Optional[DomainConfig] = None
        self.target_domain: Optional[DomainConfig] = None
        self.transfer_history: List[Dict[str, Any]] = []
    
    def set_source_domain(self, config: DomainConfig) -> None:
        """Set the source domain for transfer."""
        self.source_domain = config
    
    def set_target_domain(self, config: DomainConfig) -> None:
        """Set the target domain for transfer."""
        self.target_domain = config
    
    def transfer_state(self, source_state: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer state from source to target domain."""
        if not self.target_domain:
            return source_state
        
        target_state = {}
        for source_key, target_key in self.target_domain.state_mapping.items():
            if source_key in source_state:
                target_state[target_key] = source_state[source_key]
        
        return target_state
    
    def transfer_action(self, source_action: str) -> str:
        """Transfer action from source to target domain."""
        if not self.target_domain:
            return source_action
        
        return self.target_domain.action_mapping.get(source_action, source_action)
    
    def transfer_policy(
        self,
        source_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer policy weights between domains."""
        if not self.source_domain or not self.target_domain:
            return source_policy
        
        # Simple weight transfer with mapping
        target_policy = {}
        
        for key, value in source_policy.items():
            # Map keys if applicable
            new_key = self.target_domain.state_mapping.get(key, key)
            target_policy[new_key] = value
        
        self.transfer_history.append({
            "source": self.source_domain.domain_id,
            "target": self.target_domain.domain_id,
            "keys_transferred": len(target_policy),
        })
        
        return target_policy
    
    def evaluate_transfer(
        self,
        source_performance: float,
        target_performance: float,
    ) -> Dict[str, Any]:
        """Evaluate transfer effectiveness."""
        transfer_ratio = target_performance / max(0.01, source_performance)
        
        return {
            "source_performance": source_performance,
            "target_performance": target_performance,
            "transfer_ratio": transfer_ratio,
            "positive_transfer": transfer_ratio > 0.8,
            "negative_transfer": transfer_ratio < 0.5,
        }
    
    @classmethod
    def create_clinical_to_marketing_transfer(cls) -> "CrossDomainTransfer":
        """Create transfer config from clinical to marketing domain."""
        transfer = cls()
        
        transfer.set_source_domain(DomainConfig(
            domain_id="clinical_recruitment",
            name="Clinical Trial Recruitment",
            state_mapping={},
            action_mapping={},
        ))
        
        transfer.set_target_domain(DomainConfig(
            domain_id="marketing_funnel",
            name="Marketing Funnel Optimization",
            state_mapping={
                "enrollment_progress": "conversion_rate",
                "screening_backlog": "lead_pipeline",
                "budget_remaining": "ad_budget",
                "site_capacity": "channel_capacity",
                "retention_rate": "customer_retention",
            },
            action_mapping={
                "screen_patient": "qualify_lead",
                "allocate_to_site": "route_to_channel",
                "recontact": "nurture_lead",
                "adjust_strategy": "adjust_campaign",
            },
            reward_scale=0.8,
        ))
        
        return transfer
