"""Counterfactual Branch-Rollout Simulator.

Full implementation of counterfactual reasoning for long-horizon planning.

Key features:
1. Environment state forking - create lightweight snapshots
2. Branch rollout - simulate alternative action sequences
3. Outcome comparison - measure regret and opportunity cost
4. Causal analysis - identify critical decision points
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StateSnapshot:
    """Lightweight snapshot of environment state for forking."""

    step: int
    enrolled: int
    target: int
    budget_remaining: float
    budget_initial: float
    deadline: int
    screening_backlog: int
    uncertainty: float
    active_constraints: Dict[str, Any]
    patient_memory: Dict[str, int]
    milestones: Dict[str, bool]  # e.g., {"25%": True, "50%": False}
    active_milestone: Optional[str]
    milestone_potential: float
    sites: Dict[str, Dict[str, float]]  # Site info including capacity
    pareto_frontier: List[Dict[str, Any]]
    hypothesis_history: List[str]
    recent_dropouts: List[int]
    # Core patient state
    patients: List[Dict[str, Any]]  # Patient list from env


@dataclass
class RolloutResult:
    """Result of a counterfactual rollout."""

    actions: List[str]
    final_enrolled: int
    final_budget: float
    steps_taken: int
    milestones_reached: List[str]
    total_reward: float
    trajectory: List[Dict[str, Any]]


@dataclass
class CounterfactualAnalysis:
    """Analysis comparing actual vs counterfactual outcomes."""

    actual_outcome: RolloutResult
    counterfactual_outcomes: List[Tuple[str, RolloutResult]]  # (branch_name, result)
    regret: float  # How much worse was actual vs best counterfactual
    opportunity_cost: float  # Value of best foregone alternative
    critical_step: int  # Step where divergence was most impactful
    recommended_action: str  # What should have been done


class CounterfactualSimulator:
    """Simulates counterfactual branches from environment state."""

    def __init__(
        self,
        lookahead_steps: int = 10,
        num_rollouts_per_branch: int = 3,
        discount: float = 0.99,
        seed: Optional[int] = None,
    ):
        self.lookahead_steps = lookahead_steps
        self.num_rollouts_per_branch = num_rollouts_per_branch
        self.discount = discount
        self.rng = random.Random(seed)

    def snapshot_from_env(self, env: Any) -> StateSnapshot:
        """Create a lightweight snapshot from the full environment."""
        return StateSnapshot(
            step=env._step,
            enrolled=env._enrolled,
            target=env._target,
            budget_remaining=env._budget_remaining,
            budget_initial=env._trace.get("budget", env._budget_remaining),
            deadline=env._deadline_days,
            screening_backlog=env._screening_backlog,
            uncertainty=env._uncertainty,
            active_constraints=copy.deepcopy(env._active_constraints),
            patient_memory=copy.deepcopy(env._patient_memory),
            milestones=copy.deepcopy(env._milestones),
            active_milestone=env._active_milestone,
            milestone_potential=env._milestone_potential,
            sites=copy.deepcopy(env._sites),
            pareto_frontier=copy.deepcopy(env._pareto_frontier),
            hypothesis_history=list(env._hypothesis_history),
            recent_dropouts=list(env._recent_dropouts),
            patients=copy.deepcopy(env._patients[:20]),  # Only top 20 for efficiency
        )

    def rollout_from_snapshot(
        self,
        snapshot: StateSnapshot,
        action_sequence: List[str],
    ) -> RolloutResult:
        """Simulate a rollout from a snapshot using the given action sequence.
        
        Uses a simplified dynamics model rather than full env simulation.
        """
        # Initialize simulated state
        enrolled = snapshot.enrolled
        budget = snapshot.budget_remaining
        step = snapshot.step
        backlog = snapshot.screening_backlog
        consented_pending = snapshot.patient_memory.get("consented_pending_allocation", 0)
        followup_due = snapshot.patient_memory.get("followup_due", 0)
        milestones = [k for k, v in snapshot.milestones.items() if v]  # List of achieved milestones
        trajectory = []
        total_reward = 0.0

        # Action effect parameters (simplified model)
        SCREEN_COST = 50
        ALLOCATE_COST = 100
        RECONTACT_COST = 30
        NEGOTIATE_COST = 200

        SCREEN_SUCCESS_PROB = 0.4
        CONSENT_PROB = 0.6
        ALLOCATE_SUCCESS_PROB = 0.7
        RECONTACT_SUCCESS_PROB = 0.3
        DROPOUT_PROB = 0.05

        for i, action in enumerate(action_sequence):
            if step >= snapshot.deadline or budget <= 0:
                break

            reward = 0.0
            step += 1

            if action == "screen_patient":
                if budget >= SCREEN_COST and backlog > 0:
                    budget -= SCREEN_COST
                    backlog -= 1
                    if self.rng.random() < SCREEN_SUCCESS_PROB:
                        if self.rng.random() < CONSENT_PROB:
                            consented_pending += 1
                            reward = 0.1
                        else:
                            followup_due += 1
                            reward = 0.02

            elif action == "allocate_to_site":
                if budget >= ALLOCATE_COST and consented_pending > 0:
                    budget -= ALLOCATE_COST
                    consented_pending -= 1
                    if self.rng.random() < ALLOCATE_SUCCESS_PROB:
                        enrolled += 1
                        reward = 0.5
                        # Check milestone
                        progress = enrolled / max(1, snapshot.target)
                        if progress >= 0.25 and "25%" not in milestones:
                            milestones.append("25%")
                            reward += 0.2
                        elif progress >= 0.50 and "50%" not in milestones:
                            milestones.append("50%")
                            reward += 0.3
                        elif progress >= 0.75 and "75%" not in milestones:
                            milestones.append("75%")
                            reward += 0.4
                        elif progress >= 1.0 and "100%" not in milestones:
                            milestones.append("100%")
                            reward += 1.0

            elif action == "recontact":
                if budget >= RECONTACT_COST and followup_due > 0:
                    budget -= RECONTACT_COST
                    followup_due -= 1
                    if self.rng.random() < RECONTACT_SUCCESS_PROB:
                        consented_pending += 1
                        reward = 0.08

            elif action == "negotiate_site_capacity":
                if budget >= NEGOTIATE_COST:
                    budget -= NEGOTIATE_COST
                    # Small chance of capacity increase benefit
                    if self.rng.random() < 0.5:
                        reward = 0.1

            elif action == "adjust_strategy":
                # No direct cost, small benefit from adaptation
                reward = 0.01

            elif action == "stop_recruitment":
                # End the rollout early
                break

            # Apply dropout
            if enrolled > 0 and self.rng.random() < DROPOUT_PROB:
                enrolled = max(0, enrolled - 1)
                reward -= 0.2

            # Discounted reward
            total_reward += reward * (self.discount ** i)

            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "enrolled": enrolled,
                "budget": budget,
            })

        return RolloutResult(
            actions=action_sequence[:len(trajectory)],
            final_enrolled=enrolled,
            final_budget=budget,
            steps_taken=len(trajectory),
            milestones_reached=milestones,
            total_reward=total_reward,
            trajectory=trajectory,
        )

    def generate_action_branches(
        self,
        snapshot: StateSnapshot,
    ) -> Dict[str, List[str]]:
        """Generate action sequences for different strategic branches."""
        branches = {}

        # Branch 1: Aggressive screening
        branches["aggressive_screen"] = ["screen_patient"] * self.lookahead_steps

        # Branch 2: Focus on allocation
        consented = snapshot.patient_memory.get("consented_pending_allocation", 0)
        alloc_actions = ["allocate_to_site"] * min(consented, self.lookahead_steps)
        alloc_actions += ["screen_patient"] * (self.lookahead_steps - len(alloc_actions))
        branches["allocation_focus"] = alloc_actions

        # Branch 3: Recontact heavy
        followup = snapshot.patient_memory.get("followup_due", 0)
        recontact_actions = ["recontact"] * min(followup, self.lookahead_steps)
        recontact_actions += ["screen_patient"] * (self.lookahead_steps - len(recontact_actions))
        branches["recontact_focus"] = recontact_actions

        # Branch 4: Balanced approach
        balanced = []
        for i in range(self.lookahead_steps):
            if i % 3 == 0 and consented > 0:
                balanced.append("allocate_to_site")
            elif i % 3 == 1 and followup > 0:
                balanced.append("recontact")
            else:
                balanced.append("screen_patient")
        branches["balanced"] = balanced

        # Branch 5: Early stop if near target
        if snapshot.enrolled >= snapshot.target * 0.9:
            branches["early_stop"] = ["allocate_to_site"] * 3 + ["stop_recruitment"]

        # Branch 6: Site negotiation if constrained
        if snapshot.active_constraints.get("site_bottleneck", False):
            negotiate_actions = ["negotiate_site_capacity"] + ["screen_patient"] * (self.lookahead_steps - 1)
            branches["negotiate_first"] = negotiate_actions

        return branches

    def run_counterfactual_analysis(
        self,
        env: Any,
        actual_actions: List[str],
    ) -> CounterfactualAnalysis:
        """Run full counterfactual analysis comparing actual to alternative branches."""
        snapshot = self.snapshot_from_env(env)

        # Rollout actual trajectory
        actual_results = []
        for _ in range(self.num_rollouts_per_branch):
            result = self.rollout_from_snapshot(snapshot, actual_actions[:self.lookahead_steps])
            actual_results.append(result)
        
        # Average actual outcome
        actual_outcome = RolloutResult(
            actions=actual_actions[:self.lookahead_steps],
            final_enrolled=int(np.mean([r.final_enrolled for r in actual_results])),
            final_budget=float(np.mean([r.final_budget for r in actual_results])),
            steps_taken=int(np.mean([r.steps_taken for r in actual_results])),
            milestones_reached=actual_results[0].milestones_reached,
            total_reward=float(np.mean([r.total_reward for r in actual_results])),
            trajectory=actual_results[0].trajectory,
        )

        # Generate and rollout alternative branches
        branches = self.generate_action_branches(snapshot)
        counterfactual_outcomes = []

        for branch_name, branch_actions in branches.items():
            branch_results = []
            for _ in range(self.num_rollouts_per_branch):
                result = self.rollout_from_snapshot(snapshot, branch_actions)
                branch_results.append(result)

            # Average branch outcome
            avg_result = RolloutResult(
                actions=branch_actions,
                final_enrolled=int(np.mean([r.final_enrolled for r in branch_results])),
                final_budget=float(np.mean([r.final_budget for r in branch_results])),
                steps_taken=int(np.mean([r.steps_taken for r in branch_results])),
                milestones_reached=branch_results[0].milestones_reached,
                total_reward=float(np.mean([r.total_reward for r in branch_results])),
                trajectory=branch_results[0].trajectory,
            )
            counterfactual_outcomes.append((branch_name, avg_result))

        # Find best counterfactual
        best_cf_name, best_cf = max(counterfactual_outcomes, key=lambda x: x[1].total_reward)
        
        # Calculate regret and opportunity cost
        regret = max(0.0, best_cf.total_reward - actual_outcome.total_reward)
        opportunity_cost = best_cf.total_reward if regret > 0 else 0.0

        # Find critical step (where trajectories diverge most)
        critical_step = 0
        if actual_outcome.trajectory and best_cf.trajectory:
            max_diff = 0.0
            for i, (actual_t, cf_t) in enumerate(zip(
                actual_outcome.trajectory, best_cf.trajectory
            )):
                diff = abs(actual_t.get("reward", 0) - cf_t.get("reward", 0))
                if diff > max_diff:
                    max_diff = diff
                    critical_step = i

        # Recommend action
        if regret > 0.1:
            recommended_action = best_cf.actions[0] if best_cf.actions else "screen_patient"
        else:
            recommended_action = actual_actions[0] if actual_actions else "screen_patient"

        return CounterfactualAnalysis(
            actual_outcome=actual_outcome,
            counterfactual_outcomes=counterfactual_outcomes,
            regret=regret,
            opportunity_cost=opportunity_cost,
            critical_step=critical_step,
            recommended_action=recommended_action,
        )

    def quick_counterfactual(
        self,
        env: Any,
        proposed_action: str,
    ) -> Dict[str, Any]:
        """Quick counterfactual check for a single proposed action.
        
        Returns recommendation on whether to proceed with proposed action.
        """
        snapshot = self.snapshot_from_env(env)

        # Compare proposed action vs alternatives
        alternatives = ["screen_patient", "allocate_to_site", "recontact", "adjust_strategy"]
        if proposed_action in alternatives:
            alternatives.remove(proposed_action)

        proposed_sequence = [proposed_action] + ["screen_patient"] * (self.lookahead_steps - 1)
        proposed_result = self.rollout_from_snapshot(snapshot, proposed_sequence)

        best_alt_name = proposed_action
        best_alt_reward = proposed_result.total_reward

        for alt_action in alternatives:
            alt_sequence = [alt_action] + ["screen_patient"] * (self.lookahead_steps - 1)
            alt_result = self.rollout_from_snapshot(snapshot, alt_sequence)
            if alt_result.total_reward > best_alt_reward:
                best_alt_reward = alt_result.total_reward
                best_alt_name = alt_action

        is_optimal = best_alt_name == proposed_action
        regret = max(0.0, best_alt_reward - proposed_result.total_reward)

        return {
            "proposed_action": proposed_action,
            "proposed_reward": round(proposed_result.total_reward, 3),
            "is_optimal": is_optimal,
            "recommended_action": best_alt_name,
            "recommended_reward": round(best_alt_reward, 3),
            "regret": round(regret, 3),
            "confidence": "high" if regret < 0.05 else "medium" if regret < 0.2 else "low",
        }

    def get_pareto_optimal_branches(
        self,
        env: Any,
    ) -> List[Dict[str, Any]]:
        """Find Pareto-optimal action branches balancing enrollment and budget."""
        snapshot = self.snapshot_from_env(env)
        branches = self.generate_action_branches(snapshot)

        results = []
        for branch_name, branch_actions in branches.items():
            rollout_results = []
            for _ in range(self.num_rollouts_per_branch):
                result = self.rollout_from_snapshot(snapshot, branch_actions)
                rollout_results.append(result)

            avg_enrolled = np.mean([r.final_enrolled for r in rollout_results])
            avg_budget = np.mean([r.final_budget for r in rollout_results])
            avg_reward = np.mean([r.total_reward for r in rollout_results])

            results.append({
                "branch": branch_name,
                "enrolled": int(avg_enrolled),
                "budget_remaining": round(avg_budget, 2),
                "total_reward": round(avg_reward, 3),
                "actions": branch_actions[:3],  # First 3 actions
            })

        # Find Pareto front (maximize enrolled, maximize budget)
        pareto_front = []
        for r in results:
            dominated = False
            for other in results:
                if (other["enrolled"] > r["enrolled"] and 
                    other["budget_remaining"] >= r["budget_remaining"]):
                    dominated = True
                    break
                if (other["enrolled"] >= r["enrolled"] and 
                    other["budget_remaining"] > r["budget_remaining"]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(r)

        return sorted(pareto_front, key=lambda x: -x["total_reward"])
