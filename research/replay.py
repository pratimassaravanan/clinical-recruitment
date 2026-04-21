"""Frontier replay buffer with strict subgoal execution for long-horizon offline analysis.

Features:
1. Frontier replay buffer - prioritized experience replay
2. Strict subgoal executor - enforces subgoal completion before progression
3. Replay-driven training utilities
4. Trajectory segmentation with subgoal boundaries
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Subgoal:
    """A subgoal in the hierarchical task structure."""
    
    subgoal_id: str
    name: str
    target_metric: str  # e.g., "enrollment_progress", "budget_efficiency"
    target_value: float
    achieved: bool = False
    achieved_step: Optional[int] = None
    reward_on_completion: float = 0.1
    
    def check_completion(self, state: Dict[str, Any]) -> bool:
        """Check if subgoal is completed given current state."""
        current_value = state.get(self.target_metric, 0.0)
        if current_value >= self.target_value and not self.achieved:
            self.achieved = True
            return True
        return False


@dataclass
class SubgoalSequence:
    """A sequence of subgoals that must be completed in order."""
    
    subgoals: List[Subgoal]
    current_index: int = 0
    allow_skip: bool = False
    
    def current_subgoal(self) -> Optional[Subgoal]:
        """Get the current active subgoal."""
        if self.current_index < len(self.subgoals):
            return self.subgoals[self.current_index]
        return None
    
    def advance(self) -> bool:
        """Advance to next subgoal. Returns True if sequence complete."""
        self.current_index += 1
        return self.current_index >= len(self.subgoals)
    
    def is_complete(self) -> bool:
        return self.current_index >= len(self.subgoals)
    
    def reset(self) -> None:
        self.current_index = 0
        for sg in self.subgoals:
            sg.achieved = False
            sg.achieved_step = None


class StrictSubgoalExecutor:
    """Executor that enforces strict subgoal completion before progression."""
    
    def __init__(
        self,
        subgoal_sequence: Optional[SubgoalSequence] = None,
        strict_mode: bool = True,
    ):
        self.sequence = subgoal_sequence or self._default_sequence()
        self.strict_mode = strict_mode
        self.execution_history: List[Dict[str, Any]] = []
    
    def _default_sequence(self) -> SubgoalSequence:
        """Create default clinical trial subgoal sequence."""
        return SubgoalSequence(
            subgoals=[
                Subgoal("sg_screen_10", "Screen 10 patients", "screened_count", 10, reward_on_completion=0.05),
                Subgoal("sg_enroll_25pct", "25% enrollment", "enrollment_progress", 0.25, reward_on_completion=0.15),
                Subgoal("sg_enroll_50pct", "50% enrollment", "enrollment_progress", 0.50, reward_on_completion=0.20),
                Subgoal("sg_retention_80", "80% retention", "retention_rate", 0.80, reward_on_completion=0.10),
                Subgoal("sg_enroll_75pct", "75% enrollment", "enrollment_progress", 0.75, reward_on_completion=0.25),
                Subgoal("sg_complete", "100% enrollment", "enrollment_progress", 1.0, reward_on_completion=0.50),
            ]
        )
    
    def step(
        self,
        state: Dict[str, Any],
        action: str,
        step_num: int,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Execute a step with subgoal enforcement.
        
        Returns: (bonus_reward, subgoal_completed, info)
        """
        bonus = 0.0
        completed = False
        info = {"current_subgoal": None, "subgoal_progress": 0.0}
        
        current = self.sequence.current_subgoal()
        if current:
            info["current_subgoal"] = current.name
            
            # Check if current subgoal is completed
            if current.check_completion(state):
                current.achieved_step = step_num
                bonus = current.reward_on_completion
                completed = True
                
                self.execution_history.append({
                    "step": step_num,
                    "subgoal": current.subgoal_id,
                    "action": action,
                    "achieved": True,
                })
                
                # Advance to next subgoal
                self.sequence.advance()
            
            # Calculate progress toward current subgoal
            current_value = state.get(current.target_metric, 0.0)
            info["subgoal_progress"] = min(1.0, current_value / max(0.001, current.target_value))
        
        info["sequence_complete"] = self.sequence.is_complete()
        info["subgoals_completed"] = self.sequence.current_index
        info["total_subgoals"] = len(self.sequence.subgoals)
        
        return bonus, completed, info
    
    def get_valid_actions(
        self,
        state: Dict[str, Any],
        all_actions: List[str],
    ) -> List[str]:
        """Get actions valid for current subgoal in strict mode."""
        if not self.strict_mode:
            return all_actions
        
        current = self.sequence.current_subgoal()
        if not current:
            return all_actions
        
        # Map subgoals to preferred actions
        subgoal_actions = {
            "screened_count": ["screen_patient"],
            "enrollment_progress": ["allocate_to_site", "screen_patient", "recontact"],
            "retention_rate": ["recontact", "adjust_strategy"],
            "budget_efficiency": ["adjust_strategy"],
        }
        
        preferred = subgoal_actions.get(current.target_metric, all_actions)
        return [a for a in all_actions if a in preferred] or all_actions
    
    def reset(self) -> None:
        self.sequence.reset()
        self.execution_history = []


@dataclass
class FrontierReplayBuffer:
    """Prioritized replay buffer that maintains frontier of best experiences."""
    
    capacity: int = 128
    items: List[Dict[str, Any]] = field(default_factory=list)
    priority_keys: List[str] = field(default_factory=lambda: ["milestone_potential", "reward"])
    
    def add(self, transition: Dict[str, Any]) -> None:
        """Add transition to buffer, maintaining priority order."""
        self.items.append(dict(transition))
        self.items.sort(
            key=lambda item: tuple(
                float(item.get(key, 0.0)) for key in self.priority_keys
            ),
            reverse=True,
        )
        if len(self.items) > self.capacity:
            self.items = self.items[:self.capacity]
    
    def add_trajectory(self, trajectory: List[Dict[str, Any]]) -> None:
        """Add entire trajectory with computed returns."""
        gamma = 0.99
        returns = []
        G = 0.0
        
        for t in reversed(trajectory):
            G = t.get("reward", 0.0) + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        
        for i, t in enumerate(trajectory):
            t_copy = dict(t)
            t_copy["return"] = returns[i]
            self.add(t_copy)
    
    def sample(self, limit: int = 16) -> List[Dict[str, Any]]:
        """Sample top-priority experiences."""
        return [dict(item) for item in self.items[:limit]]
    
    def sample_random(self, limit: int = 16, seed: int = None) -> List[Dict[str, Any]]:
        """Sample random experiences with priority weighting."""
        if not self.items:
            return []
        
        rng = random.Random(seed)
        n = min(limit, len(self.items))
        
        # Priority-weighted sampling
        weights = [1.0 / (i + 1) for i in range(len(self.items))]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        indices = []
        for _ in range(n):
            r = rng.random()
            cumsum = 0.0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum and i not in indices:
                    indices.append(i)
                    break
        
        return [dict(self.items[i]) for i in indices if i < len(self.items)]
    
    def get_subgoal_transitions(self, subgoal_id: str) -> List[Dict[str, Any]]:
        """Get transitions associated with a specific subgoal."""
        return [
            dict(item) for item in self.items
            if item.get("subgoal_id") == subgoal_id
        ]
    
    def clear(self) -> None:
        self.items = []
    
    def __len__(self) -> int:
        return len(self.items)


class ReplayDrivenTrainer:
    """Training utilities using replay buffer."""
    
    def __init__(
        self,
        buffer: FrontierReplayBuffer,
        subgoal_executor: Optional[StrictSubgoalExecutor] = None,
    ):
        self.buffer = buffer
        self.executor = subgoal_executor or StrictSubgoalExecutor()
    
    def compute_advantages(
        self,
        transitions: List[Dict[str, Any]],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> List[float]:
        """Compute GAE advantages for transitions."""
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(transitions))):
            t = transitions[i]
            reward = t.get("reward", 0.0)
            value = t.get("value", 0.0)
            next_value = transitions[i + 1].get("value", 0.0) if i + 1 < len(transitions) else 0.0
            done = t.get("done", False)
            
            delta = reward + gamma * next_value * (1 - done) - value
            gae = delta + gamma * lam * (1 - done) * gae
            advantages.append(gae)
        
        return list(reversed(advantages))
    
    def generate_training_batch(
        self,
        batch_size: int = 32,
        include_subgoal_transitions: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate a training batch from the replay buffer."""
        batch = self.buffer.sample_random(batch_size)
        
        if include_subgoal_transitions:
            # Ensure subgoal completion transitions are included
            for sg in self.executor.sequence.subgoals:
                sg_transitions = self.buffer.get_subgoal_transitions(sg.subgoal_id)
                if sg_transitions:
                    batch.extend(sg_transitions[:2])  # Add up to 2 per subgoal
        
        return batch[:batch_size]
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get statistics about the replay buffer for training."""
        if not self.buffer.items:
            return {"buffer_size": 0}
        
        rewards = [item.get("reward", 0.0) for item in self.buffer.items]
        returns = [item.get("return", 0.0) for item in self.buffer.items if "return" in item]
        
        return {
            "buffer_size": len(self.buffer),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "avg_return": sum(returns) / len(returns) if returns else 0.0,
            "subgoals_completed": self.executor.sequence.current_index,
        }
