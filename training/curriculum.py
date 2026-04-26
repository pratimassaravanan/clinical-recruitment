"""Curriculum-guided progressive difficulty training.

Features:
1. Confidence-aware curriculum scheduling
2. Thompson-sampling for task selection
3. Progressive difficulty management
4. Early mistake recovery curriculum
5. Adaptive difficulty adjustment
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from load_traces import PUBLIC_TASKS, PROGRESSIVE_HORIZONS, make_stage_task_id


def confidence_curriculum_schedule(base_task: str) -> List[str]:
    """Generate staged task schedule for progressive training."""
    return [make_stage_task_id(base_task, horizon) for horizon in PROGRESSIVE_HORIZONS]


@dataclass
class ThompsonCurriculum:
    """Thompson sampling curriculum for multi-armed bandit task selection."""
    
    seed: int = 0
    priors: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.priors:
            for base_task in PUBLIC_TASKS:
                for horizon in PROGRESSIVE_HORIZONS:
                    self.priors[make_stage_task_id(base_task, horizon)] = [1.0, 1.0]

    def sample_task(self) -> str:
        scored = []
        for task_id, (alpha, beta) in self.priors.items():
            score = self.rng.betavariate(alpha, beta)
            scored.append((score, task_id))
        scored.sort(reverse=True)
        return scored[0][1]

    def update(self, task_id: str, success_score: float) -> None:
        alpha, beta = self.priors.get(task_id, [1.0, 1.0])
        self.priors[task_id] = [alpha + success_score, beta + (1.0 - success_score)]


@dataclass
class DifficultyLevel:
    """A difficulty level in the curriculum."""
    
    level_id: str
    name: str
    task_ids: List[str]
    min_success_rate: float  # Required success rate to advance
    episodes_required: int  # Minimum episodes before advancement
    
    # Performance tracking
    episodes_completed: int = 0
    total_score: float = 0.0
    successes: int = 0


class ProgressiveDifficultyCurriculum:
    """Curriculum manager with progressive difficulty levels."""
    
    def __init__(
        self,
        seed: int = 42,
        auto_advance: bool = True,
    ):
        self.rng = random.Random(seed)
        self.auto_advance = auto_advance
        self.current_level_idx = 0
        self.levels = self._create_default_levels()
        self.history: List[Dict[str, Any]] = []
    
    def _create_default_levels(self) -> List[DifficultyLevel]:
        """Create default difficulty progression."""
        return [
            DifficultyLevel(
                "beginner", "Beginner (30-day horizon)",
                task_ids=["easy_bench_stage_30", "medium_bench_stage_30"],
                min_success_rate=0.6, episodes_required=5
            ),
            DifficultyLevel(
                "intermediate", "Intermediate (90-day horizon)",
                task_ids=["easy_bench_stage_90", "medium_bench_stage_90"],
                min_success_rate=0.5, episodes_required=10
            ),
            DifficultyLevel(
                "advanced", "Advanced (180-day horizon)",
                task_ids=["easy_bench_stage_180", "medium_bench_stage_180", "hard_bench_stage_180"],
                min_success_rate=0.4, episodes_required=15
            ),
            DifficultyLevel(
                "expert", "Expert (Full benchmarks)",
                task_ids=["easy_bench", "medium_bench", "hard_bench"],
                min_success_rate=0.3, episodes_required=20
            ),
        ]
    
    def current_level(self) -> DifficultyLevel:
        return self.levels[self.current_level_idx]
    
    def sample_task(self) -> str:
        """Sample a task from the current difficulty level."""
        level = self.current_level()
        return self.rng.choice(level.task_ids)
    
    def record_episode(
        self,
        task_id: str,
        score: float,
        success_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Record episode result and check for level advancement."""
        level = self.current_level()
        level.episodes_completed += 1
        level.total_score += score
        
        success = score >= success_threshold
        if success:
            level.successes += 1
        
        result = {
            "level": level.level_id,
            "task_id": task_id,
            "score": score,
            "success": success,
            "episodes_at_level": level.episodes_completed,
            "success_rate": level.successes / max(1, level.episodes_completed),
            "advanced": False,
        }
        
        # Check for advancement
        if self.auto_advance and self._should_advance():
            self._advance_level()
            result["advanced"] = True
            result["new_level"] = self.current_level().level_id
        
        self.history.append(result)
        return result
    
    def _should_advance(self) -> bool:
        """Check if agent should advance to next level."""
        level = self.current_level()
        
        if level.episodes_completed < level.episodes_required:
            return False
        
        success_rate = level.successes / max(1, level.episodes_completed)
        return success_rate >= level.min_success_rate
    
    def _advance_level(self) -> bool:
        """Advance to next difficulty level."""
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            return True
        return False
    
    def reset_level(self) -> None:
        """Reset current level statistics."""
        level = self.current_level()
        level.episodes_completed = 0
        level.total_score = 0.0
        level.successes = 0
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum state."""
        level = self.current_level()
        return {
            "current_level": level.level_id,
            "current_level_name": level.name,
            "level_index": self.current_level_idx,
            "total_levels": len(self.levels),
            "episodes_completed": level.episodes_completed,
            "episodes_required": level.episodes_required,
            "success_rate": level.successes / max(1, level.episodes_completed),
            "min_success_rate": level.min_success_rate,
            "available_tasks": level.task_ids,
        }


@dataclass
class RecoveryScenario:
    """A scenario for early mistake recovery training."""
    
    scenario_id: str
    name: str
    description: str
    initial_state: Dict[str, Any]  # State modifications to apply
    target_recovery_metric: str
    recovery_threshold: float
    max_steps_to_recover: int
    reward_on_recovery: float = 0.3


class EarlyMistakeRecoveryCurriculum:
    """Curriculum that teaches recovery from early mistakes."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.scenarios = self._create_recovery_scenarios()
        self.current_scenario_idx = 0
        self.recovery_history: List[Dict[str, Any]] = []
    
    def _create_recovery_scenarios(self) -> List[RecoveryScenario]:
        """Create recovery training scenarios."""
        return [
            RecoveryScenario(
                "budget_overrun", "Budget Overrun Recovery",
                "Recover from spending 50% of budget with only 20% enrollment",
                initial_state={"budget_ratio": 0.5, "enrollment_progress": 0.2},
                target_recovery_metric="enrollment_progress",
                recovery_threshold=0.5,
                max_steps_to_recover=30,
            ),
            RecoveryScenario(
                "high_dropout", "High Dropout Recovery",
                "Recover from 30% dropout rate",
                initial_state={"dropout_rate": 0.3, "retention_rate": 0.7},
                target_recovery_metric="retention_rate",
                recovery_threshold=0.85,
                max_steps_to_recover=20,
            ),
            RecoveryScenario(
                "site_bottleneck", "Site Bottleneck Recovery",
                "Recover from primary site reaching capacity",
                initial_state={"site_bottleneck": True, "available_capacity": 5},
                target_recovery_metric="available_capacity",
                recovery_threshold=20,
                max_steps_to_recover=15,
            ),
            RecoveryScenario(
                "regulatory_hold", "Regulatory Hold Recovery",
                "Recover after a regulatory hold is lifted",
                initial_state={"regulatory_hold_days": 10, "screening_backlog": 30},
                target_recovery_metric="screening_backlog",
                recovery_threshold=10,
                max_steps_to_recover=25,
            ),
            RecoveryScenario(
                "screening_failure", "Screening Failure Recovery",
                "Recover from a batch of screening failures",
                initial_state={"recent_screen_failures": 10, "uncertainty": 0.6},
                target_recovery_metric="uncertainty",
                recovery_threshold=0.3,
                max_steps_to_recover=20,
            ),
        ]
    
    def sample_scenario(self) -> RecoveryScenario:
        """Sample a recovery scenario."""
        return self.rng.choice(self.scenarios)
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[RecoveryScenario]:
        """Get a specific scenario by ID."""
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        return None
    
    def apply_scenario_to_state(
        self,
        env_state: Dict[str, Any],
        scenario: RecoveryScenario,
    ) -> Dict[str, Any]:
        """Apply scenario modifications to environment state."""
        modified = dict(env_state)
        modified.update(scenario.initial_state)
        modified["_recovery_scenario"] = scenario.scenario_id
        modified["_recovery_target"] = scenario.target_recovery_metric
        modified["_recovery_threshold"] = scenario.recovery_threshold
        modified["_max_recovery_steps"] = scenario.max_steps_to_recover
        return modified
    
    def check_recovery(
        self,
        state: Dict[str, Any],
        steps_taken: int,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Check if recovery was successful.
        
        Returns: (recovered, bonus_reward, info)
        """
        scenario_id = state.get("_recovery_scenario")
        if not scenario_id:
            return False, 0.0, {}
        
        scenario = self.get_scenario_by_id(scenario_id)
        if not scenario:
            return False, 0.0, {}
        
        current_value = state.get(scenario.target_recovery_metric, 0.0)
        recovered = current_value >= scenario.recovery_threshold
        within_time = steps_taken <= scenario.max_steps_to_recover
        
        info = {
            "scenario": scenario_id,
            "current_value": current_value,
            "target": scenario.recovery_threshold,
            "steps_taken": steps_taken,
            "max_steps": scenario.max_steps_to_recover,
            "within_time": within_time,
        }
        
        if recovered and within_time:
            bonus = scenario.reward_on_recovery
            self.recovery_history.append({**info, "success": True})
            return True, bonus, info
        elif steps_taken > scenario.max_steps_to_recover:
            self.recovery_history.append({**info, "success": False})
            return False, -0.1, info
        
        return False, 0.0, info
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery training statistics."""
        if not self.recovery_history:
            return {"total_scenarios": 0, "success_rate": 0.0}
        
        successes = sum(1 for r in self.recovery_history if r.get("success"))
        by_scenario = {}
        
        for r in self.recovery_history:
            sid = r.get("scenario", "unknown")
            if sid not in by_scenario:
                by_scenario[sid] = {"attempts": 0, "successes": 0}
            by_scenario[sid]["attempts"] += 1
            if r.get("success"):
                by_scenario[sid]["successes"] += 1
        
        return {
            "total_scenarios": len(self.recovery_history),
            "success_rate": successes / len(self.recovery_history),
            "by_scenario": by_scenario,
        }


class AdaptiveCurriculumManager:
    """Combined curriculum manager with adaptive difficulty."""
    
    def __init__(self, seed: int = 42):
        self.progressive = ProgressiveDifficultyCurriculum(seed=seed)
        self.recovery = EarlyMistakeRecoveryCurriculum(seed=seed + 1)
        self.thompson = ThompsonCurriculum(seed=seed + 2)
        
        self.mode = "progressive"  # progressive, recovery, thompson
        self.consecutive_failures = 0
        self.recovery_mode_threshold = 3  # Enter recovery after N consecutive failures
    
    def sample_task(self) -> Tuple[str, Optional[RecoveryScenario]]:
        """Sample a task based on current mode.
        
        Returns: (task_id, recovery_scenario if applicable)
        """
        if self.mode == "recovery":
            # In recovery mode, use recovery curriculum
            task_id = self.progressive.sample_task()
            scenario = self.recovery.sample_scenario()
            return task_id, scenario
        elif self.mode == "thompson":
            return self.thompson.sample_task(), None
        else:
            return self.progressive.sample_task(), None
    
    def record_result(
        self,
        task_id: str,
        score: float,
        recovered: bool = False,
    ) -> Dict[str, Any]:
        """Record episode result and adapt curriculum."""
        result = {
            "mode": self.mode,
            "task_id": task_id,
            "score": score,
        }
        
        # Track failures for recovery mode entry
        if score < 0.2:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Check if we should enter recovery mode
        if self.consecutive_failures >= self.recovery_mode_threshold:
            if self.mode != "recovery":
                self.mode = "recovery"
                result["mode_change"] = "entered_recovery"
        
        # Exit recovery mode on success
        if self.mode == "recovery" and recovered:
            self.mode = "progressive"
            self.consecutive_failures = 0
            result["mode_change"] = "exited_recovery"
        
        # Record in progressive curriculum
        if self.mode == "progressive":
            prog_result = self.progressive.record_episode(task_id, score)
            result.update(prog_result)
        
        # Update Thompson priors
        self.thompson.update(task_id, score)
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get combined curriculum state."""
        return {
            "mode": self.mode,
            "consecutive_failures": self.consecutive_failures,
            "progressive": self.progressive.get_curriculum_state(),
            "recovery_stats": self.recovery.get_recovery_stats(),
        }
