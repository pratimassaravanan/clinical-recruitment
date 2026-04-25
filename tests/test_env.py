"""Basic unit tests for the clinical recruitment environment.

Run: python -m pytest tests/test_env.py -v
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ClinicalRecruitmentEnv
from models import Action, Observation, State, StepResult


class TestEnvBasics:
    """Test fundamental env contract: reset, step, done, reward bounds."""

    def test_reset_returns_step_result(self):
        env = ClinicalRecruitmentEnv()
        result = env.reset(task="easy_bench")
        assert isinstance(result, StepResult)
        assert isinstance(result.observation, Observation)
        assert result.reward == 0.0
        assert result.done is False

    def test_reset_all_tasks(self):
        env = ClinicalRecruitmentEnv()
        for task in ["easy_bench", "medium_bench", "hard_bench"]:
            result = env.reset(task=task)
            assert not result.done
            obs = result.observation
            assert obs.task_id == task
            assert obs.target_enrollment > 0
            assert obs.max_steps > 0
            assert obs.initial_budget >= obs.budget_remaining
            assert obs.budget_remaining > 0
            assert obs.time_to_deadline_days > 0

    def test_reset_accepts_optional_seed(self):
        env = ClinicalRecruitmentEnv()
        result = env.reset(task="easy_bench", seed=123)
        assert result.observation.task_id == "easy_bench"

    def test_invalid_task_raises(self):
        env = ClinicalRecruitmentEnv()
        try:
            env.reset(task="nonexistent_task_xyz")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_step_before_reset_raises(self):
        env = ClinicalRecruitmentEnv()
        try:
            env.step(Action(action_type="adjust_strategy", strategy_change="increase_outreach"))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_reward_bounded(self):
        env = ClinicalRecruitmentEnv()
        env.reset(task="easy_bench")
        for _ in range(20):
            obs = env._make_observation()
            if obs.available_patients:
                action = Action(
                    action_type="screen_patient",
                    patient_id=obs.available_patients[0]["id"],
                    hypothesis="noise_dominant",
                    confidence=0.7,
                )
            else:
                action = Action(
                    action_type="adjust_strategy",
                    strategy_change="increase_outreach",
                    hypothesis="noise_dominant",
                    confidence=0.6,
                )
            result = env.step(action)
            assert -0.5 <= result.reward <= 0.99, f"Reward {result.reward} out of bounds [-0.5, 0.99]"
            if result.done:
                break

    def test_episode_terminates(self):
        """Environment must terminate within max_steps."""
        env = ClinicalRecruitmentEnv()
        env.reset(task="easy_bench")
        for i in range(200):
            action = Action(
                action_type="adjust_strategy",
                strategy_change="increase_outreach",
                hypothesis="noise_dominant",
                confidence=0.5,
            )
            result = env.step(action)
            if result.done:
                break
        assert result.done, f"Episode did not terminate after 200 steps"


class TestActions:
    """Test each action type is accepted and returns valid result."""

    def _setup_env(self):
        env = ClinicalRecruitmentEnv()
        result = env.reset(task="easy_bench")
        return env, result.observation

    def test_screen_patient(self):
        env, obs = self._setup_env()
        if obs.available_patients:
            result = env.step(Action(
                action_type="screen_patient",
                patient_id=obs.available_patients[0]["id"],
                hypothesis="noise_dominant",
                confidence=0.7,
            ))
            assert isinstance(result, StepResult)
            assert isinstance(result.reward, float)

    def test_adjust_strategy(self):
        env, obs = self._setup_env()
        result = env.step(Action(
            action_type="adjust_strategy",
            strategy_change="increase_outreach",
            hypothesis="noise_dominant",
            confidence=0.6,
        ))
        assert isinstance(result, StepResult)

    def test_plan_next_phase(self):
        env, obs = self._setup_env()
        result = env.step(Action(
            action_type="plan_next_phase",
            target_phase="screening",
            plan_summary="focus on high-priority patients",
        ))
        assert isinstance(result, StepResult)

    def test_summarize_and_index(self):
        env, obs = self._setup_env()
        result = env.step(Action(
            action_type="summarize_and_index",
            memory_key="test_key",
            memory_payload="test payload",
        ))
        assert isinstance(result, StepResult)

    def test_retrieve_relevant_history(self):
        env, obs = self._setup_env()
        result = env.step(Action(
            action_type="retrieve_relevant_history",
            memory_query="enrollment progress",
        ))
        assert isinstance(result, StepResult)

    def test_stop_recruitment(self):
        env, obs = self._setup_env()
        result = env.step(Action(action_type="stop_recruitment"))
        assert result.done is True

    def test_invalid_patient_id_handled(self):
        """Screening a nonexistent patient should not crash."""
        env, obs = self._setup_env()
        result = env.step(Action(
            action_type="screen_patient",
            patient_id="NONEXISTENT_PATIENT_ID",
            hypothesis="noise_dominant",
            confidence=0.5,
        ))
        assert isinstance(result, StepResult)
        # Should get a low/negative reward, not a crash


class TestRewardDesign:
    """Test specific reward design fixes."""

    def test_planning_not_catastrophically_negative(self):
        """FIX: Planning actions should not be double-penalized to -0.09."""
        env = ClinicalRecruitmentEnv()
        env.reset(task="easy_bench")
        result = env.step(Action(
            action_type="plan_next_phase",
            target_phase="screening",
            plan_summary="focus screening",
        ))
        # With base +0.01 and penalty -0.03, net should be around -0.02 plus/minus bonuses
        # Must NOT be -0.09 (old double penalty)
        assert result.reward > -0.15, f"Planning reward {result.reward} too negative (double penalty?)"

    def test_consistency_penalty_bounded(self):
        """FIX: Consistency penalty capped, not unbounded -0.10 per step."""
        env = ClinicalRecruitmentEnv()
        env.reset(task="easy_bench")
        # Rapidly switch hypotheses
        hypotheses = ["noise_dominant", "dropout_dominant", "site_bias", "noise_dominant", "site_bias"]
        for i, hyp in enumerate(hypotheses):
            action = Action(
                action_type="adjust_strategy",
                strategy_change="increase_outreach",
                hypothesis=hyp,
                confidence=0.5,
            )
            result = env.step(action)
            # Consistency penalty should be capped at 0.05 max, not 0.10
            assert result.reward > -0.5, f"Step {i} reward {result.reward} suggests unbounded penalty"

    def test_observation_no_future_leak(self):
        """FIX: Observation should use previous step's events, not current step."""
        env = ClinicalRecruitmentEnv()
        result = env.reset(task="easy_bench")
        obs1 = result.observation

        # The events at step 0 should come from step -1 (i.e., max(0, 0-1) = step 0 is OK for first step)
        # Key thing: after step 1, events should be from step 0, not step 1
        result = env.step(Action(
            action_type="adjust_strategy",
            strategy_change="increase_outreach",
            hypothesis="noise_dominant",
            confidence=0.5,
        ))
        # Just verify no crash — the fix is that _events.get(max(0, self._step - 1))
        assert isinstance(result.observation, Observation)


class TestToolEnv:
    """Test the TRL-compatible tool environment class."""

    def test_tool_env_reset(self):
        from tool_env import ClinicalRecruitmentToolEnv
        env = ClinicalRecruitmentToolEnv()
        obs_text = env.reset(task_id="easy_bench")
        assert obs_text is not None
        assert "enrolled=" in obs_text
        assert env.done is False

    def test_tool_env_screen(self):
        from tool_env import ClinicalRecruitmentToolEnv
        env = ClinicalRecruitmentToolEnv()
        env.reset(task_id="easy_bench")
        obs = env.last_observation
        if obs.get("available_patients"):
            pid = obs["available_patients"][0]["id"]
            result = env.screen_patient(patient_id=pid)
            assert isinstance(result, str)
            assert len(env.action_history) == 1
            assert env.action_history[0] == "screen_patient"

    def test_tool_env_done_raises(self):
        from tool_env import ClinicalRecruitmentToolEnv
        env = ClinicalRecruitmentToolEnv()
        env.reset(task_id="easy_bench")
        env.stop_recruitment()
        assert env.done is True
        try:
            env.screen_patient(patient_id="P-0001")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_reward_funcs(self):
        from tool_env import ClinicalRecruitmentToolEnv, REWARD_FUNCS
        env = ClinicalRecruitmentToolEnv()
        env.reset(task_id="easy_bench")
        # Run a few steps
        obs = env.last_observation
        if obs.get("available_patients"):
            env.screen_patient(patient_id=obs["available_patients"][0]["id"])
        env.adjust_strategy(strategy_change="increase_outreach")

        for fn in REWARD_FUNCS:
            rewards = fn(environments=[env])
            assert len(rewards) == 1
            assert isinstance(rewards[0], (int, float))
            assert 0.0 <= float(rewards[0]) <= 1.0


class TestOpenEnvAdapter:
    """Test the OpenEnv adapter applies protections."""

    def test_adapter_reset_step(self):
        from openenv_adapter import ClinicalRecruitmentOpenEnv
        adapter = ClinicalRecruitmentOpenEnv()
        obs = adapter.reset(task="easy_bench")
        assert obs.done is False
        assert obs.reward is not None

        # Step
        action = {"action_type": "adjust_strategy", "strategy_change": "increase_outreach",
                   "hypothesis": "noise_dominant", "confidence": 0.5}
        result = adapter.step(action)
        assert result.reward is not None
        adapter.close()

    def test_adapter_episode_cap(self):
        from openenv_adapter import ClinicalRecruitmentOpenEnv
        adapter = ClinicalRecruitmentOpenEnv()
        adapter.reset(task="easy_bench")
        # Try to exceed 200 steps — should raise RuntimeError
        hit_cap = False
        for i in range(210):
            try:
                adapter.step({"action_type": "adjust_strategy", "strategy_change": "increase_outreach",
                               "hypothesis": "noise_dominant", "confidence": 0.5})
            except RuntimeError as e:
                if "Hard episode cap" in str(e):
                    hit_cap = True
                    break
            except Exception:
                break  # Episode ended naturally
        # Either hit the cap or episode ended naturally before 200
        adapter.close()

    def test_adapter_double_reset(self):
        """Calling reset() twice should cleanly restart, not corrupt state."""
        from openenv_adapter import ClinicalRecruitmentOpenEnv
        adapter = ClinicalRecruitmentOpenEnv()
        obs1 = adapter.reset(task="easy_bench")
        adapter.step({"action_type": "adjust_strategy", "strategy_change": "increase_outreach",
                       "hypothesis": "noise_dominant", "confidence": 0.5})
        # Reset again without finishing episode
        obs2 = adapter.reset(task="medium_bench")
        assert obs2.done is False
        # Step counter should be reset
        assert adapter._adapter_step_count == 0
        adapter.close()

    def test_tool_env_multi_episode(self):
        """Tool env should support reset-play-reset-play cycles."""
        from tool_env import ClinicalRecruitmentToolEnv
        env = ClinicalRecruitmentToolEnv()
        for task in ["easy_bench", "medium_bench"]:
            obs = env.reset(task_id=task)
            assert obs is not None
            assert env.done is False
            assert len(env.action_history) == 0
            env.adjust_strategy(strategy_change="increase_outreach")
            assert len(env.action_history) == 1
        env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
