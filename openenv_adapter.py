"""OpenEnv compatibility adapter for the clinical recruitment benchmark."""

from __future__ import annotations

import collections
import concurrent.futures
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import Field

from env import ClinicalRecruitmentEnv
from models import Action, Observation, State
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

# ---------------------------------------------------------------------------
# Anti-reward-hacking constants
# ---------------------------------------------------------------------------
_MAX_STEPS_PER_SECOND = 10  # rate-limit ceiling
_REPLAY_THRESHOLD = 5  # consecutive identical actions before penalty
_HARD_EPISODE_CAP = 200  # absolute step ceiling regardless of inner env
_DEFAULT_TIMEOUT_S = 30.0  # fallback when caller passes None


class ClinicalRecruitmentAction(Action):
    """OpenEnv action model with optional metadata support."""

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClinicalRecruitmentObservation(Observation):
    """OpenEnv observation model carrying reward/done fields."""

    done: bool = Field(default=False)
    reward: float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClinicalRecruitmentState(State):
    """OpenEnv-compatible state surface for the web UI and ws client."""

    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)


class ClinicalRecruitmentOpenEnv(Environment):
    """Wrap the benchmark env in the OpenEnv Environment interface."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._env = ClinicalRecruitmentEnv()
        self._episode_id: Optional[str] = None

        # --- Anti-reward-hacking bookkeeping ---
        self._step_timestamps: List[float] = []
        self._last_action_dicts: collections.deque = collections.deque(
            maxlen=_REPLAY_THRESHOLD,
        )
        self._adapter_step_count: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ClinicalRecruitmentObservation:
        del seed, kwargs
        selected_task = task or task_id or "easy_bench"
        result = self._env.reset(task=selected_task)
        self._episode_id = episode_id or str(uuid4())

        # Reset anti-reward-hacking state on new episode
        self._step_timestamps.clear()
        self._last_action_dicts.clear()
        self._adapter_step_count = 0

        return self._adapt_observation(result.observation, result.reward, result.done, result.info)

    # ------------------------------------------------------------------
    # step (with timeout, rate-limit, replay detection, episode cap)
    # ------------------------------------------------------------------
    def step(
        self,
        action: ClinicalRecruitmentAction | Dict[str, Any],
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ClinicalRecruitmentObservation:
        del kwargs

        now = time.monotonic()
        effective_timeout = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S

        # --- Rate-limit check ---
        self._step_timestamps.append(now)
        # Keep only timestamps within the last 1-second window
        cutoff = now - 1.0
        self._step_timestamps = [t for t in self._step_timestamps if t >= cutoff]
        if len(self._step_timestamps) > _MAX_STEPS_PER_SECOND:
            raise RuntimeError(
                f"Rate limit exceeded: more than {_MAX_STEPS_PER_SECOND} "
                f"steps/second (likely automation abuse)."
            )

        # --- Episode length hard cap ---
        self._adapter_step_count += 1
        if self._adapter_step_count > _HARD_EPISODE_CAP:
            raise RuntimeError(
                f"Hard episode cap of {_HARD_EPISODE_CAP} steps reached. "
                f"Call reset() to start a new episode."
            )

        # --- Parse / validate action ---
        if isinstance(action, dict):
            action_dict_canonical = dict(sorted(action.items()))
            action = ClinicalRecruitmentAction.model_validate(action)
        else:
            action_dict_canonical = dict(sorted(action.model_dump().items()))

        # --- Action replay detection ---
        replay_penalty = False
        self._last_action_dicts.append(action_dict_canonical)
        if (
            len(self._last_action_dicts) >= _REPLAY_THRESHOLD
            and all(
                d == self._last_action_dicts[0]
                for d in self._last_action_dicts
            )
        ):
            replay_penalty = True

        # --- Execute inner env step with timeout enforcement ---
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._env.step, action)
        try:
            result = future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Inner env step exceeded {effective_timeout}s timeout."
            )
        finally:
            executor.shutdown(wait=False)

        # --- Build metadata with reward breakdown + optional penalty ---
        extra_meta: Dict[str, Any] = {}
        if replay_penalty:
            extra_meta["replay_warning"] = (
                f"Same action submitted {_REPLAY_THRESHOLD}+ times consecutively"
            )

        return self._adapt_observation(
            result.observation,
            result.reward,
            result.done,
            result.info,
            extra_meta=extra_meta,
        )

    # ------------------------------------------------------------------
    # state / metadata / close
    # ------------------------------------------------------------------
    @property
    def state(self) -> ClinicalRecruitmentState:
        state = self._env.state()
        return ClinicalRecruitmentState(
            **state.model_dump(),
            episode_id=self._episode_id,
            step_count=state.step,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="adaptive-clinical-recruitment",
            description=(
                "Long-horizon clinical trial recruitment benchmark with typed actions, "
                "planning, memory, delayed effects, and multi-site constraints."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------
    # observation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _adapt_observation(
        observation: Observation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        *,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> ClinicalRecruitmentObservation:
        metadata: Dict[str, Any] = {"info": info}

        # Expose step-level reward breakdown at the top level of metadata
        # so training code can directly access multi-component signals.
        reward_breakdown = info.get("reward_breakdown")
        if reward_breakdown is not None:
            metadata["reward_breakdown"] = reward_breakdown

        if extra_meta:
            metadata.update(extra_meta)

        return ClinicalRecruitmentObservation(
            **observation.model_dump(),
            reward=reward,
            done=done,
            metadata=metadata,
        )
