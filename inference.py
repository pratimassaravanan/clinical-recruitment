"""Baseline inference script for Adaptive Clinical Trial Recruitment environment."""

import json
import os
from typing import List, Optional

import httpx
from openai import OpenAI

# -- Environment variables (checklist-compliant) --
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "https://kaushikss-clinical-recruitment.hf.space")
BENCHMARK = "adaptive-clinical-recruitment"
LLM_CALL_INTERVAL = 5
TEMPERATURE = 0.0
MAX_TOTAL_REWARD = 180.0
SUCCESS_SCORE_THRESHOLD = 0.5


# -- Structured logging (matches sample format exactly) --
def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(error: Optional[str]) -> str:
    if error is None:
        return "null"
    cleaned = str(error).replace("\r", " ").replace("\n", " ").strip()
    return cleaned or "null"


def _format_action(action: dict) -> str:
    atype = action.get("action_type", "screen_patient")
    pid = action.get("patient_id", "")
    sid = action.get("site_id", "")
    strat = action.get("strategy_change", "")
    hyp = action.get("hypothesis", "")
    parts = [atype]
    if pid:
        parts.append(pid)
    if sid:
        parts.append(sid)
    if strat:
        parts.append(strat)
    if hyp:
        parts.append(f"hyp={hyp}")
    return "/".join(parts)


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_format_bool(done)} error={_format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={_format_bool(success)} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# -- Environment client --
class EnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60)

    def reset(self, task_id: str) -> dict:
        r = self.http.post(f"{self.base}/reset", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.http.post(f"{self.base}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.http.get(f"{self.base}/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.http.close()


# -- Rule-based fallback policy --
def _infer_hypothesis(obs: dict) -> str:
    """Infer a hypothesis about dominant trial dynamics from observation."""
    dropout_7d = obs.get("dropout_rate_7d", 0)
    uncertainty = obs.get("uncertainty_level", 0)
    sites = obs.get("site_performance", {})

    if dropout_7d > 0.3:
        return "dropout_dominant"

    if uncertainty > 0.4:
        return "noise_dominant"

    # Check site variance
    conv_rates = [s.get("conversion_rate", 0.5) for s in sites.values()]
    if len(conv_rates) > 1:
        site_var = max(conv_rates) - min(conv_rates)
        if site_var > 0.2:
            return "site_bias"

    return "confounding"


def _infer_confidence(obs: dict, step: int) -> float:
    """Estimate confidence based on how much data we've seen."""
    # Confidence grows with steps (more data = more certainty)
    base = min(0.9, 0.4 + step * 0.005)
    # Lower confidence if uncertainty is high
    base -= obs.get("uncertainty_level", 0) * 0.2
    return max(0.1, min(0.95, base))


def rule_based_action(obs: dict, step: int = 0) -> dict:
    """Heuristic policy: screen available patients, allocate consented ones, manage budget."""
    enrolled = obs.get("enrolled_so_far", 0)
    target = obs.get("target_enrollment", 100)
    budget = obs.get("budget_remaining", 0)
    time_left = obs.get("time_to_deadline_days", 180)
    funnel = obs.get("current_funnel", {})
    available = obs.get("available_patients", [])
    sites = obs.get("site_performance", {})
    uncertainty = obs.get("uncertainty_level", 0)
    screening_backlog = obs.get("screening_backlog", 0)

    # Infer hypothesis and confidence
    hypothesis = _infer_hypothesis(obs)
    confidence = _infer_confidence(obs, step)

    consented = funnel.get("consented", 0)
    enrolled_count = funnel.get("enrolled", 0)
    pending_consented = consented - enrolled_count - funnel.get("dropped", 0)

    # If we have consented patients not yet enrolled, allocate them first
    if pending_consented > 0 and budget > 1500:
        best_site = None
        best_conv = -1
        for sid, sinfo in sites.items():
            cap = sinfo.get("capacity_remaining", 0)
            conv = sinfo.get("conversion_rate", 0)
            if cap > 0 and conv > best_conv:
                best_conv = conv
                best_site = sid
        if best_site:
            return {
                "action_type": "allocate_to_site",
                "patient_id": None,
                "site_id": best_site,
                "strategy_change": None,
                "hypothesis": hypothesis,
                "confidence": confidence,
            }

    # If uncertainty is high and budget allows, adjust strategy
    if uncertainty > 0.6 and budget > 1000 and time_left > 30:
        return {
            "action_type": "adjust_strategy",
            "patient_id": None,
            "site_id": None,
            "strategy_change": "relax_criteria",
            "hypothesis": hypothesis,
            "confidence": confidence,
        }

    # If behind schedule and budget allows, increase outreach
    expected_progress = 1.0 - (time_left / 180.0)
    actual_progress = enrolled / max(1, target)
    if actual_progress < expected_progress * 0.7 and budget > 1000:
        return {
            "action_type": "adjust_strategy",
            "patient_id": None,
            "site_id": None,
            "strategy_change": "increase_outreach",
            "hypothesis": hypothesis,
            "confidence": confidence,
        }

    # Default: screen next available patient (MUST provide patient_id)
    if available and budget > 900:
        # Pick the patient with highest eligibility score (intelligent selection)
        best_patient = max(available, key=lambda p: p.get("eligibility_score", 0))
        patient_id = best_patient.get("id")
        return {
            "action_type": "screen_patient",
            "patient_id": patient_id,
            "site_id": None,
            "strategy_change": None,
            "hypothesis": hypothesis,
            "confidence": confidence,
        }

    # If near target or low budget, stop with calibrated confidence
    if enrolled >= target or budget < 500:
        return {
            "action_type": "stop_recruitment",
            "patient_id": None,
            "site_id": None,
            "strategy_change": None,
            "hypothesis": hypothesis,
            "confidence": min(0.95, enrolled / max(1, target)),
        }

    # No patients available but budget left: try recontact
    if budget > 500:
        return {
            "action_type": "adjust_strategy",
            "patient_id": None,
            "site_id": None,
            "strategy_change": "increase_outreach",
            "hypothesis": hypothesis,
            "confidence": confidence,
        }

    # Fallback: stop
    return {
        "action_type": "stop_recruitment",
        "patient_id": None,
        "site_id": None,
        "strategy_change": None,
        "hypothesis": hypothesis,
        "confidence": min(0.95, enrolled / max(1, target)),
    }


# -- LLM-based policy --
SYSTEM_PROMPT = """You are an expert Clinical Trial Recruitment Optimizer at a top-5 pharma company.

Your goal is to maximize successful enrollment while minimizing cost, timeline slippage, and dropout.

CURRENT STATE:
- Day: {timestamp} / 180
- Budget left: ${budget_remaining:.0f}
- Enrolled: {enrolled_so_far}/{target_enrollment}
- Funnel: {current_funnel}
- Time to deadline: {time_to_deadline_days} days
- Uncertainty: {uncertainty_level:.2f}
- 7-day dropout rate: {dropout_rate_7d:.2f}
- Screening backlog: {screening_backlog}
- Recent events: {recent_events}
- Causal insight: {causal_insight}
- Last hypothesis accuracy: {hypothesis_accuracy:.2f}

AVAILABLE PATIENTS (up to 5):
{patient_summary}

SITE PERFORMANCE:
{site_summary}

AVAILABLE ACTIONS:
1. screen_patient (patient_id) - run screening ($800-900, may find eligible)
2. recontact (patient_id) - re-engage dropped interest (low cost, uncertain)
3. allocate_to_site (patient_id, site_id) - assign consented patient to site for enrollment
4. adjust_strategy (strategy_change) - one of: increase_outreach, relax_criteria, tighten_criteria, focus_site_A/B/C
5. stop_recruitment - end episode early (only when confident enrollment targets are met or unachievable)

REASONING REQUIREMENTS:
You MUST include a hypothesis about what is driving trial dynamics:
- "dropout_dominant" - dropout is the main challenge
- "noise_dominant" - uncertainty/noise is the main challenge
- "site_bias" - site performance variance is the main challenge
- "confounding" - multiple factors interacting

You MUST include a confidence score (0.0-1.0) for your hypothesis.
IMPORTANT: Be CONSISTENT with your hypothesis. Switching hypotheses too often is penalized.
IMPORTANT: When stopping recruitment, calibrate your confidence to match actual enrollment progress.

Respond with EXACTLY one JSON object:
{{"action_type": "<action>", "patient_id": "<id or null>", "site_id": "<id or null>", "strategy_change": "<change or null>", "hypothesis": "<hypothesis>", "confidence": <float>}}

No markdown, no explanation."""


def llm_action(client: OpenAI, obs: dict, step_num: int) -> dict:
    # Format patient summary
    patients = obs.get("available_patients", [])
    if patients:
        patient_lines = []
        for p in patients[:5]:
            patient_lines.append(
                f"  {p.get('id')}: age={p.get('age')}, elig={p.get('eligibility_score', 0):.2f}, "
                f"dropout_risk={p.get('dropout_risk', 0):.2f}"
            )
        patient_summary = "\n".join(patient_lines)
    else:
        patient_summary = "  (none available)"

    # Format site summary
    sites = obs.get("site_performance", {})
    if sites:
        site_lines = []
        for sid, sinfo in sites.items():
            site_lines.append(
                f"  {sid}: conv={sinfo.get('conversion_rate', 0):.2f}, "
                f"wait={sinfo.get('avg_wait_days', 0):.1f}d, "
                f"capacity={sinfo.get('capacity_remaining', 0)}"
            )
        site_summary = "\n".join(site_lines)
    else:
        site_summary = "  (no sites)"

    prompt = SYSTEM_PROMPT.format(
        timestamp=obs.get("timestamp", 0),
        budget_remaining=obs.get("budget_remaining", 0),
        enrolled_so_far=obs.get("enrolled_so_far", 0),
        target_enrollment=obs.get("target_enrollment", 100),
        current_funnel=obs.get("current_funnel", {}),
        time_to_deadline_days=obs.get("time_to_deadline_days", 180),
        uncertainty_level=obs.get("uncertainty_level", 0),
        dropout_rate_7d=obs.get("dropout_rate_7d", 0),
        screening_backlog=obs.get("screening_backlog", 0),
        recent_events=obs.get("recent_events", []),
        causal_insight=obs.get("causal_insight", "No insight yet."),
        hypothesis_accuracy=obs.get("hypothesis_accuracy", 0),
        patient_summary=patient_summary,
        site_summary=site_summary,
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=TEMPERATURE,
        )
        text = resp.choices[0].message.content.strip()
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
        action = json.loads(text)

        valid_actions = (
            "screen_patient",
            "recontact",
            "allocate_to_site",
            "adjust_strategy",
            "stop_recruitment",
        )
        if action.get("action_type") not in valid_actions:
            return rule_based_action(obs, step_num)

        # Normalize None values
        for key in ("patient_id", "site_id", "strategy_change"):
            if action.get(key) in (None, "null", "None", ""):
                action[key] = None

        # Ensure hypothesis and confidence are present
        valid_hypotheses = (
            "dropout_dominant",
            "noise_dominant",
            "site_bias",
            "confounding",
            "unknown",
        )
        if action.get("hypothesis") not in valid_hypotheses:
            action["hypothesis"] = _infer_hypothesis(obs)
        conf = action.get("confidence", 0.5)
        if isinstance(conf, (int, float)):
            action["confidence"] = max(0.0, min(1.0, float(conf)))
        else:
            action["confidence"] = _infer_confidence(obs, step_num)

        return action
    except Exception:
        return rule_based_action(obs, step_num)


# -- Run one task --
def run_task(task_id: str, client: OpenAI) -> float:
    env = EnvClient(ENV_URL)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_info = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id)
        obs = result["observation"]
        last_info = result.get("info", {})

        while not result.get("done", False):
            if steps_taken % LLM_CALL_INTERVAL == 0:
                action = llm_action(client, obs, steps_taken)
            else:
                action = rule_based_action(obs, steps_taken)

            action_str = _format_action(action)

            try:
                result = env.step(action)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0) or 0.0)
                done = bool(result.get("done", False))
                last_info = result.get("info", {})
                error = last_info.get("last_action_error")
                rewards.append(reward)
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error,
                )
            except Exception as exc:
                steps_taken += 1
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

        final_score = last_info.get("final_score")
        if isinstance(final_score, (int, float)):
            score = float(final_score)
        else:
            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(0.999, max(0.001, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        env.close()
        score = min(0.999, max(0.001, score))
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return score


# -- Main --
def main():
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN environment variable is required.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["easy_bench", "medium_bench", "hard_bench"]
    for task_id in tasks:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
