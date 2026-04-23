"""Standalone HTTP client for the Clinical Recruitment environment."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import httpx

DEFAULT_URL = "https://pratimassaravanan-clinical-recruitment.hf.space"


class ClinicalRecruitmentClient:
    """Session-aware client that mirrors the /reset, /step, /state API."""

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base, timeout=timeout)

    # -- core API ----------------------------------------------------------

    def reset(self, task_id: str = "easy_bench") -> Dict[str, Any]:
        """POST /reset?task_id=... and return the initial observation dict."""
        r = self._http.post("/reset", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """POST /step with an Action JSON body; return the step result."""
        r = self._http.post("/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """GET /state for the current session."""
        r = self._http.get("/state")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        """Release the underlying HTTP transport."""
        self._http.close()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "ClinicalRecruitmentClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- convenience -------------------------------------------------------

    def run_episode(
        self,
        task_id: str = "easy_bench",
        policy_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        max_steps: int = 180,
    ) -> list[Dict[str, Any]]:
        """Run a full episode, returning the list of step results.

        *policy_fn(observation) -> action_dict*; if ``None`` a trivial
        ``screen_patient`` action is used for every step.
        """
        obs = self.reset(task_id)
        results: list[Dict[str, Any]] = []
        for _ in range(max_steps):
            if policy_fn is not None:
                action = policy_fn(obs.get("observation", obs))
            else:
                patients = obs.get("observation", obs).get("available_patients", [])
                pid = patients[0]["id"] if patients else None
                action = {"action_type": "screen_patient", "patient_id": pid}
            result = self.step(action)
            results.append(result)
            if result.get("done", False):
                break
            obs = result
        return results


if __name__ == "__main__":
    print(f"Smoke-testing against {DEFAULT_URL} ...")
    with ClinicalRecruitmentClient() as client:
        info = client._http.get("/").json()
        print(f"  server: {info.get('name')} v{info.get('version')}")
        obs = client.reset("easy_bench")
        print(f"  reset ok – day {obs['observation']['timestamp']}, "
              f"budget ${obs['observation']['budget_remaining']:.0f}")
        patients = obs["observation"].get("available_patients", [])
        if patients:
            result = client.step({
                "action_type": "screen_patient",
                "patient_id": patients[0]["id"],
            })
            print(f"  step ok – reward={result.get('reward')}, done={result.get('done')}")
        st = client.state()
        print(f"  state ok – step {st.get('step', '?')}")
    print("All checks passed.")
