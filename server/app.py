"""FastAPI application exposing OpenEnv endpoints for Adaptive Clinical Recruitment."""

import os
import sys
import uuid
from threading import Lock

# Ensure root package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from env import ClinicalRecruitmentEnv
from models import Action, Observation
from openenv_adapter import (
    ClinicalRecruitmentAction,
    ClinicalRecruitmentObservation,
    ClinicalRecruitmentOpenEnv,
)

app = FastAPI(
    title="Adaptive Clinical Trial Recruitment Environment",
    description="Long-horizon sequential decision environment modeling clinical trial recruitment funnel",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASKS = ["easy_bench", "medium_bench", "hard_bench"]
ENABLE_WEB_INTERFACE = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() == "true"
web_interface_enabled = False
web_interface_error = None
SESSION_COOKIE = "acr_session_id"
_sessions: dict[str, ClinicalRecruitmentEnv] = {}
_session_lock = Lock()


def _create_session() -> tuple[str, ClinicalRecruitmentEnv]:
    session_id = uuid.uuid4().hex
    env = ClinicalRecruitmentEnv()
    with _session_lock:
        _sessions[session_id] = env
    return session_id, env


def _get_session_env(request: Request) -> ClinicalRecruitmentEnv:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /reset first.",
        )
    with _session_lock:
        env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Session expired or missing. Call /reset first.",
        )
    return env


def _drop_session(session_id: str | None) -> None:
    if not session_id:
        return
    with _session_lock:
        _sessions.pop(session_id, None)


@app.get("/")
async def root():
    return {
        "name": "adaptive-clinical-recruitment",
        "version": "1.0.0",
        "status": "running",
        "tasks": TASKS,
        "web_interface_enabled": web_interface_enabled,
        "web_interface_path": "/web" if web_interface_enabled else None,
        "web_interface_error": web_interface_error,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: Request, response: Response, task_id: str = Query(default="easy_bench")):
    session_id = None
    try:
        existing_session_id = request.cookies.get(SESSION_COOKIE)
        if existing_session_id:
            _drop_session(existing_session_id)

        session_id, env = _create_session()
        result = env.reset(task=task_id)
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
        return result.model_dump()
    except ValueError as e:
        _drop_session(session_id)
        response.delete_cookie(SESSION_COOKIE)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(action: Action, request: Request, response: Response):
    try:
        session_id = request.cookies.get(SESSION_COOKIE)
        env = _get_session_env(request)
        result = env.step(action)
        if result.done:
            _drop_session(session_id)
            response.delete_cookie(SESSION_COOKIE)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state(request: Request):
    try:
        env = _get_session_env(request)
        return env.state().model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    return {
        "easy_bench": {
            "name": "Basic Eligibility Screening",
            "description": "Stable patient pool, low dropout, generous budget/time. Goal: reach 80% enrollment with high screening accuracy.",
            "difficulty": "easy",
            "max_steps": 180,
        },
        "medium_bench": {
            "name": "Full Funnel with Site Allocation",
            "description": "Moderate uncertainty, 3 sites with different performance, some dropout risk. Optimize across sites.",
            "difficulty": "medium",
            "max_steps": 180,
        },
        "hard_bench": {
            "name": "Multi-Objective Pipeline Under Pressure",
            "description": "Tight budget/time, high dropout, non-stationary patient quality, curriculum injections. Multi-objective optimization.",
            "difficulty": "hard",
            "max_steps": 180,
        },
    }


def _attach_openenv_web_routes() -> tuple:
    try:
        from openenv.core.env_server import create_web_interface_app

        web_app = create_web_interface_app(
            ClinicalRecruitmentOpenEnv,
            ClinicalRecruitmentAction,
            ClinicalRecruitmentObservation,
            max_concurrent_envs=int(os.environ.get("OPENENV_MAX_CONCURRENT_ENVS", "16")),
        )
        existing_routes = {
            (
                tuple(sorted(getattr(route, "methods", set()) or [])),
                getattr(route, "path", None),
            )
            for route in app.router.routes
        }
        for route in web_app.router.routes:
            key = (
                tuple(sorted(getattr(route, "methods", set()) or [])),
                getattr(route, "path", None),
            )
            if key not in existing_routes:
                app.router.routes.append(route)
        return True, None
    except Exception as exc:
        return False, str(exc)


if ENABLE_WEB_INTERFACE:
    web_interface_enabled, web_interface_error = _attach_openenv_web_routes()


@app.get("/dashboard")
async def dashboard_alias():
    if not web_interface_enabled:
        raise HTTPException(
            status_code=404,
            detail=web_interface_error or "Built-in OpenEnv web interface unavailable.",
        )
    return RedirectResponse(url="/web", status_code=307)


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
