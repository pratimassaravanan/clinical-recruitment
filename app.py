"""FastAPI application exposing OpenEnv endpoints for Adaptive Clinical Recruitment."""

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from env import ClinicalRecruitmentEnv
from models import Action, Observation

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

env = ClinicalRecruitmentEnv()

TASKS = ["easy_bench", "medium_bench", "hard_bench"]
ENABLE_WEB_INTERFACE = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"
web_interface_enabled = False
web_interface_error = None


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
async def reset(task_id: str = Query(default="easy_bench")):
    try:
        result = env.reset(task=task_id)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(action: Action):
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state():
    try:
        return env.state().model_dump()
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

        web_app = create_web_interface_app(ClinicalRecruitmentEnv, Action, Observation)
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
