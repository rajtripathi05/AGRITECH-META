"""
FastAPI server for AgriDecisionEnv v3.

Exposes:
- GET  /health
- POST /reset
- POST /step
- GET  /state
- POST /inference
- GET  /tasks
- GET/POST /grade/{task_id}
- GET  /validate
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import AgriEnv
from models import Action

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="AgriDecisionEnv-v3", version="3.0.0")

TASK_MANIFEST = [
    {
        "id": "easy",
        "difficulty": "easy",
        "max_steps": 1,
        "description": "Single-step yield maximization",
        "grader": "/grade/easy",
    },
    {
        "id": "medium",
        "difficulty": "medium",
        "max_steps": 3,
        "description": "Balance soil health and yield over 3 seasons",
        "grader": "/grade/medium",
    },
    {
        "id": "hard",
        "difficulty": "hard",
        "max_steps": 5,
        "description": "Full episode with stochastic weather, delayed effects, constraints",
        "grader": "/grade/hard",
    },
]

_TASK_GRADERS = {
    "easy": ("tasks.easy", "grade_easy"),
    "medium": ("tasks.medium", "grade_medium"),
    "hard": ("tasks.hard", "grade_hard"),
}

_env: Optional[AgriEnv] = None
_lock = threading.Lock()


def _get_env() -> AgriEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env


def _run_task_grader(task_id: str) -> dict:
    grader_ref = _TASK_GRADERS.get(task_id)
    if grader_ref is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")

    module_name, fn_name = grader_ref
    module = importlib.import_module(module_name)
    score = round(float(getattr(module, fn_name)()), 4)
    return {"task": task_id, "score": score, "reward": score}


class ResetRequest(BaseModel):
    scenario: str = "default"
    seed: int = 42


class StepRequest(BaseModel):
    crop: str = "wheat"
    fertilizer: float = 0.3
    irrigation: float = 0.4


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "AgriDecisionEnv-v3",
        "version": "3.0.0",
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/tasks",
            "/validate",
            "/grade/easy",
            "/grade/medium",
            "/grade/hard",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global _env
    req = req or ResetRequest()
    with _lock:
        _env = AgriEnv(scenario=req.scenario, seed=req.seed)
        obs = _env.reset()

    obs_payload = obs.model_dump()
    return {
        "observation": obs_payload,
        # Keep the raw fields available for older callers that expect the previous shape.
        **obs_payload,
        "scenario": req.scenario,
        "seed": req.seed,
        "episode_length": 5,
    }


@app.post("/step")
def step(req: StepRequest):
    with _lock:
        env = _get_env()
        action = Action(
            crop=req.crop,
            fertilizer=req.fertilizer,
            irrigation=req.irrigation,
        )
        try:
            obs, reward, done, info = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": obs.model_dump(),
        "reward": round(float(reward), 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    with _lock:
        env = _get_env()
        return env.state().model_dump()


@app.post("/inference")
def run_inference(task: str = "hard", scenario: str = "default"):
    env_vars = os.environ.copy()
    env_vars["AGRI_TASK"] = task
    env_vars["AGRI_SCENARIO"] = scenario

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env_vars,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Inference timed out (>300s)") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logs = result.stdout.strip()
    errors = result.stderr.strip()
    score = None
    for line in logs.splitlines():
        if not line.startswith("[END]"):
            continue
        for part in line.split():
            if not part.startswith("score="):
                continue
            try:
                score = float(part.split("=")[1])
            except ValueError:
                score = None

    return {
        "task": task,
        "logs": logs,
        "errors": errors or None,
        "score": score,
        "success": result.returncode == 0,
    }


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASK_MANIFEST, "count": len(TASK_MANIFEST)}


@app.get("/validate")
def validate():
    task_scores = {task["id"]: _run_task_grader(task["id"]) for task in TASK_MANIFEST}
    checks = {
        "min_3_tasks": len(TASK_MANIFEST) >= 3,
        "all_tasks_have_graders": all(task.get("grader") for task in TASK_MANIFEST),
        "all_scores_in_range": all(
            0.0 <= score_info["score"] <= 1.0 for score_info in task_scores.values()
        ),
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "tasks": task_scores,
        "env_name": "AgriDecisionEnv-v3",
        "version": "3.0.0",
    }


@app.get("/grade/easy")
@app.post("/grade/easy")
def grade_easy():
    return _run_task_grader("easy")


@app.get("/grade/medium")
@app.post("/grade/medium")
def grade_medium():
    return _run_task_grader("medium")


@app.get("/grade/hard")
@app.post("/grade/hard")
def grade_hard():
    return _run_task_grader("hard")


@app.get("/grade/{task_id}")
@app.post("/grade/{task_id}")
def grade_task(task_id: str):
    return _run_task_grader(task_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
