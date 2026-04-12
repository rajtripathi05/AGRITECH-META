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

import sys
import os

# Fix sys.path BEFORE any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import subprocess
import threading
import uuid
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openenv.core.env_server import Environment, create_app
from openenv.core.env_server.types import (
    Action as SdkAction,
    Observation as SdkObservation,
    State as SdkState,
)

from env import AgriEnv
from models import Action, Observation, AgriState

load_dotenv()

# ---------------------------------------------------------------------------
# Task manifest & graders (unchanged)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# SDK-inheriting Environment wrapper
# ---------------------------------------------------------------------------

_DEFAULT_SCENARIO = "default"
_lock = threading.Lock()


class SelectionGradeEnvironment(Environment[Action, Observation, AgriState]):
    """OpenEnv SDK wrapper around AgriEnv with multi-session support."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    SESSIONS: Dict[str, Dict[str, Any]] = {}  # episode_id -> session dict

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_episode_id: Optional[str] = None

    # ---- helpers ----------------------------------------------------------

    @classmethod
    def _get_session(cls, episode_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if episode_id and episode_id in cls.SESSIONS:
            return cls.SESSIONS[episode_id]
        return None

    @classmethod
    def _default_observation(cls) -> Observation:
        """Return an observation using the 'default' scenario initial values."""
        env = AgriEnv(scenario="default", seed=42)
        obs = env.reset()
        return Observation(
            nitrogen=obs.nitrogen,
            moisture=obs.moisture,
            soil_quality=obs.soil_quality,
            last_crop=obs.last_crop,
            season=obs.season,
            weather=obs.weather,
            groundwater=obs.groundwater,
            budget=obs.budget,
            done=False,
            reward=None,
        )

    # ---- SDK abstract methods ---------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> Observation:
        scenario = kwargs.get("scenario", _DEFAULT_SCENARIO)
        seed = seed if seed is not None else 42
        episode_id = episode_id or str(uuid.uuid4())
        task_id = kwargs.get("task_id", "hard")

        with _lock:
            env = AgriEnv(scenario=scenario, seed=seed)
            obs = env.reset()

            agri_state = AgriState(
                episode_id=episode_id,
                step_count=0,
                task_id=task_id,
                reward_history=[],
                action_history=[],
                soil_trace=[obs.soil_quality],
                nitrogen_trace=[obs.nitrogen],
                budget_trace=[obs.budget],
                penalty_history=[],
            )

            SelectionGradeEnvironment.SESSIONS[episode_id] = {
                "env": env,
                "state": agri_state,
            }
            self._current_episode_id = episode_id

        return Observation(
            nitrogen=obs.nitrogen,
            moisture=obs.moisture,
            soil_quality=obs.soil_quality,
            last_crop=obs.last_crop,
            season=obs.season,
            weather=obs.weather,
            groundwater=obs.groundwater,
            budget=obs.budget,
            done=False,
            reward=None,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> Observation:
        episode_id = kwargs.get("episode_id") or self._current_episode_id

        with _lock:
            session = self._get_session(episode_id)
            if session is None:
                # Auto-reset with defaults if no session exists
                self.reset(episode_id=episode_id)
                session = self.SESSIONS[episode_id]

            env: AgriEnv = session["env"]
            agri_state: AgriState = session["state"]

            try:
                obs, reward, done, info = env.step(action)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            # Update trajectory in state
            agri_state.step_count += 1
            agri_state.reward_history.append(reward)
            agri_state.action_history.append(action.model_dump(exclude={"metadata"}))
            agri_state.soil_trace.append(info["soil_health"])
            agri_state.nitrogen_trace.append(obs.nitrogen)
            agri_state.budget_trace.append(info["budget_remaining"])
            agri_state.penalty_history.append(sum(info["penalties"].values()))

            final_score = None
            if done:
                # Score the actual trajectory using the appropriate grader
                final_score = _score_from_state(agri_state)

        return Observation(
            nitrogen=obs.nitrogen,
            moisture=obs.moisture,
            soil_quality=obs.soil_quality,
            last_crop=obs.last_crop,
            season=obs.season,
            weather=obs.weather,
            groundwater=obs.groundwater,
            budget=obs.budget,
            done=done,
            reward=final_score if final_score is not None else round(float(reward), 4),
        )

    @property
    def state(self) -> AgriState:
        session = self._get_session(self._current_episode_id)
        if session is not None:
            return session["state"]
        # No active session — return a default state
        return AgriState(episode_id=None, step_count=0)


# ---------------------------------------------------------------------------
# Trajectory-based scoring (delegates to task graders)
# ---------------------------------------------------------------------------

def _score_from_state(agri_state: AgriState) -> float:
    """Score an episode's actual trajectory using the appropriate task grader."""
    task_id = agri_state.task_id

    if task_id == "easy":
        from tasks.easy import grade_easy_from_state
        return grade_easy_from_state(agri_state)
    elif task_id == "medium":
        from tasks.medium import grade_medium_from_state
        return grade_medium_from_state(agri_state)
    else:
        from tasks.hard import grade_hard_from_state
        return grade_hard_from_state(agri_state)


# ---------------------------------------------------------------------------
# Build app via SDK create_app(), then mount additional endpoints
# ---------------------------------------------------------------------------

app = create_app(
    SelectionGradeEnvironment,
    Action,
    Observation,
    env_name="AgriDecisionEnv-v3",
)


# ---------------------------------------------------------------------------
# Additional endpoints (tasks, grading, validation, inference)
# ---------------------------------------------------------------------------

def _run_task_grader(task_id: str) -> dict:
    grader_ref = _TASK_GRADERS.get(task_id)
    if grader_ref is None:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")

    module_name, fn_name = grader_ref
    module = importlib.import_module(module_name)
    score = round(float(getattr(module, fn_name)()), 4)
    return {"task": task_id, "score": score, "reward": score}


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
