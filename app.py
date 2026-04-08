"""
app.py – FastAPI server for AgriDecisionEnv v3
Exposes OpenEnv-compatible REST API for HuggingFace Spaces validation.

Endpoints:
  GET  /          → health check (200 OK)
  POST /reset     → reset environment, return initial observation
  POST /step      → take action, return (observation, reward, done, info)
  GET  /state     → return current environment state
  POST /inference → run full inference.py episode, return logs
"""
import os, sys, subprocess, threading
from dotenv import load_dotenv

load_dotenv()
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from env import AgriEnv
from models import Action, Observation

app = FastAPI(title="AgriDecisionEnv-v3", version="3.0.0")

# Global env instance (one session at a time — sufficient for validator)
_env: Optional[AgriEnv] = None
_lock = threading.Lock()


def _get_env() -> AgriEnv:
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "AgriDecisionEnv-v3", "version": "3.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── OpenEnv Core API ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    scenario: str = "default"
    seed: int = 42


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    with _lock:
        _env = AgriEnv(scenario=req.scenario, seed=req.seed)
        obs = _env.reset()
    return obs.model_dump()


class StepRequest(BaseModel):
    crop: str = "wheat"
    fertilizer: float = 0.3
    irrigation: float = 0.4


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
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": obs.model_dump(),
        "reward":      round(float(reward), 4),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state():
    with _lock:
        env = _get_env()
        return env.state().model_dump()


# ── Inference endpoint — runs inference.py and returns structured logs ─────────

@app.post("/inference")
def run_inference(task: str = "hard", scenario: str = "default"):
    env_vars = os.environ.copy()
    env_vars["AGRI_TASK"]    = task
    env_vars["AGRI_SCENARIO"] = scenario

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env_vars,
        )
        logs   = result.stdout.strip()
        errors = result.stderr.strip()

        # Parse score from [END] line
        score = None
        for line in logs.splitlines():
            if line.startswith("[END]"):
                for part in line.split():
                    if part.startswith("score="):
                        try:
                            score = float(part.split("=")[1])
                        except ValueError:
                            pass

        return {
            "task":    task,
            "logs":    logs,
            "errors":  errors or None,
            "score":   score,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference timed out (>300s)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Task grader endpoints ─────────────────────────────────────────────────────

@app.post("/grade/easy")
def grade_easy():
    from tasks.easy import run_easy_task
    action = Action(crop="wheat", fertilizer=0.4, irrigation=0.5)
    score  = run_easy_task(action)
    return {"task": "easy", "score": score}


@app.post("/grade/medium")
def grade_medium():
    from tasks.medium import run_medium_task
    actions = [
        Action(crop="wheat", fertilizer=0.3, irrigation=0.5),
        Action(crop="rice",  fertilizer=0.4, irrigation=0.6),
        Action(crop="wheat", fertilizer=0.2, irrigation=0.4),
    ]
    score = run_medium_task(actions)
    return {"task": "medium", "score": score}


@app.post("/grade/hard")
def grade_hard():
    from tasks.hard import run_hard_task
    actions = [
        Action(crop="wheat", fertilizer=0.3, irrigation=0.5),
        Action(crop="rice",  fertilizer=0.5, irrigation=0.6),
        Action(crop="wheat", fertilizer=0.2, irrigation=0.4),
        Action(crop="none",  fertilizer=0.0, irrigation=0.2),
        Action(crop="wheat", fertilizer=0.3, irrigation=0.4),
    ]
    score = run_hard_task(actions)
    return {"task": "hard", "score": score}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
