"""
EASY TASK: Single-step yield maximization.
Score in [0.0, 1.0]. Deterministic. No randomness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import AgriEnv
from models import Action


def run_easy_task(action, scenario="default"):
    env = AgriEnv(scenario=scenario)
    env.reset()
    obs, reward, done, info = env.step(action)
    return _score_easy(info["yield_score"], info["soil_health"], info["penalties"])


def _score_easy(yield_score, soil_quality, penalties):
    total_penalty = sum(penalties.values())
    raw = 0.70 * yield_score + 0.30 * soil_quality - 0.20 * total_penalty
    return round(max(0.0, min(1.0, raw)), 4)


def grade_easy():
    """Self-contained grader — callable with no args by the OpenEnv validator."""
    action = Action(crop="wheat", fertilizer=0.4, irrigation=0.5)
    return run_easy_task(action, scenario="default")


if __name__ == "__main__":
    action = Action(crop="wheat", fertilizer=0.4, irrigation=0.5)
    print(f"Easy score: {run_easy_task(action)}")
