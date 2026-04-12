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


def grade_easy_from_state(agri_state):
    """Score an actual trajectory stored in AgriState for the easy task."""
    if not agri_state.reward_history:
        return 0.0
    # Use the first step's data from the trajectory
    # The soil_trace has initial + per-step values; penalty_history has per-step totals
    soil_quality = agri_state.soil_trace[-1] if len(agri_state.soil_trace) > 1 else agri_state.soil_trace[0]
    reward = agri_state.reward_history[0]
    total_penalty = agri_state.penalty_history[0] if agri_state.penalty_history else 0.0
    # Approximate yield_score from reward + penalties (reward = yield + bonus - penalties)
    # Use the same formula: 0.70 * yield + 0.30 * soil - 0.20 * penalty
    # Since we have reward and soil from the actual run, use them directly
    raw = 0.70 * reward + 0.30 * soil_quality - 0.20 * total_penalty
    return round(max(0.0, min(1.0, raw)), 4)


if __name__ == "__main__":
    action = Action(crop="wheat", fertilizer=0.4, irrigation=0.5)
    print(f"Easy score: {run_easy_task(action)}")
