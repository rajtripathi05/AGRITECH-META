"""
MEDIUM TASK: 3-step soil + yield balance.
Score in [0.0, 1.0]. Deterministic. No randomness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import AgriEnv
from models import Action


def run_medium_task(actions, scenario="default"):
    if len(actions) != 3:
        raise ValueError("Medium task requires exactly 3 actions.")
    env = AgriEnv(scenario=scenario)
    obs0 = env.reset()

    soil_trace  = [obs0.soil_quality]
    yield_accum = 0.0
    penalties   = []

    for action in actions:
        obs, reward, done, info = env.step(action)
        soil_trace.append(info["soil_health"])
        yield_accum += info["yield_score"]
        penalties.append(sum(info["penalties"].values()))
        if done:
            break

    return _score_medium(soil_trace, yield_accum, penalties)


def _score_medium(soil_trace, yield_accum, penalties):
    steps = len(soil_trace) - 1
    if steps == 0:
        return 0.0
    trend = sum(
        1.0 if soil_trace[i] >= soil_trace[i-1]
        else 0.5 if soil_trace[i] >= soil_trace[i-1] - 0.04
        else 0.0
        for i in range(1, len(soil_trace))
    ) / steps
    avg_yield   = min(1.0, yield_accum / steps)
    avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    score = 0.40 * trend + 0.45 * avg_yield - 0.15 * avg_penalty
    return round(max(0.0, min(1.0, score)), 4)


def grade_medium():
    """Self-contained grader — callable with no args by the OpenEnv validator."""
    actions = [
        Action(crop="wheat", fertilizer=0.3, irrigation=0.5),
        Action(crop="rice",  fertilizer=0.4, irrigation=0.6),
        Action(crop="wheat", fertilizer=0.2, irrigation=0.4),
    ]
    return run_medium_task(actions, scenario="default")


def grade_medium_from_state(agri_state):
    """Score an actual trajectory stored in AgriState for the medium task."""
    if not agri_state.reward_history:
        return 0.0
    return _score_medium(
        agri_state.soil_trace,
        sum(agri_state.reward_history),
        agri_state.penalty_history,
    )


if __name__ == "__main__":
    actions = [
        Action(crop="wheat", fertilizer=0.3, irrigation=0.5),
        Action(crop="rice",  fertilizer=0.4, irrigation=0.6),
        Action(crop="wheat", fertilizer=0.2, irrigation=0.4),
    ]
    print(f"Medium score: {run_medium_task(actions)}")
