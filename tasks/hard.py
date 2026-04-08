"""
HARD TASK: 5-step full episode with delayed effects, stochastic weather
(seeded), resource constraints. Score in [0.0, 1.0]. Deterministic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import AgriEnv
from models import Action

EPISODE_LENGTH = 5


def run_hard_task(actions, scenario="default"):
    if len(actions) != EPISODE_LENGTH:
        raise ValueError(f"Hard task requires exactly {EPISODE_LENGTH} actions.")
    env = AgriEnv(scenario=scenario, seed=42)
    obs0 = env.reset()

    nitrogen_trace = [obs0.nitrogen]
    soil_trace     = [obs0.soil_quality]
    reward_list    = []
    budget_trace   = [obs0.budget]
    all_penalties  = []

    for action in actions:
        obs, reward, done, info = env.step(action)
        nitrogen_trace.append(obs.nitrogen)
        soil_trace.append(info["soil_health"])
        reward_list.append(reward)
        budget_trace.append(info["budget_remaining"])
        all_penalties.append(sum(info["penalties"].values()))
        if done:
            break

    return grade_hard(nitrogen_trace, soil_trace, reward_list, budget_trace, all_penalties)


def grade_hard(nitrogen_trace, soil_trace, rewards, budget_trace, penalties):
    if not rewards:
        return 0.0
    steps = len(rewards)

    avg_reward     = sum(rewards) / steps
    degraded       = sum(1 for n in nitrogen_trace if n < 0.25) / (steps + 1)
    drop           = max(0.0, soil_trace[0] - soil_trace[-1])
    sustainability = max(0.0, 1.0 - drop / max(soil_trace[0], 0.01)) if drop > 0 else 1.0
    budget_health  = min(1.0, max(0.0, budget_trace[-1] / 120.0))
    avg_penalty    = sum(penalties) / len(penalties) if penalties else 0.0
    mean_n         = sum(nitrogen_trace) / len(nitrogen_trace)
    variance       = sum((n - mean_n)**2 for n in nitrogen_trace) / len(nitrogen_trace)
    stability      = max(0.0, 1.0 - variance * 8)

    score = (
        0.30 * avg_reward
        - 0.20 * degraded
        + 0.20 * sustainability
        + 0.15 * budget_health
        - 0.10 * avg_penalty
        + 0.05 * stability
    )
    return round(max(0.0, min(1.0, score)), 4)


if __name__ == "__main__":
    actions = [
        Action(crop="wheat", fertilizer=0.3, irrigation=0.5),
        Action(crop="rice",  fertilizer=0.5, irrigation=0.6),
        Action(crop="wheat", fertilizer=0.2, irrigation=0.4),
        Action(crop="none",  fertilizer=0.0, irrigation=0.2),
        Action(crop="wheat", fertilizer=0.3, irrigation=0.4),
    ]
    print(f"Hard score: {run_hard_task(actions)}")
