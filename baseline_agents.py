"""
Baseline agents for AgriDecisionEnv v3.
Three policies: random, rule-based (state-adaptive), greedy.
"""
import random as _random
from models import Action, Observation


def random_policy(obs, rng=None):
    """Uniformly random action."""
    r = rng or _random.Random()
    return Action(
        crop       = r.choice(["rice", "wheat", "none"]),
        fertilizer = round(r.uniform(0.0, 1.0), 2),
        irrigation = round(r.uniform(0.0, 1.0), 2),
    )


def rule_based_policy(obs):
    """
    Adaptive heuristic:
    - Avoids monocropping
    - Adjusts inputs to nitrogen/moisture deficit
    - Reduces spend if budget is tight
    - Guards groundwater
    - Skips planting if nitrogen critically low
    """
    budget_factor = min(1.0, max(0.0, float(obs.budget) / 120.0))

    # Crop selection — monocrop avoidance checked BEFORE weather branch
    if float(obs.nitrogen) < 0.25:
        crop = "none"
    elif obs.last_crop == "wheat":
        # Must rotate away from wheat; use rice unless drought forces otherwise
        crop = "none" if obs.weather == "drought" else "rice"
    elif obs.last_crop == "rice":
        crop = "wheat"
    elif obs.weather == "drought":
        crop = "wheat"
    else:
        crop = "wheat"

    # Fertilizer: fill nitrogen deficit, stay under overuse threshold
    n_deficit  = max(0.0, 0.60 - float(obs.nitrogen))
    fertilizer = round(min(0.55, n_deficit * 1.6) * budget_factor, 2)

    # Irrigation: fill moisture deficit, stay under threshold
    m_deficit  = max(0.0, 0.50 - float(obs.moisture))
    irrigation = round(min(0.60, m_deficit * 1.8) * budget_factor, 2)

    # Groundwater guard — tiered caps to prevent depletion cascade
    gw = float(obs.groundwater)
    if gw < 0.15:
        irrigation = min(irrigation, 0.05)   # critical
    elif gw < 0.25:
        irrigation = min(irrigation, 0.12)   # low
    elif gw < 0.40:
        irrigation = min(irrigation, 0.25)   # moderate

    return Action(crop=crop, fertilizer=fertilizer, irrigation=irrigation)


def greedy_policy(obs):
    """
    Always plants rice at high inputs for maximum immediate yield.
    Ignores long-term consequences — useful as single-step upper bound.
    """
    fertilizer = 0.70
    irrigation = 0.70
    if float(obs.budget) < 40.0:
        fertilizer, irrigation = 0.30, 0.30
    return Action(crop="rice", fertilizer=fertilizer, irrigation=irrigation)


def run_episode(policy_fn, scenario="default", seed=42):
    """Run a full 5-step episode and return summary."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import AgriEnv

    env = AgriEnv(scenario=scenario, seed=seed)
    obs = env.reset()
    rewards = []

    for _ in range(5):
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    return {
        "total_reward": round(sum(rewards), 4),
        "avg_reward":   round(sum(rewards) / len(rewards), 4),
        "steps":        len(rewards),
        "final_budget": obs.budget,
        "final_soil":   obs.soil_quality,
        "rewards":      rewards,
    }


if __name__ == "__main__":
    import sys, os, random
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    for name, fn in [
        ("random",     lambda o: random_policy(o, random.Random(0))),
        ("rule_based", rule_based_policy),
        ("greedy",     greedy_policy),
    ]:
        r = run_episode(fn)
        print(f"[{name:10s}] avg={r['avg_reward']:.4f}  soil={r['final_soil']:.4f}  budget={r['final_budget']:.1f}")