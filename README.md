---
title: Soil Health Intelligence
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
tags:
  - openenv
---

# AgriDecisionEnv v3 – Sustainable Farming RL Environment

## Problem Motivation

Soil degradation affects over 33% of the world's arable land, driven by monocropping, over-fertilization, and unsustainable irrigation. Traditional farming decision-support systems rely on rigid rule engines. This environment provides a testbed for training AI agents to make adaptive, multi-objective farming decisions that balance immediate yield against long-term soil and water sustainability.

This environment evaluates agent performance under delayed rewards, stochastic climate conditions, and resource constraints, closely mimicking real-world agricultural decision-making.

---

## Environment Design

The environment simulates a farm over multiple growing seasons. At each step the agent observes the farm state and chooses a crop, fertilizer level, and irrigation level. Consequences are immediate (yield) and delayed (fertilizer accumulation degrades soil over multiple steps).

### Scenarios

| Scenario  | Nitrogen | Moisture | Groundwater | Budget |
|-----------|----------|----------|-------------|--------|
| default   | 0.50     | 0.50     | 0.70        | 120    |
| fertile   | 0.75     | 0.65     | 0.90        | 150    |
| drought   | 0.45     | 0.20     | 0.35        | 100    |
| degraded  | 0.25     | 0.45     | 0.60        | 80     |

---

## State / Observation

| Field         | Type  | Range        | Description                          |
|---------------|-------|--------------|--------------------------------------|
| nitrogen      | float | [0, 1]       | Soil nitrogen level                  |
| moisture      | float | [0, 1]       | Soil moisture level                  |
| soil_quality  | float | [0, 1]       | Derived: 0.6×N + 0.4×M              |
| last_crop     | str   | rice/wheat/none | Previous crop planted             |
| season        | int   | [0, 10]      | Current step index                   |
| weather       | str   | rainy/normal/drought | Current weather condition  |
| groundwater   | float | [0, 1]       | Available groundwater reserve        |
| budget        | float | unbounded    | Remaining financial budget           |

---

## Action Space

| Field       | Type  | Range  | Description                        |
|-------------|-------|--------|------------------------------------|
| crop        | str   | rice/wheat/none | Crop to plant this season  |
| fertilizer  | float | [0, 1] | Fertilizer application rate        |
| irrigation  | float | [0, 1] | Irrigation application rate        |

---

## Reward Function

```
reward = clamp(
    yield_score
    + soil_quality_bonus        (+0.15 if N≥0.45 and M≥0.35)
    - fertilizer_penalty        (excess above 0.60)
    - delayed_fert_penalty      (rolling 3-step avg above 0.60)
    - irrigation_penalty        (excess above 0.65)
    - monocrop_penalty          (-0.12 if same crop repeated)
    - groundwater_penalty       (when groundwater < 0.20)
    - budget_penalty            (when budget < 0)
, 0.0, 1.0)
```

All reward values are normalized to **[0.0, 1.0]**.

---

## Task Descriptions

### Easy (1 step)
Single-step optimization. Score = weighted sum of yield and soil quality after one action. No weather variation.

### Medium (3 steps)
Multi-season balance. Score rewards a positive soil improvement trend combined with cumulative yield. Monocropping and overuse are penalized.

### Hard (5 steps)
Full episode. Includes stochastic weather (bounded, seeded), delayed fertilizer effects, groundwater depletion, and budget constraints. Score penalizes long-term nitrogen degradation and rewards sustainability (final soil ≥ initial soil).

---

## File Structure

```
agri_v3/
├── models.py           # Pydantic models: Observation, Action, Reward, StepInfo
├── env.py              # AgriEnv class (reset / step / state)
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── baseline_agents.py  # random_policy, rule_based_policy, greedy_policy
├── inference.py        # LLM agent loop with strict OpenEnv log format
├── openenv.yaml        # Environment metadata and config
├── Dockerfile
└── README.md
```

---

## How to Run

### Colab (recommended)

```python
# Cell 1 – install
!pip install pydantic openai

# Cell 2 – add to path
import sys
sys.path.insert(0, '/content/agri_v3')

# Cell 3 – run baseline comparison
from baseline_agents import run_episode, random_policy, rule_based_policy, greedy_policy
import random

for name, fn in [
    ("random",     lambda o: random_policy(o, random.Random(0))),
    ("rule_based", rule_based_policy),
    ("greedy",     greedy_policy),
]:
    r = run_episode(fn)
    print(f"{name}: avg_reward={r['avg_reward']} final_soil={r['final_soil']}")

# Cell 4 – run hard task grader
from tasks.hard import run_hard_task
from models import Action
actions = [
    Action(crop="wheat",  fertilizer=0.3, irrigation=0.5),
    Action(crop="rice",   fertilizer=0.5, irrigation=0.6),
    Action(crop="wheat",  fertilizer=0.2, irrigation=0.4),
    Action(crop="none",   fertilizer=0.0, irrigation=0.2),
    Action(crop="wheat",  fertilizer=0.3, irrigation=0.4),
]
print(f"Hard task score: {run_hard_task(actions)}")
```

### Docker

```bash
docker build -t agri-env .
docker run -e HF_TOKEN=your_token -e AGRI_TASK=hard agri-env
```

### Local (Linux/macOS)

```bash
pip install pydantic openai
HF_TOKEN=your_token AGRI_TASK=hard python inference.py
```

### Local (Windows PowerShell)

```powershell
pip install pydantic openai
$env:HF_TOKEN="your_token"; $env:AGRI_TASK="hard"; python inference.py
```

---

## Baseline Results (default scenario, seed=42)

### LLM Agent (Qwen/Qwen2.5-7B-Instruct via HuggingFace router)

| Task   | Score  | Steps |
|--------|--------|-------|
| easy   | 0.4024 | 1     |
| medium | 0.4214 | 3     |
| hard   | 0.2818 | 5     |

### Rule-based Baselines (hard task, 5 steps)

| Agent      | Avg Reward | Final Soil | Final Budget |
|------------|------------|------------|--------------|
| random     | ~0.21      | ~0.38      | ~35          |
| rule_based | ~0.44      | ~0.51      | ~48          |
| greedy     | ~0.38      | ~0.31      | ~15          |

The rule-based agent consistently outperforms greedy by preserving soil quality, demonstrating that short-term yield maximization is suboptimal in this environment.

---

## OpenEnv Compliance

- `reset()` → `Observation`
- `step(action)` → `(Observation, Reward, done, StepInfo)`
- `state()` → `Observation`
- All rewards in `[0.0, 1.0]`
- Deterministic grading (seeded RNG)
- Typed Pydantic models throughout
- Structured `info` dict returned on every step
