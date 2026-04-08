---
title: AgriDecisionEnv v3
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
tags:
- openenv
---

# 🌱 AgriDecisionEnv v3 – Sustainable Farming RL Environment

## 🚀 Overview

AgriDecisionEnv v3 is a real-world OpenEnv-compatible environment designed to evaluate AI agents on multi-season agricultural decision-making under uncertainty, delayed effects, and resource constraints.

The environment models how farmers must balance:
- 🌾 Crop yield (short-term productivity)
- 🌍 Soil health (long-term sustainability)
- 💧 Water usage (groundwater limits)
- 💰 Budget constraints (economic feasibility)

This environment evaluates agent performance under delayed rewards, stochastic climate conditions, and resource constraints, closely mimicking real-world agricultural decision-making.

---

## 🌍 Real-World Motivation

Over 33% of global arable land is degraded due to:
- monocropping  
- excessive fertilizer use  
- unsustainable irrigation  

Existing systems rely on static heuristics and fail under dynamic conditions.

👉 This project provides a benchmark environment for training and evaluating intelligent agents capable of adaptive, long-term agricultural decision-making.

---

## 🧠 Environment Design

The environment simulates a farm across multiple seasons.  
At each timestep, the agent observes the current state and selects actions.

### 🔁 Interaction Loop

State → Agent → Action → Environment → New State + Reward

---

## 📊 Scenario Initialization

| Scenario  | Nitrogen | Moisture | Groundwater | Budget |
|-----------|----------|----------|-------------|--------|
| default   | 0.50     | 0.50     | 0.70        | 120    |
| fertile   | 0.75     | 0.65     | 0.90        | 150    |
| drought   | 0.45     | 0.20     | 0.35        | 100    |
| degraded  | 0.25     | 0.45     | 0.60        | 80     |

---

## 🔍 Observation Space

| Field         | Type  | Range        | Description                          |
|---------------|-------|--------------|--------------------------------------|
| nitrogen      | float | [0, 1]       | Soil nitrogen level                  |
| moisture      | float | [0, 1]       | Soil moisture level                  |
| soil_quality  | float | [0, 1]       | Derived soil health metric           |
| last_crop     | str   | rice/wheat/none | Previous crop                    |
| season        | int   | [0, 10]      | Current timestep                     |
| weather       | str   | rainy/normal/drought | Climate condition         |
| groundwater   | float | [0, 1]       | Available groundwater                |
| budget        | float | [0, 150]     | Remaining resources                  |

---

## 🎯 Action Space

| Field       | Type  | Range  | Description                        |
|-------------|-------|--------|------------------------------------|
| crop        | str   | rice/wheat/none | Crop selection            |
| fertilizer  | float | [0, 1] | Fertilizer level                   |
| irrigation  | float | [0, 1] | Irrigation level                   |

---

## ⚙️ Reward Function

reward = clamp(
    yield_score
    + soil_quality_bonus
    - fertilizer_penalty
    - delayed_fertilizer_penalty
    - irrigation_penalty
    - monocrop_penalty
    - groundwater_penalty
    - budget_penalty
, 0.0, 1.0)

### Key Properties

- Dense reward signal (not sparse)
- Penalizes unsustainable practices
- Encourages long-term planning
- Strictly normalized to [0.0, 1.0]

---

## 🧪 Task Design

### 🟢 Easy (1 step)
- Objective: maximize immediate yield  
- Deterministic  
- No delayed effects  

### 🟡 Medium (3 steps)
- Objective: balance yield and soil improvement  
- Introduces temporal dependencies  
- Penalizes overuse  

### 🔴 Hard (5 steps)
- Includes:
  - stochastic weather  
  - delayed fertilizer effects  
  - groundwater depletion  
  - budget constraints  

👉 Tests long-term sustainability strategies

---

## 📈 Baseline Results

### LLM Agent (via OpenAI client + HF router)

| Task   | Score  |
|--------|--------|
| Easy   | 0.47   |
| Medium | 0.36   |
| Hard   | 0.33   |

### Rule-Based Baselines

| Agent      | Avg Reward | Final Soil | Final Budget |
|------------|------------|------------|--------------|
| Random     | 0.35       | 0.88       | -58.0        |
| Greedy     | 0.35       | 0.54       | -47.5        |
| Rule-Based | 0.18       | 0.26       | 2.5          |

👉 Rule-based agent conserves budget but trades off yield; greedy/random overspend

---

## 🤖 Inference (OpenAI Client)

The agent uses the OpenAI client interface for all LLM calls.

Required environment variables:
- API_BASE_URL  
- MODEL_NAME  
- HF_TOKEN  

### Logging Format (STRICT)

[START]  
[STEP]  
[END]  

✔ Required for evaluation  
✔ Fully reproducible  

---

## 🧱 Project Structure

agri_v3/
├── models.py  
├── env.py  
├── tasks/  
├── baseline_agents.py  
├── inference.py  
├── openenv.yaml  
├── Dockerfile  
└── README.md  

---

## ⚙️ Setup & Usage

### Local
pip install pydantic openai  
HF_TOKEN=your_token python inference.py  

### Docker
docker build -t agri-env .  
docker run -e HF_TOKEN=your_token agri-env  

### Hugging Face Spaces
- SDK: Docker  
- Tag: openenv  
- Must respond to /reset  

---

## ✅ OpenEnv Compliance

- reset() → Observation  
- step(action) → (Observation, float reward, done, info)  
- state() → Observation  

✔ Typed models  
✔ Deterministic grading  
✔ Reward range [0,1]  
✔ Docker-compatible  

---

## 🧠 Why This Matters

This environment models:
- real agricultural trade-offs  
- long-term sustainability challenges  
- constrained decision-making under uncertainty  

👉 Useful for:
- RL benchmarking  
- LLM evaluation  
- policy simulation  

---

## 🏁 Conclusion

AgriDecisionEnv v3 is a realistic and scalable benchmark for evaluating intelligent agents in sustainable agriculture — where short-term gains often conflict with long-term survival.