import random
from models import Action, Observation, Reward, StepInfo

CROP_NITROGEN_COST = {"rice": 0.20, "wheat": 0.10, "none": 0.0}
CROP_BASE_YIELD    = {"rice": 0.85, "wheat": 0.70, "none": 0.0}
CROP_WATER_NEED    = {"rice": 0.25, "wheat": 0.12, "none": 0.0}

WEATHER_SEQUENCE = ["normal", "normal", "rainy", "drought", "normal",
                    "rainy", "normal", "drought", "normal", "normal"]

SCENARIOS = {
    "fertile":  dict(nitrogen=0.75, moisture=0.65, groundwater=0.90, budget=150.0),
    "drought":  dict(nitrogen=0.45, moisture=0.20, groundwater=0.35, budget=100.0),
    "degraded": dict(nitrogen=0.25, moisture=0.45, groundwater=0.60, budget=80.0),
    "default":  dict(nitrogen=0.50, moisture=0.50, groundwater=0.70, budget=120.0),
}

EPISODE_LENGTH       = 5
FERTILIZER_THRESHOLD = 0.6
IRRIGATION_THRESHOLD = 0.65
BUDGET_PER_STEP      = 20.0
EARLY_STOP_SOIL      = 0.10
EARLY_STOP_BUDGET    = -30.0


def _clamp(v, lo=0.0, hi=1.0):
    """Hard clamp — applied to every computed float value."""
    return max(lo, min(hi, v))


class AgriEnv:
    """
    AgriDecisionEnv v3 – Sustainable Farming RL Environment.

    API (OpenEnv compliant):
        reset()      -> Observation
        step(action) -> (Observation, float, bool, dict)
        state()      -> Observation

    - reward is a plain float, always in [0.0, 1.0]
    - info is a plain dict (not Pydantic)
    - No randomness in step() — fully deterministic via WEATHER_SEQUENCE
    """

    def __init__(self, scenario="default", seed=42):
        self._scenario = scenario
        self._seed = seed
        self._nitrogen    = 0.5
        self._moisture    = 0.5
        self._groundwater = 0.7
        self._budget      = 120.0
        self._season      = 0
        self._last_crop   = "none"
        self._done        = False
        self._history     = []
        self._fertilizer_window = []

    def reset(self):
        preset = SCENARIOS.get(self._scenario, SCENARIOS["default"])
        self._nitrogen    = preset["nitrogen"]
        self._moisture    = preset["moisture"]
        self._groundwater = preset["groundwater"]
        self._budget      = preset["budget"]
        self._season      = 0
        self._last_crop   = "none"
        self._done        = False
        self._history     = []
        self._fertilizer_window = []
        return self.state()

    def step(self, action):
        """
        Returns: (Observation, float reward [0,1], bool done, dict info)
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        crop       = action.crop if action.crop in CROP_NITROGEN_COST else "none"
        fertilizer = _clamp(round(float(action.fertilizer), 2))
        irrigation = _clamp(round(float(action.irrigation), 2))

        # Weather — pure lookup, zero randomness
        weather = WEATHER_SEQUENCE[self._season % len(WEATHER_SEQUENCE)]

        # Nitrogen
        new_nitrogen = _clamp(
            self._nitrogen - CROP_NITROGEN_COST[crop] + fertilizer * 0.30
        )

        # Delayed fertilizer penalty (3-step rolling avg)
        self._fertilizer_window.append(fertilizer)
        if len(self._fertilizer_window) > 3:
            self._fertilizer_window.pop(0)
        avg_fert = sum(self._fertilizer_window) / len(self._fertilizer_window)
        delayed_fert_penalty = _clamp(max(0.0, avg_fert - FERTILIZER_THRESHOLD) * 0.4)

        # Moisture + groundwater
        weather_m = {"rainy": +0.12, "normal": 0.0, "drought": -0.10}[weather]
        new_moisture    = _clamp(self._moisture + irrigation * 0.28 + weather_m - 0.08)
        new_groundwater = _clamp(self._groundwater - irrigation * 0.18 - CROP_WATER_NEED[crop])

        # Budget
        new_budget = self._budget - (BUDGET_PER_STEP + fertilizer * 15.0 + irrigation * 10.0)

        # Derived metrics
        soil_quality = _clamp(new_nitrogen * 0.6 + new_moisture * 0.4)
        yield_score  = _clamp(CROP_BASE_YIELD[crop] * (new_nitrogen + new_moisture) / 2.0)

        # Penalties — each individually clamped
        fert_penalty        = _clamp(max(0.0, fertilizer  - FERTILIZER_THRESHOLD) * 0.35)
        irrig_penalty       = _clamp(max(0.0, irrigation  - IRRIGATION_THRESHOLD) * 0.30)
        groundwater_penalty = _clamp(max(0.0, 0.20 - new_groundwater) * 0.50)
        budget_penalty      = _clamp(max(0.0, -new_budget / 100.0) * 0.40)
        monocrop_penalty    = 0.12 if (crop == self._last_crop and crop != "none") else 0.0
        soil_bonus          = 0.15 if new_nitrogen >= 0.45 and new_moisture >= 0.35 else 0.0

        # Reward — HARD CLAMP guarantees [0.0, 1.0]
        reward: float = round(_clamp(
            yield_score + soil_bonus
            - fert_penalty - delayed_fert_penalty
            - irrig_penalty - monocrop_penalty
            - groundwater_penalty - budget_penalty
        ), 4)

        # History
        self._history.append({
            "season": self._season, "crop": crop, "weather": weather,
            "nitrogen": new_nitrogen, "moisture": new_moisture,
            "groundwater": new_groundwater, "soil_quality": soil_quality,
            "budget": new_budget, "reward": reward,
        })

        # Update state
        self._nitrogen    = round(new_nitrogen,    4)
        self._moisture    = round(new_moisture,    4)
        self._groundwater = round(new_groundwater, 4)
        self._budget      = round(new_budget,      4)
        self._last_crop   = crop
        self._season     += 1

        terminated = soil_quality < EARLY_STOP_SOIL or new_budget < EARLY_STOP_BUDGET
        self._done = self._season >= EPISODE_LENGTH or terminated

        info = {
            "yield_score":      round(yield_score, 4),
            "soil_health":      round(soil_quality, 4),
            "water_used":       round(irrigation + CROP_WATER_NEED[crop], 4),
            "budget_remaining": round(new_budget, 2),
            "weather":          weather,
            "penalties": {
                "fertilizer":   round(fert_penalty, 4),
                "delayed_fert": round(delayed_fert_penalty, 4),
                "irrigation":   round(irrig_penalty, 4),
                "monocrop":     round(monocrop_penalty, 4),
                "groundwater":  round(groundwater_penalty, 4),
                "budget":       round(budget_penalty, 4),
            },
        }
        return self.state(), reward, self._done, info

    def state(self):
        weather      = WEATHER_SEQUENCE[self._season % len(WEATHER_SEQUENCE)]
        soil_quality = _clamp(self._nitrogen * 0.6 + self._moisture * 0.4)
        return Observation(
            nitrogen    = self._nitrogen,
            moisture    = self._moisture,
            soil_quality= round(soil_quality, 4),
            last_crop   = self._last_crop,
            season      = self._season,
            weather     = weather,
            groundwater = self._groundwater,
            budget      = self._budget,
        )

    @property
    def history(self): return self._history

    @property
    def done(self): return self._done
