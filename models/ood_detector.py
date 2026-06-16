from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SHIFT_LABELS = [
    "novel_demand_pattern",
    "sensor_degradation",
    "incident_not_in_training",
    "weather_effect",
    "time_pattern_unseen",
    "infrastructure_change",
    "event_dispersal",
    "emergency_corridor",
]


@dataclass
class OODResult:
    score: float
    is_ood: bool
    shift_type: str


class OODDetector:
    def __init__(self, tau_ood: float = 0.05):
        self.tau_ood = tau_ood

    def score(self, embedding: np.ndarray | list[float]) -> OODResult:
        arr = np.asarray(embedding, dtype=float)
        val = float(np.var(arr)) if arr.size else 0.0
        idx = int(abs(val * 1000)) % len(SHIFT_LABELS)
        return OODResult(val, val > self.tau_ood, SHIFT_LABELS[idx])
