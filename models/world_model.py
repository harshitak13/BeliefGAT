from __future__ import annotations


class WorldModel:
    def rollout(self, intersection_id: str, current_phase: int, horizon: int = 3) -> list[dict]:
        return [
            {"t_plus": k, "intersection_id": intersection_id, "phase": (current_phase + k) % 4, "queue_delta": -0.05 * k}
            for k in range(horizon)
        ]
