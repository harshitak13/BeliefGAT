from __future__ import annotations

import json
from pathlib import Path


class BackendRanker:
    def score(self, traffic_efficiency: float, intervention_control: float, large_network_reward: float) -> float:
        raw = 0.35 * traffic_efficiency + 0.40 * intervention_control + 0.25 * large_network_reward
        return max(0.0, min(100.0, raw))

    def rank(self, rows: list[dict]) -> list[dict]:
        scored = []
        for row in rows:
            item = dict(row)
            item["composite"] = self.score(
                float(row.get("traffic_efficiency", 0)),
                float(row.get("intervention_control", 0)),
                float(row.get("large_network_reward", 0)),
            )
            scored.append(item)
        return sorted(scored, key=lambda r: r["composite"], reverse=True)

    def save(self, rows: list[dict], path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.rank(rows), indent=2), encoding="utf-8")
