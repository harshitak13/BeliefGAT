from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from safety.safety_projection import SafetyProjector


@dataclass
class AgentConfig:
    use_ood: bool = False
    use_civic: bool = False
    llm_enabled: bool = True
    use_safety_projection: bool = True


ABLATION_CONFIGS = {
    "V1_gat_only": dict(use_ood=False, use_civic=False, llm_enabled=False, use_safety_projection=False),
    "V2_safegat": dict(use_ood=False, use_civic=False, llm_enabled=True, use_safety_projection=True),
    "V3_beliefgat": dict(use_ood=True, use_civic=False, llm_enabled=True, use_safety_projection=True),
    "V4_citygat": dict(use_ood=False, use_civic=True, llm_enabled=True, use_safety_projection=True),
    "V5_full": dict(use_ood=True, use_civic=True, llm_enabled=True, use_safety_projection=True),
    "V6_no_safety": dict(use_ood=True, use_civic=True, llm_enabled=True, use_safety_projection=False),
    "V7_no_llm": dict(use_ood=True, use_civic=True, llm_enabled=False, use_safety_projection=True),
}


class SafeGATAgent:
    def __init__(self, cfg: dict[str, Any], agent_cfg: AgentConfig | None = None):
        self.cfg = cfg
        self.agent_cfg = agent_cfg or AgentConfig()
        self.num_phases = int(cfg.get("num_phases", 4))
        self.projector = SafetyProjector(self.num_phases, int(cfg.get("g_min", 10)))
        self._elapsed = {}
        self._current = {}

    def act(self, observations: dict[str, np.ndarray]) -> dict[str, int]:
        actions = {}
        for iid, obs in observations.items():
            current = self._current.get(iid, 0)
            elapsed = self._elapsed.get(iid, int(self.cfg.get("g_min", 10)))
            candidate = int(np.argmax(obs[: self.num_phases])) % self.num_phases
            if self.agent_cfg.use_safety_projection:
                candidate = self.projector.project(candidate, range(self.num_phases), current, elapsed).action
            actions[iid] = candidate
            self._elapsed[iid] = elapsed + int(self.cfg.get("action_interval", 10)) if candidate == current else 0
            self._current[iid] = candidate
        return actions

    def risk_score(self, values: dict[str, float]) -> float:
        return (
            self.cfg.get("lambda_u", 2.0) * values.get("uncertain", 0.0)
            + self.cfg.get("lambda_a", 1.5) * values.get("anomaly", 0.0)
            + self.cfg.get("lambda_q", 1.0) * values.get("queue", 0.0)
            + self.cfg.get("lambda_w", 0.5) * values.get("wait", 0.0)
            + self.cfg.get("lambda_s", 1.0) * values.get("spillback", 0.0)
            + self.cfg.get("lambda_OOD", 3.0) * values.get("ood_flag", 0.0) * values.get("ood_score", 0.0)
            + self.cfg.get("lambda_ev", 5.0) * values.get("ev_proximity", 0.0)
            + self.cfg.get("lambda_event", 1.2) * values.get("event_pressure", 0.0)
            + self.cfg.get("lambda_inc", 0.8) * values.get("incident_score", 0.0)
        )
