from __future__ import annotations

from typing import Any

import numpy as np

from agents.safegat_agent import AgentConfig, SafeGATAgent


class FixedTimeWebsterAgent:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.num_phases = int(cfg.get("num_phases", 4))
        self.action_interval = int(cfg.get("action_interval", 10))
        self.cycle = max(self.num_phases, int(cfg.get("webster_cycle", self.num_phases * 3)))
        self._tick = 0

    def act(self, observations: dict[str, np.ndarray], civic_context: dict[str, dict] | None = None) -> dict[str, int]:
        phase = (self._tick // max(1, self.cycle // self.num_phases)) % self.num_phases
        self._tick += 1
        return {iid: int(phase) for iid in observations}

    def observe(self, *args, **kwargs) -> None:
        return None

    def metrics_snapshot(self, reset: bool = True) -> dict[str, float]:
        return {}


class ActuatedAgent:
    def __init__(self, cfg: dict[str, Any]):
        self.num_phases = int(cfg.get("num_phases", 4))

    def act(self, observations: dict[str, np.ndarray], civic_context: dict[str, dict] | None = None) -> dict[str, int]:
        return {iid: int(np.argmax(obs[0 : self.num_phases])) % self.num_phases for iid, obs in observations.items()}

    def observe(self, *args, **kwargs) -> None:
        return None

    def metrics_snapshot(self, reset: bool = True) -> dict[str, float]:
        return {}


class CoLightLikeAgent(ActuatedAgent):
    def act(self, observations: dict[str, np.ndarray], civic_context: dict[str, dict] | None = None) -> dict[str, int]:
        if not observations:
            return {}
        mean_pressure = np.mean([obs[0 : self.num_phases] for obs in observations.values()], axis=0)
        return {iid: int(np.argmax(0.7 * obs[0 : self.num_phases] + 0.3 * mean_pressure)) % self.num_phases for iid, obs in observations.items()}


def build_baseline_agent(name: str, cfg: dict[str, Any]):
    key = name.lower()
    if key == "webster":
        return FixedTimeWebsterAgent(cfg)
    if key == "actuated":
        return ActuatedAgent(cfg)
    if key == "colight":
        return CoLightLikeAgent(cfg)
    if key in {"plain_dqn", "gat_dqn"}:
        return SafeGATAgent(cfg, AgentConfig(use_ood=False, use_civic=False, llm_enabled=False, use_safety_projection=False))
    if key in {"llmlight", "illm_tsc"}:
        return SafeGATAgent(cfg, AgentConfig(use_ood=False, use_civic=True, llm_enabled=True, use_safety_projection=True))
    raise ValueError(f"Unknown baseline: {name}")
