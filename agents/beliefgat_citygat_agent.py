from __future__ import annotations

from agents.safegat_agent import AgentConfig, SafeGATAgent


class BeliefGATCityGATAgent(SafeGATAgent):
    def __init__(self, cfg: dict, agent_cfg: AgentConfig | None = None):
        super().__init__(cfg, agent_cfg or AgentConfig(use_ood=True, use_civic=True, llm_enabled=True, use_safety_projection=True))
