from __future__ import annotations

import json
from typing import Any


class PromptBuilder:
    @staticmethod
    def build(payload: dict[str, Any]) -> str:
        legal = payload.get("legal_phases", [0, 1, 2, 3])
        return (
            "Block 1 - Deployment & Temporal Context\n"
            f"{json.dumps(payload.get('temporal_context', {}), indent=2)}\n\n"
            "Block 2 - OOD Characterization (BeliefGAT)\n"
            f"{json.dumps(payload.get('ood', {}), indent=2)}\n\n"
            "Block 3 - Policy Confidence (BeliefGAT)\n"
            f"{json.dumps(payload.get('policy', {}), indent=2)}\n\n"
            "Block 4 - Current Traffic State + Neighbour Summary\n"
            f"{json.dumps(payload.get('traffic_state', {}), indent=2)}\n\n"
            "Block 5 - World Model Rollouts (OOD-aware)\n"
            f"{json.dumps(payload.get('rollouts', {}), indent=2)}\n\n"
            "Block 6 - Civic Context (CityGAT): EVs, Events, Incidents\n"
            f"{json.dumps(payload.get('civic', {}), indent=2)}\n\n"
            "Block 7 - Return compact JSON only. Do not include hidden reasoning.\n"
            "Schema: {\"decision\":\"accept|override\", \"final_phase\":int, "
            "\"policy_trust\":\"online|offline|conservative\", "
            "\"causal_summary\":\"one sentence\", \"safe_trajectory_hint\":[int,int,int], "
            "\"stakeholder_resolution\":\"one sentence\", \"adaptation_note\":\"one sentence\"}\n"
            f"legal_phases={legal}"
        )
