from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class LLMDecision:
    decision: str
    final_phase: int
    policy_trust: str
    causal_summary: str
    safe_trajectory_hint: list[int]
    stakeholder_resolution: str
    adaptation_note: str
    backend: str
    latency_ms: float
    raw: str = ""
    valid: bool = True


class OutputParser:
    REQUIRED = {
        "decision": "accept",
        "policy_trust": "conservative",
        "causal_summary": "fallback to safe RL action",
        "safe_trajectory_hint": [],
        "stakeholder_resolution": "no stakeholder conflict resolved",
        "adaptation_note": "parser fallback",
    }

    @classmethod
    def parse(cls, raw: str, legal_phases: Iterable[int], backend: str, latency_ms: float, rl_action: int | None = None) -> LLMDecision:
        legal = list(legal_phases)
        fallback = legal[0] if rl_action is None else rl_action
        try:
            payload = cls._extract_json(raw)
            phase = int(payload.get("final_phase", fallback))
            if phase not in legal:
                phase = fallback
            return LLMDecision(
                decision=str(payload.get("decision", "accept")),
                final_phase=phase,
                policy_trust=str(payload.get("policy_trust", "conservative")),
                causal_summary=str(payload.get("causal_summary", cls.REQUIRED["causal_summary"])),
                safe_trajectory_hint=list(payload.get("safe_trajectory_hint", []))[:3],
                stakeholder_resolution=str(payload.get("stakeholder_resolution", cls.REQUIRED["stakeholder_resolution"])),
                adaptation_note=str(payload.get("adaptation_note", cls.REQUIRED["adaptation_note"])),
                backend=backend,
                latency_ms=latency_ms,
                raw=raw,
                valid=True,
            )
        except Exception as exc:
            return cls.fallback(backend, latency_ms, fallback, raw, f"parse_error:{exc}")

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
        if fenced:
            return json.loads(fenced.group(1))
        braces = re.search(r"\{.*\}", raw, flags=re.S)
        if braces:
            return json.loads(braces.group(0))
        return json.loads(raw)

    @classmethod
    def fallback(cls, backend: str, latency_ms: float, rl_action: int, raw: str = "", reason: str = "fallback") -> LLMDecision:
        return LLMDecision(
            decision="accept",
            final_phase=rl_action,
            policy_trust="conservative",
            causal_summary=reason,
            safe_trajectory_hint=[rl_action],
            stakeholder_resolution="fallback to RL/safety action",
            adaptation_note=reason,
            backend=backend,
            latency_ms=latency_ms,
            raw=raw,
            valid=False,
        )
