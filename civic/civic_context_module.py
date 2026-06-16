from __future__ import annotations

from civic.mock_civic_generator import generate_mock_civic_context


class CivicContextModule:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def get_context(self, intersection_ids: list[str]) -> dict[str, dict]:
        if not self.enabled:
            return {iid: {"ev_eta": None, "event_pressure": 0.0, "incident_score": 0.0} for iid in intersection_ids}
        return generate_mock_civic_context(intersection_ids)
