from __future__ import annotations


def generate_mock_civic_context(intersection_ids: list[str]) -> dict[str, dict]:
    return {
        iid: {
            "ev_eta": 35 if idx == 0 else None,
            "event_pressure": 0.1 if idx % 7 == 0 else 0.0,
            "incident_score": 0.2 if idx % 11 == 0 else 0.0,
        }
        for idx, iid in enumerate(intersection_ids)
    }
