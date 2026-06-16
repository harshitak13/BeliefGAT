from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class SafetyProjectionResult:
    action: int
    adjusted: bool
    reason: str


class SafetyProjector:
    """Projects candidate phases into the legal, minimum-green safe set."""

    def __init__(self, num_phases: int = 4, g_min: int = 10):
        self.num_phases = num_phases
        self.g_min = g_min

    def safe_set(
        self,
        legal_phases: Iterable[int],
        current_phase: int,
        elapsed_green: int,
        transition_row: Mapping[int, bool] | None = None,
    ) -> list[int]:
        if elapsed_green < self.g_min:
            return [current_phase]
        legal = list(legal_phases)
        if transition_row is None:
            return legal
        return [a for a in legal if transition_row.get(a, False)]

    def project(
        self,
        candidate: int,
        legal_phases: Iterable[int],
        current_phase: int,
        elapsed_green: int,
        transition_row: Mapping[int, bool] | None = None,
    ) -> SafetyProjectionResult:
        omega = self.safe_set(legal_phases, current_phase, elapsed_green, transition_row)
        if candidate in omega:
            return SafetyProjectionResult(candidate, False, "candidate_safe")
        if elapsed_green < self.g_min:
            return SafetyProjectionResult(current_phase, True, "minimum_green")
        if not omega:
            return SafetyProjectionResult(current_phase, True, "empty_safe_set")
        projected = min(omega, key=lambda a: abs(a - candidate))
        return SafetyProjectionResult(projected, True, "nearest_safe_phase")
