from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    context: Any = None


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.data), min(batch_size, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)
