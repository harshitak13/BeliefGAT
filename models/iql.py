from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IQLConfig:
    tau_expectile: float = 0.7
    beta_awr: float = 3.0
    lr_q: float = 1e-3
    lr_v: float = 1e-3
    lr_pi: float = 3e-4


def expectile_loss(u, tau: float):
    import torch

    weight = torch.where(u < 0, 1 - tau, tau)
    return weight * u.pow(2)


class OfflineIQLTrainer:
    def __init__(self, cfg: IQLConfig | None = None):
        self.cfg = cfg or IQLConfig()

    def train_placeholder(self, epochs: int) -> dict[str, float]:
        return {"epochs": float(epochs), "iql_loss": 0.0, "policy_loss": 0.0}
