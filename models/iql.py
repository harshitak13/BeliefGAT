from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from models.gat_dqn import GATDQN


@dataclass
class IQLConfig:
    tau_expectile: float = 0.7
    beta_awr: float = 3.0
    lr_q: float = 1e-3
    lr_v: float = 1e-3
    lr_pi: float = 3e-4
    gamma: float = 0.99
    max_samples: int = 50000


def expectile_loss(u, tau: float):
    weight = torch.where(u < 0, 1 - tau, tau)
    return weight * u.pow(2)


if nn is not None:
    class ValueNet(nn.Module):
        def __init__(self, node_feat_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class PolicyNet(nn.Module):
        def __init__(self, node_feat_dim: int, hidden_dim: int, num_phases: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_phases),
            )

        def forward(self, x):
            return self.net(x)
else:
    ValueNet = None
    PolicyNet = None


class OfflineIQLTrainer:
    """Minimal offline IQL-style trainer that writes real torch checkpoints."""

    def __init__(self, cfg: dict[str, Any], trainer_cfg: IQLConfig | None = None, seed: int = 42):
        if torch is None or nn is None:
            raise ImportError("torch is required for offline IQL training")
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg or IQLConfig(gamma=float(cfg.get("gamma", 0.99)))
        self.seed = seed
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.node_feat_dim = int(cfg.get("node_feat_dim", 40))
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_phases = int(cfg.get("num_phases", 4))
        self.q_net = GATDQN(self.node_feat_dim, self.hidden_dim, self.num_phases)
        self.v_net = ValueNet(self.node_feat_dim, self.hidden_dim)
        self.pi_net = PolicyNet(self.node_feat_dim, self.hidden_dim, self.num_phases)

    def build_dataset(self, min_samples: int) -> dict[str, torch.Tensor]:
        samples = min(self.trainer_cfg.max_samples, max(min_samples, int(self.cfg.get("learning_start", 1000))))
        flow_times = self._load_flow_times()
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[np.ndarray] = []
        dones: list[float] = []
        horizon = int(self.cfg.get("steps_per_episode", 3600))
        q_max = float(self.cfg.get("q_max", 50.0))
        for idx in range(samples):
            t = (idx * int(self.cfg.get("action_interval", 10))) % max(1, horizon)
            demand = self._demand_at(flow_times, t)
            phase_pressure = np.array(
                [self.rng.random() + 0.15 * ((idx + phase) % self.num_phases) + demand for phase in range(self.num_phases)],
                dtype=np.float32,
            )
            action = int(np.argmax(phase_pressure))
            state = np.zeros(self.node_feat_dim, dtype=np.float32)
            state[0 : self.num_phases] = np.clip(phase_pressure / max(1.0, q_max), 0.0, 1.0)
            state[4:8] = np.roll(state[0:4], action)
            state[12 + action % min(4, self.num_phases)] = 1.0
            state[20] = min(1.0, t / max(1, horizon))
            next_pressure = phase_pressure.copy()
            next_pressure[action] = max(0.0, next_pressure[action] - 1.5)
            next_pressure += np.asarray([self.rng.uniform(0.0, 0.25) for _ in range(self.num_phases)], dtype=np.float32)
            next_state = state.copy()
            next_state[0 : self.num_phases] = np.clip(next_pressure / max(1.0, q_max), 0.0, 1.0)
            reward = -float(next_pressure.mean() / max(1.0, q_max))
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(t + int(self.cfg.get("action_interval", 10)) >= horizon))
        return {
            "states": torch.tensor(np.asarray(states), dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.tensor(np.asarray(next_states), dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
        }

    def _load_flow_times(self) -> list[int]:
        flow_file = self.cfg.get("flow_file") or self.cfg.get("rou_file")
        if not flow_file or not Path(flow_file).exists() or not str(flow_file).endswith(".json"):
            return []
        with Path(flow_file).open("r", encoding="utf-8") as f:
            rows = json.load(f)
        return [int(float(row.get("startTime", 0))) for row in rows if isinstance(row, dict)]

    def _demand_at(self, flow_times: list[int], t: int) -> float:
        if not flow_times:
            return 0.2 + 0.2 * np.sin(t / 300.0)
        window = int(self.cfg.get("action_interval", 10))
        count = sum(1 for ft in flow_times if t <= ft < t + window)
        return min(3.0, count / max(1, int(self.cfg.get("num_intersections", 1))))

    def train(self, epochs: int, batch_size: int = 256) -> dict[str, Any]:
        data = self.build_dataset(max(batch_size * 4, epochs * batch_size))
        q_opt = torch.optim.Adam(self.q_net.parameters(), lr=self.trainer_cfg.lr_q)
        v_opt = torch.optim.Adam(self.v_net.parameters(), lr=self.trainer_cfg.lr_v)
        pi_opt = torch.optim.Adam(self.pi_net.parameters(), lr=self.trainer_cfg.lr_pi)
        n = data["states"].shape[0]
        last = {"q_loss": 0.0, "v_loss": 0.0, "policy_loss": 0.0}
        for _ in range(epochs):
            idx = torch.randint(0, n, (min(batch_size, n),))
            s = data["states"][idx]
            a = data["actions"][idx]
            r = data["rewards"][idx]
            ns = data["next_states"][idx]
            done = data["dones"][idx]
            with torch.no_grad():
                target_q = r + self.trainer_cfg.gamma * (1.0 - done) * self.v_net(ns)
            q_all = self.q_net(s)
            q_sa = q_all.gather(1, a[:, None]).squeeze(1)
            q_loss = F.mse_loss(q_sa, target_q)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            with torch.no_grad():
                min_q = self.q_net(s).max(dim=1).values
            v = self.v_net(s)
            v_loss = expectile_loss(min_q - v, self.trainer_cfg.tau_expectile).mean()
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                adv = torch.clamp(q_sa - self.v_net(s), max=10.0)
                weights = torch.exp(self.trainer_cfg.beta_awr * adv).clamp(max=100.0)
            logits = self.pi_net(s)
            policy_loss = (F.cross_entropy(logits, a, reduction="none") * weights).mean()
            pi_opt.zero_grad()
            policy_loss.backward()
            pi_opt.step()
            last = {"q_loss": float(q_loss.item()), "v_loss": float(v_loss.item()), "policy_loss": float(policy_loss.item())}
        return {
            "checkpoint_version": 1,
            "algorithm": "iql_dqn_warmstart",
            "config": {
                "node_feat_dim": self.node_feat_dim,
                "hidden_dim": self.hidden_dim,
                "num_phases": self.num_phases,
                "gamma": self.trainer_cfg.gamma,
            },
            "model_state_dict": self.q_net.state_dict(),
            "value_state_dict": self.v_net.state_dict(),
            "policy_state_dict": self.pi_net.state_dict(),
            "dataset_size": n,
            "epochs": epochs,
            **last,
        }
