from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class CityFlowMultiAgentEnv:
    """CityFlow wrapper facade for Jinan, Hangzhou, and New York datasets."""

    def __init__(self, cfg_dict: dict[str, Any], seed: int = 42, mock: bool = False):
        self.cfg = cfg_dict
        self.seed = seed
        self.mock = mock
        self.roadnet_file = cfg_dict["roadnet_file"]
        self.flow_file = cfg_dict["flow_file"]
        self.num_phases = int(cfg_dict.get("num_phases", 4))
        self.node_feat_dim = int(cfg_dict.get("node_feat_dim", 40))
        self.steps_per_episode = int(cfg_dict.get("steps_per_episode", 3600))
        self.q_max = float(cfg_dict.get("q_max", 50.0))
        self.rng = random.Random(seed)
        self.intersection_ids = self._parse_intersections()
        self.num_intersections = len(self.intersection_ids)
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_history: list[float] = []

    def _parse_intersections(self) -> list[str]:
        p = Path(self.roadnet_file)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                roadnet = json.load(f)
            ids = []
            for inter in roadnet.get("intersections", []):
                if inter.get("virtual", False):
                    continue
                iid = inter.get("id")
                if iid:
                    ids.append(iid)
            if ids:
                return sorted(ids)
        return [f"intersection_{i}" for i in range(int(self.cfg.get("num_intersections", 1)))]

    def get_intersection_ids(self) -> list[str]:
        return self.intersection_ids

    def reset(self) -> dict[str, np.ndarray]:
        self._step = 0
        self._queue_history.clear()
        self._phase = {iid: 0 for iid in self.intersection_ids}
        return self._observations()

    def step(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        self._step += int(self.cfg.get("action_interval", 10))
        rewards: dict[str, float] = {}
        queue_total = 0.0
        event_pressure = 0.15 if self.cfg.get("use_civic_shaping", True) else 0.0
        for idx, iid in enumerate(self.intersection_ids):
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            self._phase[iid] = phase
            demand = 10.0 + (idx % 5) + event_pressure * 5.0
            q = max(0.0, self.rng.gauss(demand, 3.5))
            queue_total += q
            rewards[iid] = -min(q / self.q_max, 1.0) - float(self.cfg.get("beta_event", 0.15)) * event_pressure
        mean_queue = queue_total / max(1, self.num_intersections)
        self._queue_history.append(mean_queue)
        done = self._step >= self.steps_per_episode
        info = {"mean_queue": mean_queue, "sim_step": self._step, "flow_file": self.flow_file, "mock": True}
        return self._observations(), rewards, done, info

    def _observations(self) -> dict[str, np.ndarray]:
        obs = {}
        step_norm = min(1.0, self._step / max(1, self.steps_per_episode))
        for iid in self.intersection_ids:
            phase = self._phase.get(iid, 0)
            base = np.zeros(self.node_feat_dim, dtype=np.float32)
            base[0:4] = np.array([self.rng.random() for _ in range(4)], dtype=np.float32)
            base[4:8] = np.array([self.rng.random() for _ in range(4)], dtype=np.float32)
            base[8:12] = np.array([self.rng.random() for _ in range(4)], dtype=np.float32)
            base[12 + phase % 4] = 1.0
            base[20] = step_norm
            obs[iid] = base
        return obs

    def get_civic_context(self) -> dict[str, dict]:
        ctx = {}
        for idx, iid in enumerate(self.intersection_ids):
            ev_eta = 35 if idx == 0 and self.cfg.get("use_civic_shaping", True) else None
            ctx[iid] = {"ev_eta": ev_eta, "event_pressure": 0.15, "incident_score": 0.05}
        return ctx

    def get_episode_metrics(self) -> dict[str, float]:
        q = float(np.mean(self._queue_history)) if self._queue_history else 0.0
        return {"att": q * 5.0, "queue": q, "delay": q * 3.0, "throughput": max(0, int(1200 - q * 10))}

    def close(self) -> None:
        return None
