from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import sumolib
import traci


class SUMOMultiAgentEnv:
    """SUMO wrapper facade that references existing .sumocfg/.net.xml files."""

    def __init__(self, cfg_dict: dict[str, Any], gui: bool = False, seed: int = 42, mock: bool = False):
        self.cfg = cfg_dict
        self.gui = gui
        self.seed = seed
        self.mock = mock
        self.sumo_cfg = cfg_dict["sumo_cfg"]
        self.net_file = cfg_dict.get("net_file")
        self.num_phases = int(cfg_dict.get("num_phases", 4))
        self.node_feat_dim = int(cfg_dict.get("node_feat_dim", 40))
        self.steps_per_episode = int(cfg_dict.get("steps_per_episode", 3600))
        self.q_max = float(cfg_dict.get("q_max", 50.0))
        self.rng = random.Random(seed)
        self.intersection_ids = self._parse_tls_ids()
        self.num_intersections = len(self.intersection_ids)
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_history: list[float] = []
        self.traci_started = False
        
        
    def _get_queue_length(self):
        total_queue = 0

        for lane in traci.lane.getIDList():
            total_queue += traci.lane.getLastStepHaltingNumber(lane)

        return total_queue

    def _parse_tls_ids(self) -> list[str]:
        if self.net_file and Path(self.net_file).exists():
            root = ET.parse(self.net_file).getroot()
            ids = sorted({tl.attrib["id"] for tl in root.findall("tlLogic") if "id" in tl.attrib})
            if ids:
                return ids
        return [f"tls_{i}" for i in range(int(self.cfg.get("num_intersections", 1)))]

    def get_intersection_ids(self) -> list[str]:
        return self.intersection_ids

    def reset(self):
        if self.traci_started:
            traci.close()

        traci.start([
        "sumo",
        "-c",
        self.sumo_cfg
        ])


        self.traci_started = True

        self._step = 0
        self._queue_history.clear()

        return self._observations()

    def step(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:

        self._step += int(self.cfg.get("action_interval", 10))
        rewards: dict[str, float] = {}
        queue_total = 0.0

        for iid in self.intersection_ids:
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            self._phase[iid] = phase

        for _ in range(int(self.cfg.get("action_interval", 10))):
            traci.simulationStep()

        q = self._get_queue_length()

        for iid in self.intersection_ids:
            queue_total += q
            rewards[iid] = -min(q / self.q_max, 1.0)

        mean_queue = queue_total / max(1, self.num_intersections)

        self._queue_history.append(mean_queue)

        done = self._step >= self.steps_per_episode

        info = {
        "mean_queue": mean_queue,
        "sim_step": self._step,
        "mock": False
    }

        return self._observations(), rewards, done, info

    def _observations(self) -> dict[str, np.ndarray]: #RANDOM VALUES
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
        return {iid: {"ev_eta": None, "event_pressure": 0.0, "incident_score": 0.0} for iid in self.intersection_ids}

    def get_episode_metrics(self) -> dict[str, float]:    #RANDOM VALUES
        q = float(np.mean(self._queue_history)) if self._queue_history else 0.0
        return {"att": q * 5.0, "queue": q, "delay": q * 3.0, "throughput": max(0, int(1000 - q * 10))}

    def close(self):
        if self.traci_started:
            traci.close()
            self.traci_started = False
