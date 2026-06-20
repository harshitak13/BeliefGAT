from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

try:  # SUMO is optional for smoke tests and CI.
    import traci
except Exception:  # pragma: no cover
    traci = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_asset_path(path_value: str | Path | None, label: str) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, REPO_ROOT / "envs" / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} not found: {path_value}. Expected it under {REPO_ROOT / 'envs'}.")


class SUMOMultiAgentEnv:
    """Multi-agent SUMO wrapper with an explicit lightweight mock mode."""

    def __init__(self, cfg_dict: dict[str, Any], gui: bool = False, seed: int = 42, mock: bool = False):
        self.cfg = cfg_dict
        self.gui = gui
        self.seed = seed
        self.mock = mock
        self.sumo_cfg = _resolve_asset_path(cfg_dict["sumo_cfg"], "SUMO config")
        self.net_file = _resolve_asset_path(cfg_dict.get("net_file"), "SUMO network file")
        self.num_phases = int(cfg_dict.get("num_phases", 4))
        self.node_feat_dim = int(cfg_dict.get("node_feat_dim", 40))
        self.steps_per_episode = int(cfg_dict.get("steps_per_episode", 3600))
        self.action_interval = int(cfg_dict.get("action_interval", 10))
        self.q_max = float(cfg_dict.get("q_max", 50.0))
        self.rng = random.Random(seed)
        self.intersection_ids = self._parse_tls_ids()
        self.num_intersections = len(self.intersection_ids)
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_state = {iid: float(5 + idx % 4) for idx, iid in enumerate(self.intersection_ids)}
        self._queue_history: list[float] = []
        self._throughput_history: list[float] = []
        self._delay_history: list[float] = []
        self.traci_started = False

    def _parse_tls_ids(self) -> list[str]:
        if self.net_file and self.net_file.exists():
            root = ET.parse(self.net_file).getroot()
            ids = sorted({tl.attrib["id"] for tl in root.findall("tlLogic") if "id" in tl.attrib})
            if ids:
                return ids
        return [f"tls_{i}" for i in range(int(self.cfg.get("num_intersections", 1)))]

    def get_intersection_ids(self) -> list[str]:
        return self.intersection_ids

    def reset(self) -> dict[str, np.ndarray]:
        if self.mock:
            self._reset_mock_state()
            return self._observations()
        if traci is None:
            raise RuntimeError("SUMO/TraCI is not installed. Install SUMO or run with --mock for smoke tests.")
        if self.traci_started:
            traci.close()
        binary = "sumo-gui" if self.gui else "sumo"
        traci.start([binary, "-c", str(self.sumo_cfg), "--no-warnings", "true"])
        self.traci_started = True
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_history.clear()
        self._delay_history.clear()
        self._throughput_history.clear()
        return self._observations()

    def _reset_mock_state(self) -> None:
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_state = {iid: float(5 + idx % 4) for idx, iid in enumerate(self.intersection_ids)}
        self._queue_history.clear()
        self._delay_history.clear()
        self._throughput_history.clear()

    def step(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        if self.mock:
            return self._step_mock(actions)

        for iid in self.intersection_ids:
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            self._phase[iid] = phase
            if traci.trafficlight.getPhase(iid) != phase:
                traci.trafficlight.setPhase(iid, phase)

        for _ in range(self.action_interval):
            traci.simulationStep()
        self._step += self.action_interval

        queues = self._tls_queue_lengths()
        delays = self._tls_waiting_times()
        rewards = {iid: -min(queues.get(iid, 0.0) / self.q_max, 1.0) for iid in self.intersection_ids}
        mean_queue = float(np.mean(list(queues.values()))) if queues else 0.0
        total_wait = float(sum(delays.values()))
        throughput = float(traci.simulation.getArrivedNumber())
        self._queue_history.append(mean_queue)
        self._delay_history.append(total_wait)
        self._throughput_history.append(throughput)
        done = self._step >= self.steps_per_episode
        info = {"mean_queue": mean_queue, "sim_step": self._step, "mock": False}
        return self._observations(), rewards, done, info

    def _step_mock(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        self._step += self.action_interval
        rewards: dict[str, float] = {}
        for idx, iid in enumerate(self.intersection_ids):
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            old_phase = self._phase[iid]
            self._phase[iid] = phase
            demand = 1.0 + 0.4 * ((idx + self._step // max(1, self.action_interval)) % self.num_phases)
            service = 2.6 if phase == idx % self.num_phases else 1.1
            switching_penalty = 0.7 if phase != old_phase else 0.0
            noise = self.rng.uniform(-0.25, 0.35)
            q = max(0.0, self._queue_state[iid] + demand - service + switching_penalty + noise)
            self._queue_state[iid] = q
            rewards[iid] = -min(q / self.q_max, 1.0)
        mean_queue = float(np.mean(list(self._queue_state.values()))) if self._queue_state else 0.0
        self._queue_history.append(mean_queue)
        self._delay_history.append(mean_queue * self.action_interval)
        self._throughput_history.append(max(0.0, self.num_intersections * 2.0 - mean_queue * 0.1))
        done = self._step >= self.steps_per_episode
        info = {"mean_queue": mean_queue, "sim_step": self._step, "mock": True}
        return self._observations(), rewards, done, info

    def _tls_lanes(self, tls_id: str) -> list[str]:
        if self.mock or traci is None:
            return []
        try:
            return list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        except Exception:
            return []

    def _tls_queue_lengths(self) -> dict[str, float]:
        if self.mock:
            return dict(self._queue_state)
        out = {}
        for iid in self.intersection_ids:
            lanes = self._tls_lanes(iid)
            out[iid] = float(sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes))
        return out

    def _tls_waiting_times(self) -> dict[str, float]:
        if self.mock:
            return {iid: q * self.action_interval for iid, q in self._queue_state.items()}
        out = {}
        for iid in self.intersection_ids:
            lanes = self._tls_lanes(iid)
            out[iid] = float(sum(traci.lane.getWaitingTime(lane) for lane in lanes))
        return out

    def _observations(self) -> dict[str, np.ndarray]:
        obs = {}
        step_norm = min(1.0, self._step / max(1, self.steps_per_episode))
        queues = self._tls_queue_lengths()
        waits = self._tls_waiting_times()
        for iid in self.intersection_ids:
            phase = self._phase.get(iid, 0)
            base = np.zeros(self.node_feat_dim, dtype=np.float32)
            if self.mock:
                q = queues.get(iid, 0.0)
                base[0 : self.num_phases] = q / max(1.0, self.q_max)
                base[4:8] = np.roll(base[0:4], phase)
                base[8:12] = q / max(1.0, self.q_max)
            else:
                lanes = self._tls_lanes(iid)
                lane_count = max(1, len(lanes))
                base[0] = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes) / lane_count
                base[1] = queues.get(iid, 0.0) / lane_count
                base[2] = waits.get(iid, 0.0) / lane_count
                base[3] = sum(traci.lane.getLastStepMeanSpeed(lane) for lane in lanes) / lane_count
            base[12 + phase % min(4, self.num_phases)] = 1.0
            base[20] = step_norm
            obs[iid] = base
        return obs

    def get_civic_context(self) -> dict[str, dict]:
        return {iid: {"ev_eta": None, "event_pressure": 0.0, "incident_score": 0.0} for iid in self.intersection_ids}

    def get_episode_metrics(self) -> dict[str, float]:
        queue = float(np.mean(self._queue_history)) if self._queue_history else 0.0
        delay = float(np.mean(self._delay_history)) if self._delay_history else 0.0
        throughput = float(np.sum(self._throughput_history)) if self._throughput_history else 0.0
        return {"att": delay, "queue": queue, "delay": delay, "throughput": throughput}

    def close(self) -> None:
        if self.traci_started and traci is not None:
            traci.close()
            self.traci_started = False
