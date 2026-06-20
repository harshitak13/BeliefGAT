from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cityflow
except Exception:  # pragma: no cover
    cityflow = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_asset_path(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, REPO_ROOT / "envs" / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} not found: {path_value}. Expected it under {REPO_ROOT / 'envs'}.")


def _relative_to(base: Path, target: Path) -> str:
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return str(target)


class CityFlowMultiAgentEnv:
    """CityFlow multi-agent wrapper with explicit real and mock execution modes."""

    def __init__(self, cfg_dict: dict[str, Any], seed: int = 42, mock: bool = False):
        self.cfg = cfg_dict
        self.seed = seed
        self.mock = mock
        self.roadnet_file = _resolve_asset_path(cfg_dict["roadnet_file"], "CityFlow roadnet file")
        self.flow_file = _resolve_asset_path(cfg_dict["flow_file"], "CityFlow flow file")
        self.num_phases = int(cfg_dict.get("num_phases", 4))
        self.node_feat_dim = int(cfg_dict.get("node_feat_dim", 40))
        self.steps_per_episode = int(cfg_dict.get("steps_per_episode", 3600))
        self.action_interval = int(cfg_dict.get("action_interval", 10))
        self.q_max = float(cfg_dict.get("q_max", 50.0))
        self.rng = random.Random(seed)
        self.intersection_ids = self._parse_intersections()
        self.num_intersections = len(self.intersection_ids)
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_state = {iid: float(4 + idx % 6) for idx, iid in enumerate(self.intersection_ids)}
        self._queue_history: list[float] = []
        self._delay_history: list[float] = []
        self._throughput_history: list[float] = []
        self._engine = None
        self._flow_demand = self._load_flow_demand()

    def _parse_intersections(self) -> list[str]:
        p = self.roadnet_file
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                roadnet = json.load(f)
            ids = [inter["id"] for inter in roadnet.get("intersections", []) if inter.get("id") and not inter.get("virtual", False)]
            if ids:
                return sorted(ids)
        return [f"intersection_{i}" for i in range(int(self.cfg.get("num_intersections", 1)))]

    def _load_flow_demand(self) -> dict[int, int]:
        p = self.flow_file
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            flows = json.load(f)
        demand: dict[int, int] = {}
        for row in flows:
            start = int(float(row.get("startTime", 0)))
            end = int(float(row.get("endTime", start)))
            interval = max(1, int(float(row.get("interval", 1))))
            for t in range(start, end + 1, interval):
                demand[t] = demand.get(t, 0) + 1
        return demand

    def _cityflow_config_path(self) -> str:
        explicit = self.cfg.get("cityflow_config")
        if explicit:
            return str(_resolve_asset_path(explicit, "CityFlow config"))
        roadnet = self.roadnet_file
        flow = self.flow_file
        base_dir = roadnet.parent if roadnet.parent == flow.parent else REPO_ROOT
        out_dir = (REPO_ROOT / ".cache" / "cityflow").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / f"{self.cfg.get('env_name', 'cityflow')}_{self.seed}.json"
        payload = {
            "interval": 1.0,
            "seed": self.seed,
            "dir": str(base_dir),
            "roadnetFile": _relative_to(base_dir, roadnet),
            "flowFile": _relative_to(base_dir, flow),
            "rlTrafficLight": True,
            "saveReplay": bool(self.cfg.get("cityflow_save_replay", False)),
            "roadnetLogFile": "roadnetLog.json",
            "replayLogFile": "replayLog.txt",
        }
        cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(cfg_path)

    def get_intersection_ids(self) -> list[str]:
        return self.intersection_ids

    def reset(self) -> dict[str, np.ndarray]:
        self._step = 0
        self._phase = {iid: 0 for iid in self.intersection_ids}
        self._queue_state = {iid: float(4 + idx % 6) for idx, iid in enumerate(self.intersection_ids)}
        self._queue_history.clear()
        self._delay_history.clear()
        self._throughput_history.clear()
        if not self.mock:
            if cityflow is None:
                raise RuntimeError("CityFlow is not installed. Install cityflow or run with --mock for smoke tests.")
            self._engine = cityflow.Engine(self._cityflow_config_path(), int(self.cfg.get("cityflow_thread_num", 1)))
        return self._observations()

    def step(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        if self.mock:
            return self._step_mock(actions)
        for iid in self.intersection_ids:
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            self._phase[iid] = phase
            self._engine.set_tl_phase(iid, phase)
        for _ in range(self.action_interval):
            self._engine.next_step()
        self._step += self.action_interval
        queues = self._lane_waiting_count()
        total_queue = float(sum(queues.values()))
        mean_queue = total_queue / max(1, self.num_intersections)
        rewards = {iid: -min(mean_queue / self.q_max, 1.0) for iid in self.intersection_ids}
        self._queue_history.append(mean_queue)
        self._delay_history.append(total_queue)
        self._throughput_history.append(float(len(self._lane_vehicle_count())))
        done = self._step >= self.steps_per_episode
        info = {"mean_queue": mean_queue, "sim_step": self._step, "flow_file": str(self.flow_file), "mock": False}
        return self._observations(), rewards, done, info

    def _step_mock(self, actions: dict[str, int]) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict]:
        self._step += self.action_interval
        event_pressure = 0.15 if self.cfg.get("use_civic_shaping", True) else 0.0
        demand_now = self._flow_demand.get(self._step, 0) / max(1, self.num_intersections)
        rewards: dict[str, float] = {}
        for idx, iid in enumerate(self.intersection_ids):
            phase = int(actions.get(iid, self._phase[iid])) % self.num_phases
            old_phase = self._phase[iid]
            self._phase[iid] = phase
            demand = 0.8 + demand_now + 0.25 * (idx % self.num_phases) + event_pressure
            service = 2.4 if phase == idx % self.num_phases else 1.0
            switching_penalty = 0.6 if phase != old_phase else 0.0
            q = max(0.0, self._queue_state[iid] + demand - service + switching_penalty + self.rng.uniform(-0.2, 0.3))
            self._queue_state[iid] = q
            rewards[iid] = -min(q / self.q_max, 1.0) - float(self.cfg.get("beta_event", 0.15)) * event_pressure
        mean_queue = float(np.mean(list(self._queue_state.values()))) if self._queue_state else 0.0
        self._queue_history.append(mean_queue)
        self._delay_history.append(mean_queue * self.action_interval)
        self._throughput_history.append(max(0.0, self.num_intersections * 2.0 - mean_queue * 0.2))
        done = self._step >= self.steps_per_episode
        info = {"mean_queue": mean_queue, "sim_step": self._step, "flow_file": str(self.flow_file), "mock": True}
        return self._observations(), rewards, done, info

    def _lane_waiting_count(self) -> dict[str, int]:
        if self.mock or self._engine is None:
            return {iid: int(q) for iid, q in self._queue_state.items()}
        return self._engine.get_lane_waiting_vehicle_count()

    def _lane_vehicle_count(self) -> dict[str, int]:
        if self.mock or self._engine is None:
            return {iid: int(q) for iid, q in self._queue_state.items()}
        return self._engine.get_lane_vehicle_count()

    def _observations(self) -> dict[str, np.ndarray]:
        obs = {}
        step_norm = min(1.0, self._step / max(1, self.steps_per_episode))
        if self.mock:
            queue_by_id = dict(self._queue_state)
            vehicle_by_id = dict(self._queue_state)
        else:
            lane_waits = self._lane_waiting_count()
            lane_counts = self._lane_vehicle_count()
            total_wait = float(sum(lane_waits.values()))
            total_count = float(sum(lane_counts.values()))
            queue_by_id = {iid: total_wait / max(1, self.num_intersections) for iid in self.intersection_ids}
            vehicle_by_id = {iid: total_count / max(1, self.num_intersections) for iid in self.intersection_ids}
        for iid in self.intersection_ids:
            phase = self._phase.get(iid, 0)
            q = queue_by_id.get(iid, 0.0)
            vehicles = vehicle_by_id.get(iid, 0.0)
            base = np.zeros(self.node_feat_dim, dtype=np.float32)
            base[0 : self.num_phases] = min(1.0, q / max(1.0, self.q_max))
            base[4:8] = min(1.0, vehicles / max(1.0, self.q_max))
            base[8:12] = min(1.0, q / max(1.0, self.q_max))
            base[12 + phase % min(4, self.num_phases)] = 1.0
            base[20] = step_norm
            obs[iid] = base
        return obs

    def get_civic_context(self) -> dict[str, dict]:
        ctx = {}
        for idx, iid in enumerate(self.intersection_ids):
            ev_eta = 35 if idx == 0 and self.cfg.get("use_civic_shaping", True) else None
            ctx[iid] = {"ev_eta": ev_eta, "event_pressure": 0.15 if idx % 7 == 0 else 0.0, "incident_score": 0.05 if idx % 11 == 0 else 0.0}
        return ctx

    def get_episode_metrics(self) -> dict[str, float]:
        q = float(np.mean(self._queue_history)) if self._queue_history else 0.0
        delay = float(np.mean(self._delay_history)) if self._delay_history else 0.0
        throughput = float(np.sum(self._throughput_history)) if self._throughput_history else 0.0
        return {"att": delay, "queue": q, "delay": delay, "throughput": throughput}

    def close(self) -> None:
        self._engine = None
