from __future__ import annotations

import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

from llm.output_parser import LLMDecision
from llm.prompt_builder import PromptBuilder
from llm.llm_gateway import LLMGateway
from models.gat_dqn import GATDQN
from models.ood_detector import OODDetector, OODResult
from models.world_model import WorldModel
from safety.safety_projection import SafetyProjector
from training.replay_buffer import Transition


@dataclass
class AgentConfig:
    use_ood: bool = False
    use_civic: bool = False
    llm_enabled: bool = True
    use_safety_projection: bool = True


ABLATION_CONFIGS = {
    "V1_gat_only": dict(use_ood=False, use_civic=False, llm_enabled=False, use_safety_projection=False),
    "V2_safegat": dict(use_ood=False, use_civic=False, llm_enabled=True, use_safety_projection=True),
    "V3_beliefgat": dict(use_ood=True, use_civic=False, llm_enabled=True, use_safety_projection=True),
    "V4_citygat": dict(use_ood=False, use_civic=True, llm_enabled=True, use_safety_projection=True),
    "V5_full": dict(use_ood=True, use_civic=True, llm_enabled=True, use_safety_projection=True),
    "V6_no_safety": dict(use_ood=True, use_civic=True, llm_enabled=True, use_safety_projection=False),
    "V7_no_llm": dict(use_ood=True, use_civic=True, llm_enabled=False, use_safety_projection=True),
}


class SafeGATAgent:
    def __init__(self, cfg: dict[str, Any], agent_cfg: AgentConfig | None = None):
        self.cfg = cfg
        self.agent_cfg = agent_cfg or AgentConfig()
        self.num_phases = int(cfg.get("num_phases", 4))
        self.node_feat_dim = int(cfg.get("node_feat_dim", 40))
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.learning_start = int(cfg.get("learning_start", 1000))
        self.target_update_freq = int(cfg.get("target_update_freq", 500))
        self.epsilon = float(cfg.get("epsilon_start", 0.1))
        self.epsilon_end = float(cfg.get("epsilon_end", 0.05))
        self.epsilon_decay = float(cfg.get("epsilon_decay", 0.995))
        self.projector = SafetyProjector(self.num_phases, int(cfg.get("g_min", 10)))
        self.ood_detector = OODDetector(float(cfg.get("tau_OOD", cfg.get("tau_ood", 0.05))))
        self.world_model = WorldModel()
        self.prompt_builder = PromptBuilder()
        self._elapsed: dict[str, int] = {}
        self._current: dict[str, int] = {}
        self._replay: deque[Transition] = deque(maxlen=int(cfg.get("replay_buffer_size", 100000)))
        self._updates = 0
        self._offline_loaded = False
        self._stats = Counter()
        self._latencies: list[float] = []
        self._kl_values: list[float] = []
        self._last_loss = 0.0
        self.llm_gateway = None
        self._llm_init_error = ""
        if self.agent_cfg.llm_enabled and bool(cfg.get("remote_llm", False)):
            try:
                self.llm_gateway = LLMGateway(
                    str(cfg.get("llm_backend", "llama_31_8b")),
                    timeout_ms=int(cfg.get("llm_timeout_ms", 500)),
                    api_keys_path=str(cfg.get("api_keys_path", "configs/api_keys.yaml")),
                    log_path=str(cfg.get("llm_log_path", "results/decisions.jsonl")),
                )
            except Exception as exc:
                self._llm_init_error = str(exc)
        self._init_networks()

    def _init_networks(self) -> None:
        self.model = None
        self.target_model = None
        self.optimizer = None
        if torch is None:
            return
        self.model = GATDQN(self.node_feat_dim, self.hidden_dim, self.num_phases)
        self.target_model = GATDQN(self.node_feat_dim, self.hidden_dim, self.num_phases)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.get("lr_base", 1e-3)))

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        if torch is None or self.model is None or self.target_model is None:
            raise ImportError("torch is required to load offline checkpoints")
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Offline checkpoint not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict") or checkpoint.get("q_state_dict")
        if state is None:
            raise ValueError(f"Checkpoint {path} does not contain a model_state_dict")
        self.model.load_state_dict(state, strict=False)
        self.target_model.load_state_dict(self.model.state_dict())
        self._offline_loaded = True
        self.epsilon = min(self.epsilon, 0.1)

    def act(self, observations: dict[str, np.ndarray], civic_context: dict[str, dict] | None = None) -> dict[str, int]:
        actions: dict[str, int] = {}
        civic_context = civic_context or {}
        for iid, obs in observations.items():
            current = self._current.get(iid, 0)
            elapsed = self._elapsed.get(iid, int(self.cfg.get("g_min", 10)))
            q_values = self._q_values(obs)
            candidate = self._select_action(q_values, obs)
            ood = self._ood(obs)
            civic = civic_context.get(iid, {"ev_eta": None, "event_pressure": 0.0, "incident_score": 0.0})
            risk = self.risk_score(
                {
                    "uncertain": float(np.std(q_values)),
                    "anomaly": float(ood.score),
                    "queue": float(np.mean(obs[0 : self.num_phases])),
                    "wait": float(np.mean(obs[8:12])) if obs.size >= 12 else 0.0,
                    "spillback": float(np.max(obs[0 : self.num_phases])) if obs.size else 0.0,
                    "ood_flag": float(ood.is_ood),
                    "ood_score": float(ood.score),
                    "ev_proximity": 1.0 if civic.get("ev_eta") is not None else 0.0,
                    "event_pressure": float(civic.get("event_pressure", 0.0)),
                    "incident_score": float(civic.get("incident_score", 0.0)),
                }
            )
            llm_action = candidate
            if self._should_query_llm(ood, civic, risk):
                decision = self._llm_decision(iid, obs, q_values, candidate, current, ood, civic, risk)
                llm_action = decision.final_phase
                self._stats["llm_triggers"] += 1
                self._stats[f"trust_{decision.policy_trust}"] += 1
                if llm_action != candidate:
                    self._stats["overrides"] += 1
            else:
                self._stats["trust_online" if not self._offline_loaded else "trust_offline"] += 1
            if self.agent_cfg.use_safety_projection:
                projected = self.projector.project(llm_action, range(self.num_phases), current, elapsed)
                if projected.adjusted:
                    self._stats["safety_adjustments"] += 1
                final = projected.action
            else:
                final = llm_action
            if elapsed < int(self.cfg.get("g_min", 10)) and final != current:
                self._stats["safety_violations"] += 1
            actions[iid] = final
            self._elapsed[iid] = elapsed + int(self.cfg.get("action_interval", 10)) if final == current else 0
            self._current[iid] = final
            self._stats["decisions"] += 1
        return actions

    def _q_values(self, obs: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None:
            return np.asarray(obs[0 : self.num_phases], dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.model(x).squeeze(0).cpu().numpy()
        return q.astype(np.float32)

    def _select_action(self, q_values: np.ndarray, obs: np.ndarray) -> int:
        if self.model is not None and random.random() < self.epsilon:
            return random.randrange(self.num_phases)
        if q_values.size:
            return int(np.argmax(q_values)) % self.num_phases
        return int(np.argmax(obs[0 : self.num_phases])) % self.num_phases

    def _ood(self, obs: np.ndarray) -> OODResult:
        if not self.agent_cfg.use_ood:
            return OODResult(score=0.0, is_ood=False, shift_type="disabled")
        result = self.ood_detector.score(obs)
        if result.is_ood:
            self._stats["ood_triggers"] += 1
        return result

    def _should_query_llm(self, ood: OODResult, civic: dict[str, Any], risk: float) -> bool:
        if not self.agent_cfg.llm_enabled:
            return False
        if ood.is_ood:
            return True
        if self.agent_cfg.use_civic and (civic.get("ev_eta") is not None or civic.get("event_pressure", 0.0) > 0.0 or civic.get("incident_score", 0.0) > 0.0):
            return True
        return risk >= float(self.cfg.get("tau_q", 3.0))

    def _llm_decision(
        self,
        intersection_id: str,
        obs: np.ndarray,
        q_values: np.ndarray,
        rl_action: int,
        current_phase: int,
        ood: OODResult,
        civic: dict[str, Any],
        risk: float,
    ) -> LLMDecision:
        t0 = time.perf_counter()
        payload = {
            "temporal_context": {"step_norm": float(obs[20]) if obs.size > 20 else 0.0},
            "ood": {"score": ood.score, "is_ood": ood.is_ood, "shift_type": ood.shift_type},
            "policy": {"rl_action": rl_action, "q_values": [float(v) for v in q_values], "risk": risk},
            "traffic_state": {"queue_features": [float(v) for v in obs[0 : self.num_phases]]},
            "rollouts": self.world_model.rollout(intersection_id, current_phase, horizon=3),
            "civic": civic,
            "legal_phases": list(range(self.num_phases)),
        }
        prompt = self.prompt_builder.build(payload)
        if self.llm_gateway is not None:
            try:
                decision = self.llm_gateway.query_many(
                    [
                        {
                            "prompt": prompt,
                            "legal_phases": list(range(self.num_phases)),
                            "intersection_id": intersection_id,
                            "rl_action": rl_action,
                        }
                    ]
                )[intersection_id]
                self._latencies.append(decision.latency_ms)
                self._kl_values.append(self._policy_kl(q_values, decision.final_phase))
                return decision
            except Exception as exc:
                self._llm_init_error = str(exc)
        final = rl_action
        trust = "online" if not self._offline_loaded else "offline"
        note = "accepted learned policy"
        if civic.get("ev_eta") is not None:
            final = 0
            trust = "conservative"
            note = "prioritized emergency corridor phase"
        elif ood.is_ood and risk >= float(self.cfg.get("tau_q", 3.0)):
            final = current_phase
            trust = "conservative"
            note = "held current phase under OOD risk"
        decision = LLMDecision(
            decision="override" if final != rl_action else "accept",
            final_phase=int(final) % self.num_phases,
            policy_trust=trust,
            causal_summary=note,
            safe_trajectory_hint=[int(final) % self.num_phases],
            stakeholder_resolution="local deterministic governance policy",
            adaptation_note=note,
            backend="local_policy_gateway",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            raw=prompt[:512],
            valid=True,
        )
        self._latencies.append(decision.latency_ms)
        self._kl_values.append(self._policy_kl(q_values, final))
        return decision

    def _policy_kl(self, q_values: np.ndarray, final: int) -> float:
        logits = q_values - np.max(q_values)
        probs = np.exp(logits) / max(1e-8, float(np.exp(logits).sum()))
        target = np.full_like(probs, 1e-6)
        target[final] = 1.0 - 1e-6 * (len(probs) - 1)
        return float(np.sum(target * (np.log(target) - np.log(np.clip(probs, 1e-6, 1.0)))))

    def observe(
        self,
        observations: dict[str, np.ndarray],
        actions: dict[str, int],
        rewards: dict[str, float],
        next_observations: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        for iid, obs in observations.items():
            if iid not in next_observations:
                continue
            self._replay.append(Transition(obs, int(actions.get(iid, 0)), float(rewards.get(iid, 0.0)), next_observations[iid], done))
        self._train_step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _train_step(self) -> None:
        if torch is None or F is None or self.model is None or self.target_model is None or self.optimizer is None:
            return
        if len(self._replay) < max(1, min(self.learning_start, self.batch_size)):
            return
        batch = random.sample(list(self._replay), min(self.batch_size, len(self._replay)))
        states = torch.tensor(np.asarray([t.state for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.asarray([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)
        q_sa = self.model(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * (1.0 - dones) * self.target_model(next_states).max(dim=1).values
        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._updates += 1
        self._last_loss = float(loss.item())
        if self._updates % max(1, self.target_update_freq) == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def metrics_snapshot(self, reset: bool = True) -> dict[str, float]:
        decisions = max(1, self._stats["decisions"])
        lat = np.asarray(self._latencies, dtype=float)
        metrics = {
            "ood_trigger_rate": self._stats["ood_triggers"] / decisions,
            "civic_llm_trigger_rate": self._stats["llm_triggers"] / decisions if self.agent_cfg.use_civic else 0.0,
            "llm_intervention_ratio": self._stats["llm_triggers"] / decisions,
            "override_rate": self._stats["overrides"] / decisions,
            "safety_adjustment_rate": self._stats["safety_adjustments"] / decisions,
            "safety_violation_count": float(self._stats["safety_violations"]),
            "policy_trust_online": self._stats["trust_online"] / decisions,
            "policy_trust_offline": self._stats["trust_offline"] / decisions,
            "policy_trust_conservative": self._stats["trust_conservative"] / decisions,
            "kl_divergence": float(np.mean(self._kl_values)) if self._kl_values else 0.0,
            "latency_mean_ms": float(np.mean(lat)) if lat.size else 0.0,
            "latency_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
            "latency_p99_ms": float(np.percentile(lat, 99)) if lat.size else 0.0,
            "train_loss": self._last_loss,
        }
        if reset:
            self._stats.clear()
            self._latencies.clear()
            self._kl_values.clear()
        return metrics

    def risk_score(self, values: dict[str, float]) -> float:
        return (
            self.cfg.get("lambda_u", 2.0) * values.get("uncertain", 0.0)
            + self.cfg.get("lambda_a", 1.5) * values.get("anomaly", 0.0)
            + self.cfg.get("lambda_q", 1.0) * values.get("queue", 0.0)
            + self.cfg.get("lambda_w", 0.5) * values.get("wait", 0.0)
            + self.cfg.get("lambda_s", 1.0) * values.get("spillback", 0.0)
            + self.cfg.get("lambda_OOD", 3.0) * values.get("ood_flag", 0.0) * values.get("ood_score", 0.0)
            + self.cfg.get("lambda_ev", 5.0) * values.get("ev_proximity", 0.0)
            + self.cfg.get("lambda_event", 1.2) * values.get("event_pressure", 0.0)
            + self.cfg.get("lambda_inc", 0.8) * values.get("incident_score", 0.0)
        )
