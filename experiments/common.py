from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.baseline_agents import build_baseline_agent
from agents.beliefgat_citygat_agent import BeliefGATCityGATAgent
from agents.safegat_agent import ABLATION_CONFIGS, AgentConfig
from civic.civic_context_module import CivicContextModule
from env_wrappers.cityflow_env import CityFlowMultiAgentEnv
from env_wrappers.sumo_env import SUMOMultiAgentEnv
from evaluation.metrics import MetricsTracker
from utils.config import apply_cli_overrides, load_yaml


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def build_env(cfg: dict[str, Any], seed: int = 42, mock: bool = False):
    if cfg["env_type"] == "sumo":
        return SUMOMultiAgentEnv(cfg, seed=seed, mock=mock)
    if cfg["env_type"] == "cityflow":
        return CityFlowMultiAgentEnv(cfg, seed=seed, mock=mock)
    raise ValueError(f"Unknown env_type: {cfg['env_type']}")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, flow_file=getattr(args, "flow_file", None), steps_per_episode=getattr(args, "steps_per_episode", None))
    episodes = args.episodes or int(cfg.get("train_episodes", 1))
    results_dir = Path(args.results_dir or f"results/{cfg['env_name']}/{getattr(args, 'llm_backend', 'no_llm')}")
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg["llm_backend"] = getattr(args, "llm_backend", None)
    cfg["remote_llm"] = bool(getattr(args, "remote_llm", False))
    cfg["llm_log_path"] = str(results_dir / "decisions.jsonl")
    cfg["api_keys_path"] = getattr(args, "api_keys", "configs/api_keys.yaml")

    ablation_name = getattr(args, "ablation", "V5_full") or "V5_full"
    ablation = dict(ABLATION_CONFIGS.get(ablation_name, ABLATION_CONFIGS["V5_full"]))
    if hasattr(args, "llm_enabled"):
        ablation["llm_enabled"] = str_to_bool(args.llm_enabled)
    baseline = getattr(args, "baseline", None)
    if baseline:
        ablation.update(use_ood=False, use_civic=False, llm_enabled=False)

    agent_cfg = AgentConfig(**ablation)
    env = build_env(cfg, mock=getattr(args, "mock", False))
    agent = build_baseline_agent(baseline, cfg) if baseline else BeliefGATCityGATAgent(cfg, agent_cfg)
    if getattr(args, "load_offline", None) and hasattr(agent, "load_checkpoint"):
        agent.load_checkpoint(args.load_offline)
    civic_module = CivicContextModule(enabled=agent_cfg.use_civic and not bool(baseline))
    tracker = MetricsTracker()

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            env_context = env.get_civic_context() if hasattr(env, "get_civic_context") else {}
            civic_context = civic_module.get_context(env.get_intersection_ids())
            for iid, values in env_context.items():
                civic_context.setdefault(iid, {}).update(values)
            actions = agent.act(obs, civic_context)
            next_obs, rewards, done, info = env.step(actions)
            if hasattr(agent, "observe"):
                agent.observe(obs, actions, rewards, next_obs, done)
            obs = next_obs
        metrics = env.get_episode_metrics()
        agent_metrics = agent.metrics_snapshot(reset=True) if hasattr(agent, "metrics_snapshot") else {}
        tracker.add(
            {
                "episode": ep + 1,
                **metrics,
                **agent_metrics,
            }
        )

    tracker.save_csv(results_dir / "metrics.csv")
    tracker.save_training_plot(results_dir / "training_9panel.png")
    summary = {
        "env_name": cfg["env_name"],
        "config": args.config,
        "episodes": episodes,
        "llm_backend": getattr(args, "llm_backend", None),
        "ablation": ablation_name,
        "baseline": baseline,
        "results_dir": str(results_dir),
        "final_metrics": tracker.rows[-1] if tracker.rows else {},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()
    print(json.dumps(summary, indent=2))
    return summary


def experiment_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--llm_backend", default="llama_31_8b")
    parser.add_argument("--load_offline", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps_per_episode", type=int, default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--ablation", choices=list(ABLATION_CONFIGS), default="V5_full")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--flow_file", default=None)
    parser.add_argument("--llm_enabled", default="True")
    parser.add_argument("--remote_llm", action="store_true", help="Call the configured external LLM backend instead of the local policy gateway.")
    parser.add_argument("--api_keys", default="configs/api_keys.yaml")
    parser.add_argument("--mock", action="store_true", help="Use the lightweight built-in simulator facade.")
    return parser
