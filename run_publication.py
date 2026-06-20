from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agents.safegat_agent import ABLATION_CONFIGS


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GROQ_LLMS = ["llama_31_8b", "llama_33_70b", "gpt_oss_120b", "gpt_oss_20b", "qwen3_32b"]


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    config: str
    runner: str
    default_episodes: int
    simulator: str


NETWORKS = {
    "sumo_4x4": NetworkSpec("sumo_4x4", "configs/sumo_4x4.yaml", "experiments/run_sumo_4x4.py", 200, "sumo"),
    "sumo_7x28": NetworkSpec("sumo_7x28", "configs/sumo_7x28.yaml", "experiments/run_sumo_7x28.py", 300, "sumo"),
    "jinan": NetworkSpec("jinan", "configs/jinan.yaml", "experiments/run_jinan.py", 300, "cityflow"),
    "hangzhou": NetworkSpec("hangzhou", "configs/hangzhou.yaml", "experiments/run_hangzhou.py", 300, "cityflow"),
    "new_york": NetworkSpec("new_york", "configs/new_york.yaml", "experiments/run_new_york.py", 400, "cityflow"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_asset(path_value: str | None, label: str) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([REPO_ROOT / path, REPO_ROOT / "envs" / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} not found: {path_value}. Expected the real environment asset under {REPO_ROOT / 'envs'}.")


def validate_environment_assets(spec: NetworkSpec) -> None:
    cfg = load_yaml(REPO_ROOT / spec.config)
    if spec.simulator == "sumo":
        resolve_asset(cfg.get("sumo_cfg"), "SUMO config")
        resolve_asset(cfg.get("net_file"), "SUMO network")
        resolve_asset(cfg.get("rou_file"), "SUMO route file")
    else:
        resolve_asset(cfg.get("roadnet_file"), "CityFlow roadnet")
        resolve_asset(cfg.get("flow_file"), "CityFlow flow")


def validate_groq_key(api_keys_path: Path, enabled: bool) -> None:
    if not enabled:
        return
    if not api_keys_path.exists():
        raise FileNotFoundError(f"Groq API key file missing: {api_keys_path}. Copy configs/api_keys.yaml.example to configs/api_keys.yaml.")
    cfg = load_yaml(api_keys_path)
    key = str(cfg.get("groq", {}).get("api_key", ""))
    if not key or key.startswith("REPLACE_WITH"):
        raise ValueError(f"Set groq.api_key in {api_keys_path} before running remote LLM publication experiments.")


def run_command(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n[BeliefGAT] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def should_run_ablation(ablation_name: str, args: argparse.Namespace) -> bool:
    llm_enabled = bool(ABLATION_CONFIGS[ablation_name].get("llm_enabled", False))
    if llm_enabled and args.skip_llm:
        return False
    if not llm_enabled and args.skip_no_llm:
        return False
    return True


def run_ablation(
    spec: NetworkSpec,
    common: list[str],
    results_root: Path,
    ablation_name: str,
    backend: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    llm_enabled = bool(ABLATION_CONFIGS[ablation_name].get("llm_enabled", False))
    run_dir = results_root / "ablations" / ablation_name
    if llm_enabled:
        run_dir = run_dir / str(backend)
    else:
        run_dir = run_dir / "no_llm"

    cmd = [
        sys.executable,
        spec.runner,
        *common,
        "--results_dir",
        str(run_dir),
        "--ablation",
        ablation_name,
        "--llm_enabled",
        str(llm_enabled),
    ]
    if llm_enabled:
        cmd.extend(["--llm_backend", str(backend)])
        if not args.local_llm:
            cmd.append("--remote_llm")

    run_command(cmd, args.dry_run)
    return read_json(run_dir / "summary.json")


def run_publication(args: argparse.Namespace) -> dict[str, Any]:
    spec = NETWORKS[args.network]
    validate_environment_assets(spec)
    validate_groq_key(REPO_ROOT / args.api_keys, enabled=not args.local_llm and not args.skip_llm)

    episodes = args.episodes or spec.default_episodes
    results_root = (REPO_ROOT / args.results_root / spec.name).resolve()
    checkpoint_dir = (REPO_ROOT / args.checkpoint_root / spec.name).resolve()
    checkpoint_path = checkpoint_dir / "iql.pt"
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {
        "network": spec.name,
        "config": spec.config,
        "simulator": spec.simulator,
        "episodes": episodes,
        "steps_per_episode": args.steps_per_episode,
        "checkpoint": str(checkpoint_path),
        "llm_backends": [] if args.skip_llm else args.llm_backends,
        "ablations": args.ablations,
        "remote_llm": not args.local_llm,
        "mock": args.mock,
        "runs": {},
    }

    if args.force_pretrain or not checkpoint_path.exists():
        run_command(
            [
                sys.executable,
                "training/offline_pretrain.py",
                "--config",
                spec.config,
                "--epochs",
                str(args.offline_epochs),
                "--batch_size",
                str(args.batch_size),
                "--save_dir",
                str(checkpoint_dir),
            ],
            args.dry_run,
        )
    else:
        print(f"[BeliefGAT] Reusing offline checkpoint: {checkpoint_path}", flush=True)

    common = [
        "--config",
        spec.config,
        "--load_offline",
        str(checkpoint_path),
        "--episodes",
        str(episodes),
        "--api_keys",
        args.api_keys,
    ]
    if args.steps_per_episode is not None:
        common.extend(["--steps_per_episode", str(args.steps_per_episode)])
    if args.mock:
        common.append("--mock")

    for ablation_name in args.ablations:
        if not should_run_ablation(ablation_name, args):
            print(f"[BeliefGAT] Skipping ablation due to skip flags: {ablation_name}", flush=True)
            continue
        llm_enabled = bool(ABLATION_CONFIGS[ablation_name].get("llm_enabled", False))
        stages["runs"][ablation_name] = {}
        if llm_enabled:
            for backend in args.llm_backends:
                stages["runs"][ablation_name][backend] = run_ablation(
                    spec,
                    common,
                    results_root,
                    ablation_name,
                    backend,
                    args,
                )
        else:
            stages["runs"][ablation_name]["no_llm"] = run_ablation(
                spec,
                common,
                results_root,
                ablation_name,
                None,
                args,
            )

    summary_path = results_root / "publication_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    print(f"\n[BeliefGAT] Publication summary: {summary_path}", flush=True)
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BeliefGAT publication workflow for one traffic network.")
    parser.add_argument("--network", choices=sorted(NETWORKS), required=True)
    parser.add_argument("--episodes", type=int, default=None, help="Override publication episode count.")
    parser.add_argument("--steps_per_episode", type=int, default=None, help="Override simulator horizon for short tests.")
    parser.add_argument("--offline_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--results_root", default="results/publication")
    parser.add_argument("--checkpoint_root", default="checkpoints/publication")
    parser.add_argument("--api_keys", default="configs/api_keys.yaml")
    parser.add_argument("--llm_backends", nargs="+", default=DEFAULT_GROQ_LLMS)
    parser.add_argument("--ablations", nargs="+", choices=list(ABLATION_CONFIGS), default=list(ABLATION_CONFIGS))
    parser.add_argument("--local_llm", action="store_true", help="Use deterministic local LLM policy gateway instead of Groq API calls.")
    parser.add_argument("--skip_no_llm", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--force_pretrain", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use wrapper mock mode for wiring checks only.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_publication(parse_args())
