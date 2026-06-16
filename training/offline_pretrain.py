from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.iql import OfflineIQLTrainer
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline IQL pretraining for BeliefGAT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--collect_if_missing", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stats = OfflineIQLTrainer().train_placeholder(args.epochs)
    payload = {"config": args.config, "env_name": cfg.get("env_name"), "batch_size": args.batch_size, **stats}
    (save_dir / "iql.pt").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (save_dir / "pretrain_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved offline checkpoint placeholder to {save_dir / 'iql.pt'}")


if __name__ == "__main__":
    main()
