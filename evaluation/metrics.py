from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "episode",
    "att",
    "queue",
    "delay",
    "throughput",
    "ood_trigger_rate",
    "policy_trust_online",
    "policy_trust_offline",
    "policy_trust_conservative",
    "kl_divergence",
    "ev_clearance_time",
    "post_event_clearance_time",
    "civic_llm_trigger_rate",
    "llm_intervention_ratio",
    "override_rate",
    "safety_adjustment_rate",
    "safety_violation_count",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
]


class MetricsTracker:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        complete = {field: row.get(field, 0) for field in METRIC_FIELDS}
        self.rows.append(complete)

    def save_csv(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def save_training_plot(self, path: str | Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        names = ["att", "queue", "delay", "throughput", "ood_trigger_rate", "llm_intervention_ratio", "override_rate", "latency_p95_ms", "safety_adjustment_rate"]
        for ax, name in zip(axes.ravel(), names):
            ax.plot([r["episode"] for r in self.rows], [r[name] for r in self.rows])
            ax.set_title(name)
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
