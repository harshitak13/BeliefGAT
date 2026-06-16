from __future__ import annotations


BASELINES = ["webster", "actuated", "plain_dqn", "colight", "llmlight", "illm_tsc", "gat_dqn"]


class Evaluator:
    def summarize(self, metrics: dict, method: str) -> dict:
        out = dict(metrics)
        out["method"] = method
        return out
