from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable

import aiohttp

from llm.output_parser import LLMDecision, OutputParser


GROQ_ALIASES = {"llama_31_8b", "llama_33_70b", "gpt_oss_120b", "gpt_oss_20b", "llama_4_scout_17b", "qwen3_32b"}
OPENAI_ALIASES = {"gpt_4o"}


class LLMGateway:
    def __init__(
        self,
        backend: str,
        timeout_ms: int = 500,
        api_keys_path: str = "configs/api_keys.yaml",
        log_path: str = "results/decisions.jsonl",
    ):
        self.backend = backend
        self.timeout_ms = timeout_ms
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if backend in GROQ_ALIASES:
            from llm.backends.groq_backend import GroqBackend

            self._backend = GroqBackend(backend, api_keys_path)
        elif backend in OPENAI_ALIASES:
            from llm.backends.openai_backend import OpenAIBackend

            self._backend = OpenAIBackend(api_keys_path)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

    async def _query_single_async(self, session: aiohttp.ClientSession, prompt: str, legal_phases: Iterable[int], i_id: str, rl_action: int) -> LLMDecision:
        t0 = time.time()
        try:
            raw = await self._backend.query_async(session, prompt, self.timeout_ms)
            latency = (time.time() - t0) * 1000
            decision = OutputParser.parse(raw, legal_phases, self.backend, latency, rl_action)
        except asyncio.TimeoutError:
            latency = (time.time() - t0) * 1000
            decision = OutputParser.fallback(self.backend, latency, rl_action, reason="timeout")
        except Exception as exc:
            latency = (time.time() - t0) * 1000
            decision = OutputParser.fallback(self.backend, latency, rl_action, reason=f"backend_error:{exc}")
        self._log(i_id, prompt, decision)
        return decision

    async def query_many_async(self, requests: list[dict]) -> dict[str, LLMDecision]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._query_single_async(
                    session,
                    req["prompt"],
                    req.get("legal_phases", [0, 1, 2, 3]),
                    req["intersection_id"],
                    req.get("rl_action", 0),
                )
                for req in requests
            ]
            decisions = await asyncio.gather(*tasks)
        return {req["intersection_id"]: dec for req, dec in zip(requests, decisions)}

    def query_many(self, requests: list[dict]) -> dict[str, LLMDecision]:
        return asyncio.run(self.query_many_async(requests))

    def _log(self, intersection_id: str, prompt: str, decision: LLMDecision) -> None:
        row = {
            "intersection_id": intersection_id,
            "backend": decision.backend,
            "latency_ms": decision.latency_ms,
            "final_phase": decision.final_phase,
            "valid": decision.valid,
            "causal_summary": decision.causal_summary,
            "adaptation_note": decision.adaptation_note,
            "prompt_chars": len(prompt),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
