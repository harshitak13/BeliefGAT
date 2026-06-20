from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
import yaml


GROQ_MODEL_IDS = {
    "llama_31_8b": "llama-3.1-8b-instant",
    "llama_33_70b": "llama-3.3-70b-versatile",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "llama_4_scout_17b": "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen3_32b": "qwen/qwen3-32b",
}


def _load_groq_config(api_keys_path: str = "configs/api_keys.yaml") -> dict:
    p = Path(api_keys_path)
    if not p.exists():
        raise FileNotFoundError(
            f"API keys file not found: {api_keys_path}. Copy configs/api_keys.yaml.example "
            "to configs/api_keys.yaml and set groq.api_key."
        )
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    groq_cfg = cfg.get("groq", {})
    key = groq_cfg.get("api_key", "")
    if not key or key.startswith("REPLACE_WITH"):
        raise ValueError("Groq API key not set in configs/api_keys.yaml.")
    return groq_cfg


class GroqBackend:
    def __init__(self, model_alias: str, api_keys_path: str = "configs/api_keys.yaml"):
        if model_alias not in GROQ_MODEL_IDS:
            raise ValueError(f"Unknown Groq model alias: {model_alias}")
        cfg = _load_groq_config(api_keys_path)
        self.model_alias = model_alias
        self.model_id = GROQ_MODEL_IDS[model_alias]
        self.api_key = cfg["api_key"]
        self.base_url = cfg.get("base_url", "https://api.groq.com/openai/v1").rstrip("/")
        self.temperature = cfg.get("default_temperature", 0.1)
        self.max_tokens = cfg.get("default_max_tokens", 600)

    async def query_async(self, session: aiohttp.ClientSession, prompt: str, timeout_ms: int = 500) -> str:
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=timeout) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"Groq API error {resp.status}: {body[:240]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    def query_sync(self, prompt: str, timeout_ms: int = 500) -> str:
        async def _run() -> str:
            async with aiohttp.ClientSession() as session:
                return await self.query_async(session, prompt, timeout_ms)

        return asyncio.run(_run())
