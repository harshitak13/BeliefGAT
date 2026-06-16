from __future__ import annotations

import aiohttp
import yaml
from pathlib import Path


def _load_openai_config(path: str = "configs/api_keys.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"API keys file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("openai", {})


class OpenAIBackend:
    def __init__(self, api_keys_path: str = "configs/api_keys.yaml"):
        cfg = _load_openai_config(api_keys_path)
        self.api_key = cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = cfg.get("default_model", "gpt-4o")
        self.temperature = cfg.get("default_temperature", 0.1)
        self.max_tokens = cfg.get("default_max_tokens", 600)

    async def query_async(self, session: aiohttp.ClientSession, prompt: str, timeout_ms: int = 500) -> str:
        if not self.api_key or self.api_key.startswith("REPLACE_WITH"):
            raise ValueError("OpenAI API key not set in configs/api_keys.yaml.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
        async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=timeout) as resp:
            body = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"OpenAI API error {resp.status}: {body[:240]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
