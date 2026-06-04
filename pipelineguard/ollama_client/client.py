"""Thin HTTP client wrapper for the Ollama API."""

import httpx
from rich.console import Console

console = Console()


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable."""


class OllamaTimeoutError(Exception):
    """Raised when an Ollama request exceeds the timeout."""


class OllamaClient:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
        )

    def generate(self, system_prompt: str, user_message: str, timeout: float = 60.0) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.1,
            },
        }
        try:
            resp = self._client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {timeout}s") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.host}") from exc

    def is_available(self) -> bool:
        try:
            resp = self._client.get(f"{self.host}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_model_pulled(self) -> None:
        try:
            resp = self._client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            tags = resp.json()
            existing = [m.get("name", "") for m in tags.get("models", [])]

            if not any(self.model in name for name in existing):
                console.print(f"[Ollama] Pulling model '{self.model}'...", style="yellow")
                pull_resp = httpx.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model},
                    timeout=600.0,
                )
                pull_resp.raise_for_status()
                console.print(f"[Ollama] Model '{self.model}' ready.", style="green")
            else:
                console.print(f"[Ollama] Model '{self.model}' already present.", style="dim green")
        except Exception as exc:
            console.print(f"[Ollama] Could not ensure model: {exc}", style="yellow")