"""Thin HTTP client wrapper for the Ollama API."""

import httpx
from rich.console import Console

console = Console()


class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable."""


class OllamaTimeoutError(Exception):
    """Raised when an Ollama request exceeds the timeout."""


class OllamaClient:
    """Minimal client for Ollama's /api/chat and /api/tags endpoints."""

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, user_message: str, timeout: float = 30.0) -> str:
        """
        Send a chat request to Ollama.

        Returns the assistant message content string.
        Raises OllamaConnectionError or OllamaTimeoutError on failure.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {timeout}s") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.host}") from exc

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and responsive."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def ensure_model_pulled(self) -> None:
        """Pull the configured model if it is not already present."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.host}/api/tags")
                resp.raise_for_status()
                tags = resp.json()
                existing = [m.get("name", "") for m in tags.get("models", [])]

            if not any(self.model in name for name in existing):
                console.print(
                    f"[Ollama] Pulling model '{self.model}' — this may take a few minutes...",
                    style="yellow",
                )
                with httpx.Client(timeout=600.0) as client:
                    pull_resp = client.post(
                        f"{self.host}/api/pull",
                        json={"name": self.model},
                    )
                    pull_resp.raise_for_status()
                console.print(f"[Ollama] Model '{self.model}' ready.", style="green")
            else:
                console.print(f"[Ollama] Model '{self.model}' already present.", style="dim green")
        except Exception as exc:
            console.print(f"[Ollama] Could not ensure model: {exc}", style="yellow")
