import httpx
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Ollama client for local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b",
    ):
        self.base_url = base_url
        self.default_model = default_model
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.info("ollama_request", model=payload["model"], message_count=len(messages))
        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    def get_text(self, response: dict[str, Any]) -> str:
        return response["message"]["content"]

    async def get_embeddings(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            resp = await self.client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
