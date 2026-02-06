import httpx
from typing import Any, Protocol

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]: ...

    def get_text(self, response: dict[str, Any]) -> str: ...

    async def health_check(self) -> bool: ...

    async def close(self) -> None: ...


class OllamaClient:
    """Ollama client for local LLM inference."""

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.1:8b"):
        self.base_url = base_url
        self.default_model = default_model
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def complete(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.7) -> dict[str, Any]:
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

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()


class OpenRouterClient:
    """OpenRouter client for cloud LLM inference."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, default_model: str = "deepseek/deepseek-chat"):
        self.default_model = default_model
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://ai-research-agent.app",
                "X-Title": "AI-Research-Agent",
            },
            timeout=60.0,
        )

    async def complete(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.7) -> dict[str, Any]:
        model_name = model or self.default_model
        logger.info("openrouter_request", model=model_name, message_count=len(messages))
        response = await self.client.post(
            "/chat/completions",
            json={"model": model_name, "messages": messages, "temperature": temperature},
        )
        response.raise_for_status()
        return response.json()

    def get_text(self, response: dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/models")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()


def get_llm_client() -> LLMClient:
    """Factory: get the configured LLM client."""
    if settings.llm_provider == "ollama":
        return OllamaClient(base_url=settings.ollama_base_url, default_model=settings.ollama_model)
    elif settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when using openrouter provider")
        return OpenRouterClient(api_key=settings.openrouter_api_key, default_model=settings.openrouter_model)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
