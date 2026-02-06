from typing import Any, Protocol

from src.config import settings


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


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "ollama":
        from src.clients.ollama_client import OllamaClient

        return OllamaClient(
            base_url=settings.ollama_base_url,
            default_model=settings.ollama_model,
        )
    elif settings.llm_provider == "openrouter":
        from src.clients.openrouter_client import OpenRouterClient

        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when using openrouter provider")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            default_model=settings.openrouter_model,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
