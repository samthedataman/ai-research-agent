import httpx
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


class OpenRouterClient:
    """OpenRouter client for cloud LLM inference."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        default_model: str = "deepseek/deepseek-chat",
    ):
        self.api_key = api_key
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

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        logger.info("openrouter_request", model=model_name, message_count=len(messages))
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            },
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
