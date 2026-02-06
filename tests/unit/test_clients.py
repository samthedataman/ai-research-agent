import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.clients.ollama_client import OllamaClient
from src.clients.openrouter_client import OpenRouterClient


class TestOllamaClient:
    def test_init_defaults(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.default_model == "llama3.1:8b"

    def test_init_custom(self):
        client = OllamaClient(base_url="http://custom:11434", default_model="mistral")
        assert client.base_url == "http://custom:11434"
        assert client.default_model == "mistral"

    def test_get_text(self):
        client = OllamaClient()
        response = {"message": {"content": "Hello world"}}
        assert client.get_text(response) == "Hello world"

    @pytest.mark.asyncio
    async def test_complete(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "test response"}}
        mock_response.raise_for_status = MagicMock()

        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        result = await client.complete([{"role": "user", "content": "hello"}])
        assert result["message"]["content"] == "test response"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.status_code = 200

        client.client = AsyncMock()
        client.client.get = AsyncMock(return_value=mock_response)

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = OllamaClient()
        client.client = AsyncMock()
        client.client.get = AsyncMock(side_effect=Exception("connection error"))

        assert await client.health_check() is False


class TestOpenRouterClient:
    def test_init(self):
        client = OpenRouterClient(api_key="test-key")
        assert client.default_model == "deepseek/deepseek-chat"

    def test_get_text(self):
        client = OpenRouterClient(api_key="test-key")
        response = {"choices": [{"message": {"content": "Hello"}}]}
        assert client.get_text(response) == "Hello"

    @pytest.mark.asyncio
    async def test_complete(self):
        client = OpenRouterClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        client.client = AsyncMock()
        client.client.post = AsyncMock(return_value=mock_response)

        result = await client.complete([{"role": "user", "content": "hello"}])
        assert result["choices"][0]["message"]["content"] == "response"
