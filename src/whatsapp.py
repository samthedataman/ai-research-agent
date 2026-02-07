"""WhatsApp bot — webhook-based interface for the AI Research Agent.

Supports two backends:
1. Meta Cloud API — for 1-on-1 DMs (webhook-based)
2. Green API (green-api.com) — for group messaging + DMs (free tier)
"""

import re

import httpx

from src.config import settings
from src.graph import build_graph
from src.logging_config import get_logger
from src.storage import add_wa_subscriber, log_query, remove_wa_subscriber

logger = get_logger(__name__)

workflow = build_graph()

WHATSAPP_API_URL = "https://graph.facebook.com/v21.0"

# Map short command names → collector registry names (mirrors bot.py)
SOURCE_MAP = {
    "news": "news",
    "weather": "weather",
    "crypto": "crypto",
    "reddit": "reddit",
    "github": "github",
    "arxiv": "arxiv",
    "stocks": "stocks",
    "wiki": "wikipedia",
    "ddg": "ddg",
    "ddgnews": "ddg_news",
    "serper": "serper",
    "tmz": "tmz",
    "cryptonews": "cryptonews",
}


# ── Green API (group messaging) ─────────────────────────────────────────────


def _greenapi_base_url() -> str:
    """Build Green API base URL for this instance."""
    return f"https://api.green-api.com/waInstance{settings.greenapi_instance_id}"


async def send_greenapi_message(chat_id: str, text: str) -> None:
    """Send a message via Green API (works with groups and DMs).

    chat_id formats:
    - Group: "120363XXXXX@g.us"
    - DM: "1234567890@c.us"
    """
    url = f"{_greenapi_base_url()}/sendMessage/{settings.greenapi_api_token}"
    # Green API has ~4096 char limit per message
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            payload = {"chatId": chat_id, "message": chunk}
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    logger.info("greenapi_sent", chat_id=chat_id)
                else:
                    logger.error("greenapi_send_error", status=resp.status_code, body=resp.text)
            except Exception as e:
                logger.error("greenapi_send_exception", error=str(e))


async def send_to_group(text: str) -> None:
    """Send a message to the configured WhatsApp group via Green API."""
    if not settings.greenapi_group_id:
        logger.warning("no_greenapi_group_id")
        return
    await send_greenapi_message(settings.greenapi_group_id, text)


async def get_greenapi_groups() -> list[dict]:
    """List all groups the Green API instance is part of."""
    url = f"{_greenapi_base_url()}/getContacts/{settings.greenapi_api_token}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                contacts = resp.json()
                return [c for c in contacts if c.get("id", "").endswith("@g.us")]
            return []
        except Exception:
            return []


# ── Meta Cloud API (1-on-1 DMs) ─────────────────────────────────────────────


async def send_whatsapp_message(to: str, text: str) -> None:
    """Send a text message — routes to Green API or Meta Cloud API."""
    # If Green API is configured, prefer it (works for both DMs and groups)
    if settings.greenapi_instance_id and settings.greenapi_api_token:
        # Convert phone number to Green API chat ID format
        chat_id = f"{to}@c.us" if not to.endswith(("@c.us", "@g.us")) else to
        await send_greenapi_message(chat_id, text)
        return

    # Fall back to Meta Cloud API
    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    chunks = [text[i : i + 4096] for i in range(0, len(text), 4096)]
    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            payload = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": chunk},
            }
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("whatsapp_send_error", status=resp.status_code, body=resp.text)


async def send_template_message(to: str, template_name: str, language: str = "en_US") -> None:
    """Send a pre-approved template message (Meta Cloud API only)."""
    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {"name": template_name, "language": {"code": language}},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.error("whatsapp_template_error", status=resp.status_code, body=resp.text)


# ── Inbound: process incoming webhook messages ───────────────────────────────


async def handle_incoming_message(phone_number: str, message_text: str) -> None:
    """Process an incoming WhatsApp message through the LangGraph pipeline."""
    text = message_text.strip()

    # Handle subscribe/unsubscribe
    if text.lower() in ("/subscribe", "subscribe"):
        await add_wa_subscriber(phone_number)
        await send_whatsapp_message(
            phone_number,
            "You're subscribed to daily briefings!\n"
            "You'll receive a morning update with news, crypto, and stock highlights.\n\n"
            "Send /unsubscribe to stop.",
        )
        return

    if text.lower() in ("/unsubscribe", "unsubscribe"):
        await remove_wa_subscriber(phone_number)
        await send_whatsapp_message(phone_number, "You've been unsubscribed from daily briefings.")
        return

    if text.lower() in ("/help", "help", "/start", "hi", "hello"):
        await send_whatsapp_message(phone_number, _help_text())
        return

    # Parse command (e.g., "/news AI agents") or use auto-routing
    source, query = _parse_command(text)

    state = {
        "user_message": text,
        "source": source or "",
        "query": query or "",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
        "model": "",
        "analysis_model": "",
    }

    try:
        result = await workflow.ainvoke(state)
        response_text = _clean_for_whatsapp(result.get("response", "Something went wrong."))
    except Exception as e:
        logger.error("whatsapp_workflow_error", error=str(e))
        response_text = f"An error occurred: {e}"

    await send_whatsapp_message(phone_number, response_text)

    # Log to database
    try:
        await log_query(
            user_id=hash(phone_number) % (2**31),
            source=source or result.get("source", "auto"),
            query=query or text,
            response=response_text[:2000],
        )
    except Exception as e:
        logger.warning("wa_log_error", error=str(e))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_command(text: str) -> tuple[str | None, str | None]:
    """Parse /source query from message. Returns (source, query) or (None, None)."""
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        if cmd in SOURCE_MAP:
            query = parts[1].strip() if len(parts) > 1 else None
            return SOURCE_MAP[cmd], query
    return None, None


def _clean_for_whatsapp(text: str) -> str:
    """Convert Telegram-flavored Markdown to WhatsApp-friendly text.

    WhatsApp supports *bold*, _italic_, ~strikethrough~, ```code``` natively.
    We just strip Telegram-specific artifacts like model tags.
    """
    text = re.sub(r"`\[.*?\]`", "", text)  # Remove [model_name] tags
    return text.strip()


def _help_text() -> str:
    return (
        "*AI Research Agent*\n\n"
        "I can search free data sources and summarize results with AI.\n\n"
        "*Commands:*\n"
        "/news <query> - Search news\n"
        "/crypto <query> - Crypto data\n"
        "/stocks <tickers> - Stock quotes\n"
        "/weather <city> - Weather\n"
        "/reddit <query> - Reddit posts\n"
        "/arxiv <query> - Research papers\n"
        "/github <query> - GitHub repos\n"
        "/wiki <query> - Wikipedia\n"
        "/ddg <query> - Web search\n\n"
        "Or just send a question and I'll figure it out!\n\n"
        "/subscribe - Daily morning briefing\n"
        "/unsubscribe - Stop daily briefing"
    )
