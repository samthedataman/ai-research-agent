"""Telegram bot — the user-facing interface for the AI Research Agent."""

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.collectors import COLLECTOR_REGISTRY
from src.config import settings
from src.graph import build_graph
from src.logging_config import get_logger, setup_logging
from src.storage import init_db, log_query

logger = get_logger(__name__)

# Compile the LangGraph workflow once at module level
workflow = build_graph()

# Per-user model selection (chat_id → {model, analysis_model})
user_models: dict[int, str] = {}
user_analysis_models: dict[int, str] = {}

# Sources available as /commands
SOURCE_COMMANDS = {
    "news": "Search Google News (free)",
    "weather": "Get weather from wttr.in (free)",
    "crypto": "Crypto data from CoinGecko (free)",
    "reddit": "Search Reddit posts (free)",
    "github": "Search GitHub repos (free)",
    "arxiv": "Search research papers (free)",
    "stocks": "Stock quotes from Yahoo Finance (free)",
    "wiki": "Search Wikipedia (free)",
    "ddg": "DuckDuckGo web search (free)",
    "ddgnews": "DuckDuckGo news search (free)",
    "serper": "Google search via Serper (API key)",
    "tmz": "TMZ celebrity news (free)",
    "cryptonews": "CryptoPanic crypto news (free)",
}


# ── Command handlers ─────────────────────────────────────────────────────────


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — welcome message."""
    sources_list = "\n".join(f"  /{cmd} — {desc}" for cmd, desc in SOURCE_COMMANDS.items())
    await update.message.reply_text(
        f"👋 *AI Research Agent*\n\n"
        f"I can search free data sources and summarize results with AI.\n\n"
        f"*Available commands:*\n{sources_list}\n\n"
        f"Or just send me a question and I'll pick the best source automatically!\n\n"
        f"*Settings:*\n  /model — Switch AI model\n\n"
        f"*Examples:*\n"
        f"  /news artificial intelligence\n"
        f"  /crypto trending\n"
        f"  /weather London\n"
        f"  What's happening in machine learning?",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — usage guide."""
    await update.message.reply_text(
        "*How to use this bot:*\n\n"
        "1️⃣ Use a command to search a specific source:\n"
        "   `/news AI agents` — search news\n"
        "   `/stocks AAPL,TSLA` — get stock quotes\n"
        "   `/arxiv transformers` — find papers\n\n"
        "2️⃣ Or just send a message and I'll figure it out:\n"
        "   `What's Bitcoin doing today?` → crypto\n"
        "   `Latest AI research` → arxiv/news\n\n"
        "3️⃣ I'll collect data, analyze it with AI, and send you a summary.",
        parse_mode="Markdown",
    )


async def _get_ollama_models() -> list[dict]:
    """Fetch available models from Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
    except Exception:
        return []


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model — show model picker with two sections."""
    models = await _get_ollama_models()
    if not models:
        await update.message.reply_text("No Ollama models found. Is Ollama running?")
        return

    chat_id = update.effective_chat.id
    cur_route = user_models.get(chat_id, settings.ollama_model)
    cur_analysis = user_analysis_models.get(chat_id, settings.ollama_analysis_model)

    buttons = []
    # Analysis model buttons (the big one)
    buttons.append([InlineKeyboardButton("── Analysis Model (quality) ──", callback_data="noop")])
    for m in models:
        name = m["name"]
        size = m.get("details", {}).get("parameter_size", "")
        check = "✅ " if name == cur_analysis else ""
        label = f"{check}{name} ({size})" if size else f"{check}{name}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"amodel:{name}")])

    # Routing model buttons (the fast one)
    buttons.append([InlineKeyboardButton("── Routing Model (speed) ──", callback_data="noop")])
    for m in models:
        name = m["name"]
        size = m.get("details", {}).get("parameter_size", "")
        check = "✅ " if name == cur_route else ""
        label = f"{check}{name} ({size})" if size else f"{check}{name}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"rmodel:{name}")])

    await update.message.reply_text(
        f"*Model Settings*\n"
        f"Analysis: `{cur_analysis}` _(generates the response)_\n"
        f"Routing: `{cur_route}` _(picks the data source)_",
        reply_markup=InlineKeyboardMarkup(buttons),
        parse_mode="Markdown",
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button press for model selection."""
    query = update.callback_query
    data = query.data

    if data == "noop":
        await query.answer()
        return

    await query.answer()
    chat_id = update.effective_chat.id

    if data.startswith("amodel:"):
        model_name = data.removeprefix("amodel:")
        user_analysis_models[chat_id] = model_name
        await query.edit_message_text(
            f"✅ *Analysis model* switched to `{model_name}`\n\n"
            f"This larger model will synthesize your results.",
            parse_mode="Markdown",
        )
        logger.info("analysis_model_switched", chat_id=chat_id, model=model_name)

    elif data.startswith("rmodel:"):
        model_name = data.removeprefix("rmodel:")
        user_models[chat_id] = model_name
        await query.edit_message_text(
            f"✅ *Routing model* switched to `{model_name}`\n\n"
            f"This fast model picks the best data source.",
            parse_mode="Markdown",
        )
        logger.info("route_model_switched", chat_id=chat_id, model=model_name)


async def source_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /<source> <query> commands (e.g., /news AI agents)."""
    # Extract source from the command (e.g., "/news" → "news")
    command = update.message.text.split()[0].lstrip("/").lower()
    # Map short command names to collector registry names
    source_map = {
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
    source = source_map.get(command, command)

    # Extract query (everything after the command)
    parts = update.message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await update.message.reply_text(
            f"Usage: `/{command} <query>`\nExample: `/{command} artificial intelligence`",
            parse_mode="Markdown",
        )
        return

    query = parts[1].strip()
    await _run_workflow(update, source=source, query=query)


async def free_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free text messages — LLM routes to the best source."""
    user_message = update.message.text.strip()
    if not user_message:
        return
    await _run_workflow(update, user_message=user_message)


# ── Workflow runner ──────────────────────────────────────────────────────────


async def _run_workflow(
    update: Update,
    source: str | None = None,
    query: str | None = None,
    user_message: str | None = None,
) -> None:
    """Run the LangGraph workflow and send the response."""
    # Show typing indicator
    await update.message.chat.send_action("typing")

    # Get user's selected models (or defaults)
    chat_id = update.effective_chat.id
    model = user_models.get(chat_id, "")
    analysis_model = user_analysis_models.get(chat_id, "")

    state = {
        "user_message": user_message or f"/{source} {query}",
        "source": source or "",
        "query": query or "",
        "items": [],
        "analysis": "",
        "response": "",
        "error": "",
        "model": model,
        "analysis_model": analysis_model,
    }

    try:
        result = await workflow.ainvoke(state)
        response_text = result.get("response", "Something went wrong. Please try again.")
    except Exception as e:
        logger.error("workflow_error", error=str(e))
        response_text = f"❌ An error occurred: {e}"

    # Send response (try Markdown, fall back to plain text)
    try:
        await update.message.reply_text(response_text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(response_text)

    # Log to database (fire and forget)
    try:
        await log_query(
            user_id=update.effective_user.id,
            source=source or result.get("source", "auto"),
            query=query or user_message or "",
            response=response_text[:2000],
        )
    except Exception as e:
        logger.warning("log_error", error=str(e))


# ── Bot startup ──────────────────────────────────────────────────────────────


async def start_bot() -> None:
    """Initialize and run the Telegram bot."""
    setup_logging()
    await init_db()

    if not settings.telegram_bot_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is required. Get one from @BotFather on Telegram."
        )

    app = Application.builder().token(settings.telegram_bot_token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^(amodel:|rmodel:|noop)"))

    # Register a handler for each source command
    for cmd in SOURCE_COMMANDS:
        app.add_handler(CommandHandler(cmd, source_command))

    # Free text (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, free_text_handler))

    logger.info("bot_started", sources=list(SOURCE_COMMANDS.keys()))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    # Keep running until stopped
    import asyncio
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
