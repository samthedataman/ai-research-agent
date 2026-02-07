# AI Research Agent

A multi-platform AI research assistant that searches 14+ free data sources and delivers AI-powered summaries via **Telegram bot**, **WhatsApp group messages**, **REST API**, and **scheduled daily briefings**.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) for self-healing workflows, [Ollama](https://ollama.ai) for local inference, and [OpenRouter](https://openrouter.ai) for cloud deployment.

## Features

- **14 free data sources** — news, crypto, stocks, weather, Reddit, GitHub, arXiv, Wikipedia, DuckDuckGo, TMZ, and more
- **Self-healing pipeline** — if a source fails or returns empty, it automatically retries with fallback sources (max 2 retries)
- **Telegram bot** — full command interface with inline model switching
- **WhatsApp group messaging** — daily briefings sent to your WhatsApp group via [Green API](https://green-api.com)
- **REST API** — query any source programmatically
- **Daily scheduler** — automated morning briefings with configurable sources and time
- **Dual LLM** — fast model for routing, large model for analysis/synthesis
- **Deploy free** on [Render](https://render.com) (web service + worker)

## Architecture

```
                     ┌──────────────────────┐
                     │   LangGraph Pipeline  │
                     │                      │
User ──► Telegram ──►│ route → collect ─┐   │
     ──► WhatsApp ──►│                  ├──►│ analyze → respond
     ──► REST API ──►│ retry (fallback) ┘   │
     ──► Scheduler ──►│                      │
                     └──────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  14+ Data Sources  │
                    │  (all free)        │
                    └───────────────────┘
```

## Quick Start (Local)

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running

### 1. Clone and install

```bash
git clone https://github.com/samthedataman/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
```

### 2. Pull Ollama models

```bash
# Fast model for routing (picks the best data source)
ollama pull llama3.2:latest

# Larger model for analysis (synthesizes results)
ollama pull llama3:latest
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings (see Configuration section below)
```

### 4. Run the Telegram bot

```bash
python main.py
# Or use the auto-start script:
./start.sh
```

### 5. Run the REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Configuration

All configuration is via environment variables (`.env` file).

### Required

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Get from [@BotFather](https://t.me/BotFather) on Telegram |

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `openrouter` (cloud) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Fast model for routing |
| `OLLAMA_ANALYSIS_MODEL` | `llama3:latest` | Large model for synthesis |
| `OPENROUTER_API_KEY` | | Get free at [openrouter.ai](https://openrouter.ai) |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat` | Cloud model to use |

### WhatsApp Group (via Green API)

| Variable | Description |
|---|---|
| `GREENAPI_INSTANCE_ID` | Get from [green-api.com](https://green-api.com) (free tier) |
| `GREENAPI_API_TOKEN` | Green API authentication token |
| `GREENAPI_GROUP_ID` | WhatsApp group chat ID (e.g., `120363XXXXX@g.us`) |

### WhatsApp DMs (via Meta Cloud API)

| Variable | Description |
|---|---|
| `WHATSAPP_ACCESS_TOKEN` | Get from [developers.facebook.com](https://developers.facebook.com) |
| `WHATSAPP_PHONE_NUMBER_ID` | WhatsApp Business phone number ID |
| `WHATSAPP_VERIFY_TOKEN` | Any secret string for webhook verification |
| `WHATSAPP_APP_SECRET` | Meta app secret (optional, for signature verification) |

### Daily Briefing Scheduler

| Variable | Default | Description |
|---|---|---|
| `DAILY_BRIEFING_HOUR` | `7` | Hour (UTC) to send daily briefing |
| `DAILY_BRIEFING_MINUTE` | `0` | Minute to send daily briefing |
| `DAILY_BRIEFING_SOURCES` | `news,crypto,stocks` | Comma-separated sources for the briefing |

### Optional API Keys

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub personal access token (higher rate limits) |
| `SERPER_API_KEY` | [Serper.dev](https://serper.dev) for Google search |
| `RAPIDAPI_KEY` | RapidAPI key for premium news |

## Data Sources

All sources are **free** and work without API keys (unless noted):

| Source | Command | Description |
|---|---|---|
| Google News | `/news` | RSS-based news search |
| DuckDuckGo | `/ddg` | Web search |
| DuckDuckGo News | `/ddgnews` | News search |
| Weather | `/weather` | Weather from wttr.in |
| Crypto | `/crypto` | CoinGecko prices + trending |
| Stocks | `/stocks` | Yahoo Finance quotes |
| Reddit | `/reddit` | Reddit post search |
| GitHub | `/github` | Repository search |
| arXiv | `/arxiv` | Research paper search |
| Wikipedia | `/wiki` | Article search |
| TMZ | `/tmz` | Celebrity news |
| CryptoPanic | `/cryptonews` | Crypto news aggregator |
| Serper | `/serper` | Google search (API key required) |

## Deploy to Render (Free)

### 1. Push to GitHub

```bash
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ai-research-agent.git
git push -u origin main
```

### 2. Create Render services

1. Go to [render.com](https://render.com) > **New** > **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and creates 2 free services:
   - **ai-research-api** — REST API + WhatsApp webhooks (web service)
   - **telegram-bot** — Telegram bot + daily scheduler (worker)

### 3. Set environment variables

In the Render dashboard, set these secrets for **both services**:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | Get free at [openrouter.ai](https://openrouter.ai) |
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather |
| `GREENAPI_INSTANCE_ID` | From [green-api.com](https://green-api.com) |
| `GREENAPI_API_TOKEN` | From green-api.com |
| `GREENAPI_GROUP_ID` | Your WhatsApp group ID |

Render uses **OpenRouter** (cloud LLM) instead of Ollama since there's no GPU on the free tier. The default model `deepseek/deepseek-chat` is cheap and capable.

### 4. Deploy

Click **Deploy** and wait ~2 minutes. Your services will be live at:
- API: `https://ai-research-api.onrender.com`
- Telegram bot: starts polling automatically

## WhatsApp Group Setup (Step by Step)

### 1. Create a Green API account

1. Go to [green-api.com](https://green-api.com) and sign up (free tier)
2. Create a new instance
3. Scan the QR code with your phone's WhatsApp (like WhatsApp Web)
4. Copy your **Instance ID** and **API Token**

### 2. Get your group's chat ID

After linking your WhatsApp, find your group's chat ID:

```bash
# List all your WhatsApp groups:
curl -s "https://api.green-api.com/waInstanceYOUR_INSTANCE_ID/getContacts/YOUR_API_TOKEN" | python3 -m json.tool | grep -A2 "@g.us"
```

The group ID looks like `120363XXXXX@g.us`. Add it to your `.env` as `GREENAPI_GROUP_ID`.

### 3. Test it

```bash
# Send a test message to your group:
curl -X POST "https://api.green-api.com/waInstanceYOUR_INSTANCE_ID/sendMessage/YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chatId": "YOUR_GROUP_ID@g.us", "message": "Hello from AI Research Agent!"}'
```

### 4. Enable daily briefings

Set these in your `.env`:

```env
GREENAPI_INSTANCE_ID=your_instance_id
GREENAPI_API_TOKEN=your_api_token
GREENAPI_GROUP_ID=120363XXXXX@g.us
DAILY_BRIEFING_HOUR=7
DAILY_BRIEFING_SOURCES=news,crypto,stocks
```

The scheduler runs inside the Telegram worker and sends a briefing to your WhatsApp group every day at the configured time (UTC).

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + LLM provider info |
| `GET` | `/sources` | List all available data sources |
| `POST` | `/query` | Auto-route a query to the best source |
| `POST` | `/query/{source}` | Query a specific source |
| `GET` | `/webhook/whatsapp` | WhatsApp webhook verification |
| `POST` | `/webhook/whatsapp` | WhatsApp incoming message handler |

### Example API requests

```bash
# Auto-route (LLM picks the best source)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Bitcoin doing today?"}'

# Query a specific source
curl -X POST http://localhost:8000/query/weather \
  -H "Content-Type: application/json" \
  -d '{"message": "London"}'

# List available sources
curl http://localhost:8000/sources
```

## Customization

### Add a new data source

1. Create `src/collectors/your_source.py`:

```python
from src.collectors.base import BaseCollector, CollectedItem

class YourCollector(BaseCollector):
    async def _fetch(self, query: str, **kwargs) -> list[CollectedItem]:
        # Fetch data from your API
        return [CollectedItem(title="...", content="...", url="...", source="your_source")]

    async def close(self):
        pass
```

2. Register it in `src/collectors/__init__.py`:

```python
from src.collectors.your_source import YourCollector
COLLECTOR_REGISTRY["your_source"] = YourCollector
```

3. Add to `SOURCE_COMMANDS` in `src/bot.py` and `SOURCE_MAP` in `src/whatsapp.py` for command support.

### Change the LLM models

**Local (Ollama):**
```bash
# Use any Ollama model
ollama pull gemma2:9b
# Update .env
OLLAMA_MODEL=gemma2:9b
OLLAMA_ANALYSIS_MODEL=gemma2:9b
```

**Cloud (OpenRouter):**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

Browse models at [openrouter.ai/models](https://openrouter.ai/models).

### Change daily briefing content

```env
# Only crypto and stocks
DAILY_BRIEFING_SOURCES=crypto,stocks

# Add weather and Reddit
DAILY_BRIEFING_SOURCES=news,crypto,stocks,weather,reddit

# Change time (e.g., 9:30 AM UTC)
DAILY_BRIEFING_HOUR=9
DAILY_BRIEFING_MINUTE=30
```

## Project Structure

```
ai-research-agent/
├── main.py                  # Entry point (Telegram bot + scheduler)
├── api.py                   # FastAPI REST API + WhatsApp webhooks
├── render.yaml              # Render deployment config (2 free services)
├── requirements.txt         # Python dependencies
├── start.sh                 # Auto-start script (local)
├── .env.example             # Environment variable template
├── src/
│   ├── bot.py               # Telegram bot (commands, model picker)
│   ├── whatsapp.py          # WhatsApp handler (Green API + Meta Cloud API)
│   ├── scheduler.py         # Daily briefing scheduler
│   ├── graph.py             # LangGraph pipeline (route → collect → retry → analyze → respond)
│   ├── llm.py               # LLM clients (Ollama + OpenRouter)
│   ├── config.py            # Settings (pydantic-settings)
│   ├── storage.py           # SQLite database (query logs + subscribers)
│   ├── logging_config.py    # Structured logging (structlog)
│   └── collectors/          # 14+ data source plugins
│       ├── base.py          # BaseCollector with auto-retry
│       ├── news.py          # Google News
│       ├── weather.py       # wttr.in
│       ├── crypto.py        # CoinGecko
│       ├── stocks.py        # Yahoo Finance
│       ├── reddit.py        # Reddit
│       ├── github.py        # GitHub
│       ├── arxiv.py         # arXiv
│       ├── wiki.py          # Wikipedia
│       ├── ddg.py           # DuckDuckGo
│       ├── serper.py        # Google (Serper)
│       ├── tmz.py           # TMZ
│       └── cryptonews.py    # CryptoPanic
└── tests/
    └── test_e2e.py          # End-to-end integration tests
```

## Self-Healing Pipeline

The LangGraph workflow automatically retries with fallback sources when a collector fails:

```
ddg fails     → tries ddg_news → tries news → tries reddit
crypto fails  → tries cryptonews → tries ddg_news
arxiv fails   → tries ddg → tries news → tries github
wikipedia fails → tries ddg → tries news
```

Max 2 retries per query. If all fallbacks fail, it returns a graceful error message listing what was tried.

## License

MIT
