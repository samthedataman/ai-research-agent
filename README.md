# AI Research Agent

An AI-powered research assistant that automatically searches the internet, collects data from 14+ free sources, and delivers intelligent summaries — all without paying for a single API.

Accessible via **Telegram bot**, **WhatsApp group**, **REST API**, or **automated daily briefings**.

## What This Project Teaches

This project is a working example of how to build an **AI agent** — not just a chatbot that answers questions, but a system that can:

1. **Understand intent** — figure out what data source to use based on a natural language question
2. **Take actions** — go fetch real data from the internet (news, stocks, crypto, weather, etc.)
3. **Self-heal** — if one data source fails, automatically try alternatives
4. **Synthesize information** — use an LLM to turn raw data into a useful briefing
5. **Deliver results** — send the output wherever users are (Telegram, WhatsApp, API)
6. **Run autonomously** — schedule daily briefings without any human trigger

These are the same patterns used in production AI agents at companies like Google, Bloomberg, and OpenAI.

## How It Works (The Big Picture)

```
  "What's happening in crypto?"
              │
              ▼
     ┌─────────────────┐
     │   1. ROUTE       │  A small, fast LLM reads the question and decides:
     │                   │  "This is about crypto → use CoinGecko"
     └────────┬──────────┘
              │
              ▼
     ┌─────────────────┐
     │   2. COLLECT     │  The agent calls the CoinGecko API,
     │                   │  pulls trending coins, prices, and 24h changes
     └────────┬──────────┘
              │
              ▼  (empty results?)
     ┌─────────────────┐
     │   3. RETRY       │  Self-healing: tries CryptoPanic → DuckDuckGo News
     │   (if needed)    │  → Google News until it gets data (max 2 retries)
     └────────┬──────────┘
              │
              ▼
     ┌─────────────────┐
     │   4. ANALYZE     │  A larger, smarter LLM reads all the collected data
     │                   │  and writes a structured briefing with key takeaways
     └────────┬──────────┘
              │
              ▼
     ┌─────────────────┐
     │   5. RESPOND     │  Formats the output for the target platform
     │                   │  (Telegram Markdown, WhatsApp, or JSON API)
     └──────────────────┘
```

This is built using **LangGraph** — a framework for building stateful, multi-step AI workflows as directed graphs. Each box above is a "node" in the graph, and the arrows are "edges" that define the flow.

## Key Concepts Explained

### Why Two LLMs?

The system uses two models for different jobs:

| Role | Model | Why |
|---|---|---|
| **Router** (fast) | `llama3.2` (3B params) | Picks the right data source. Needs to be fast, not smart. Runs in <1 second. |
| **Analyzer** (smart) | `llama3` (8B params) | Synthesizes data into a briefing. Needs to be smart, speed matters less. |

This is a common production pattern called **model routing** — use cheap/fast models for simple decisions, expensive/smart models for complex reasoning.

### What is LangGraph?

[LangGraph](https://langchain-ai.github.io/langgraph/) is a framework for building AI workflows as state machines. Instead of a simple prompt → response, you define:

- **State** — a dictionary that flows through the graph (the query, collected items, analysis, etc.)
- **Nodes** — functions that read and modify the state (route, collect, analyze, respond)
- **Edges** — connections between nodes, including conditional edges (if collect returns empty → retry)

This gives you control flow that a simple LLM chain can't provide — branching, loops, retries, error handling.

Our graph in `src/graph.py`:
```python
route → collect → [conditional: has data?]
                  ├── YES → analyze → respond → END
                  └── NO  → retry → collect (loop back, max 2x)
```

### What is Ollama?

[Ollama](https://ollama.ai) lets you run open-source LLMs locally on your laptop. No API keys, no cloud costs, no data leaving your machine. You download a model once and run it forever for free.

```bash
ollama pull llama3.2:latest   # Downloads ~2GB, runs on any Mac/Linux
ollama pull llama3:latest     # Downloads ~4.7GB, needs 8GB+ RAM
```

The tradeoff: local models are slower and less capable than cloud models like GPT-4 or Claude. But they're **free** and **private**.

### What is OpenRouter?

[OpenRouter](https://openrouter.ai) is a unified API that gives you access to 100+ LLMs (GPT-4, Claude, Llama, Mistral, DeepSeek, etc.) through a single endpoint. You pay per token, and many models are extremely cheap (<$0.001 per query).

We use OpenRouter when deploying to the cloud (Render), since Ollama requires a GPU which free hosting doesn't provide.

### What is the Collector Pattern?

Each data source is a "collector" — a Python class that:
1. Takes a search query
2. Calls a free API
3. Returns structured results (title, content, URL, source)

All collectors inherit from `BaseCollector` and implement one method: `_fetch()`. This makes it trivial to add new sources — just write a class and register it.

```python
class WeatherCollector(BaseCollector):
    async def _fetch(self, query, **kwargs):
        # Call wttr.in API
        resp = await self.client.get(f"https://wttr.in/{query}?format=j1")
        # Parse and return structured data
        return [CollectedItem(title=f"Weather: {query}", content=..., url=..., source="weather")]
```

### What is Self-Healing?

Real-world APIs fail. They rate-limit you, return empty results, or go down entirely. A production agent needs to handle this gracefully.

Our self-healing system uses a **fallback chain** — each source has an ordered list of alternatives:

```python
FALLBACK_CHAIN = {
    "ddg":       ["ddg_news", "news", "reddit"],      # DuckDuckGo fails → try DDG News → Google News → Reddit
    "crypto":    ["cryptonews", "ddg_news", "news"],   # CoinGecko fails → try CryptoPanic → DDG News
    "arxiv":     ["ddg", "news", "github"],             # arXiv fails → try DuckDuckGo → News → GitHub
    "wikipedia": ["ddg", "news"],                       # Wikipedia fails → try DuckDuckGo → News
}
```

The LangGraph conditional edge checks: "Did we get data?" If no, it picks the next untried source and loops back to collect. Max 2 retries, then it gives a graceful "I tried X, Y, Z but couldn't find results" message.

## Data Sources (All Free)

| Source | Command | API | What it returns |
|---|---|---|---|
| Google News | `/news` | RSS feeds | Headlines, summaries, links |
| DuckDuckGo | `/ddg` | duckduckgo-search | Web search results |
| DuckDuckGo News | `/ddgnews` | duckduckgo-search | News articles |
| Weather | `/weather` | wttr.in | Temperature, forecast, humidity |
| Crypto | `/crypto` | CoinGecko | Prices, market cap, 24h change, trending |
| Stocks | `/stocks` | Yahoo Finance | Price, change, volume, P/E ratio |
| Reddit | `/reddit` | Reddit JSON | Top posts from subreddits |
| GitHub | `/github` | GitHub API | Repos, stars, descriptions |
| arXiv | `/arxiv` | arXiv API | Research paper titles, abstracts |
| Wikipedia | `/wiki` | Wikipedia API | Article summaries |
| TMZ | `/tmz` | Web scraping | Celebrity news |
| CryptoPanic | `/cryptonews` | CryptoPanic API | Crypto news feed |
| Serper | `/serper` | Serper.dev | Google search (needs API key) |
| DexScreener | - | DexScreener API | DEX token trading data |

**Zero API keys required for 13 of 14 sources.** Only Serper needs a paid key.

---

## Setup Guide

### Option A: Run Locally with Ollama (Free, Private)

#### Step 1: Install prerequisites

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.ai

# Install Python 3.12+
# macOS: brew install python@3.12
# Or use pyenv: pyenv install 3.12.1
```

#### Step 2: Pull LLM models

```bash
ollama serve  # Start Ollama (runs in background)
ollama pull llama3.2:latest  # Fast routing model (~2GB)
ollama pull llama3:latest    # Smart analysis model (~4.7GB)
```

#### Step 3: Clone and install

```bash
git clone https://github.com/samthedataman/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
```

#### Step 4: Configure

```bash
cp .env.example .env
```

Edit `.env` and add your Telegram bot token:
```env
TELEGRAM_BOT_TOKEN=your_token_here  # Get from @BotFather on Telegram
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_ANALYSIS_MODEL=llama3:latest
```

#### Step 5: Run

```bash
# Start the Telegram bot
python main.py

# Or start the REST API
uvicorn api:app --host 0.0.0.0 --port 8000

# Or use the auto-start script (handles Ollama + bot)
chmod +x start.sh
./start.sh
```

### Option B: Deploy to Render (Free, Cloud)

Render is a cloud platform that hosts your app for free. Since there's no GPU on the free tier, we use OpenRouter (cloud LLM) instead of Ollama.

#### Step 1: Get API keys

1. **Telegram bot token** — Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the prompts
2. **OpenRouter API key** — Sign up at [openrouter.ai](https://openrouter.ai), go to Keys, create one (free credits included)

#### Step 2: Push to GitHub

```bash
# Fork or clone the repo
git clone https://github.com/samthedataman/ai-research-agent.git
cd ai-research-agent

# Push to YOUR GitHub account
git remote set-url origin https://github.com/YOUR_USERNAME/ai-research-agent.git
git push -u origin main
```

#### Step 3: Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** > **Blueprint**
3. Connect your GitHub repo
4. Render reads `render.yaml` and creates **2 free services**:
   - **ai-research-api** — REST API + WhatsApp webhooks (web service, sleeps after 15 min inactivity)
   - **telegram-bot** — Telegram bot + daily scheduler (worker, always running)

#### Step 4: Set environment variables

In the Render dashboard, go to each service's **Environment** tab and add:

| Variable | Value | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `sk-or-...` | From openrouter.ai |
| `TELEGRAM_BOT_TOKEN` | `123456:ABC...` | From @BotFather |

That's it. The `render.yaml` already sets `LLM_PROVIDER=openrouter` and `OPENROUTER_MODEL=deepseek/deepseek-chat` for you.

#### Step 5: Verify

- Send `/start` to your Telegram bot — it should respond
- Visit `https://ai-research-api.onrender.com/` — should return `{"status": "ok"}`
- Try `https://ai-research-api.onrender.com/sources` — should list all 14+ sources

### Understanding render.yaml (Render Deployment Blueprint)

The `render.yaml` file tells Render exactly how to deploy your app. It creates **2 free services** that use **OpenRouter** (cloud LLM) instead of Ollama — no GPU needed.

Here's the full file with explanations:

```yaml
services:
  # SERVICE 1: REST API + WhatsApp Webhooks
  # This is a "web" service — it responds to HTTP requests
  # Render gives it a public URL like https://ai-research-api.onrender.com
  - type: web
    name: ai-research-api
    env: python                                    # Python runtime
    plan: free                                     # Free tier (750 hours/month)
    buildCommand: pip install -r requirements.txt  # Runs once during deploy
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT  # FastAPI server
    envVars:
      - key: LLM_PROVIDER
        value: openrouter          # Use cloud LLM (not local Ollama)
      - key: OPENROUTER_API_KEY
        sync: false                # "sync: false" = set manually in dashboard (secret)
      - key: OPENROUTER_MODEL
        value: deepseek/deepseek-chat  # Cheap + capable model (~$0.001/query)
      - key: OLLAMA_ANALYSIS_MODEL
        value: deepseek/deepseek-chat  # Same model for analysis on cloud
      - key: PYTHON_VERSION
        value: "3.12.1"
      # WhatsApp secrets (optional, set in dashboard if using)
      - key: GREENAPI_INSTANCE_ID
        sync: false
      - key: GREENAPI_API_TOKEN
        sync: false
      - key: GREENAPI_GROUP_ID
        sync: false

  # SERVICE 2: Telegram Bot + Daily Scheduler
  # This is a "worker" — it runs continuously in the background
  # No public URL, just a long-running Python process
  - type: worker
    name: telegram-bot
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py    # Runs bot polling + scheduler loop
    envVars:
      - key: TELEGRAM_BOT_TOKEN
        sync: false                 # Set your bot token in the dashboard
      - key: LLM_PROVIDER
        value: openrouter
      - key: OPENROUTER_API_KEY
        sync: false
      - key: OPENROUTER_MODEL
        value: deepseek/deepseek-chat
      - key: OLLAMA_ANALYSIS_MODEL
        value: deepseek/deepseek-chat
      # Green API for WhatsApp daily briefings
      - key: GREENAPI_INSTANCE_ID
        sync: false
      - key: GREENAPI_API_TOKEN
        sync: false
      - key: GREENAPI_GROUP_ID
        sync: false
      # Scheduler config
      - key: DAILY_BRIEFING_HOUR
        value: "7"                  # 7:00 AM UTC
      - key: DAILY_BRIEFING_SOURCES
        value: "news,crypto,stocks" # What to include in the daily briefing
```

**Key things to understand:**
- `type: web` = receives HTTP requests (REST API, webhooks). Sleeps after 15 min of inactivity, wakes on next request.
- `type: worker` = runs forever in the background (Telegram polling, scheduler). Never sleeps.
- `sync: false` = secret value, set manually in Render dashboard (never committed to git)
- `LLM_PROVIDER=openrouter` = uses cloud LLM, not local Ollama. This is the key difference from running locally.
- `deepseek/deepseek-chat` = a very cheap cloud model (~$0.14/million tokens). You can swap to any model on [openrouter.ai/models](https://openrouter.ai/models).

**To use a different cloud model**, just change `OPENROUTER_MODEL` and `OLLAMA_ANALYSIS_MODEL` to any model ID from OpenRouter. Examples:
- `google/gemma-2-9b-it` — free on OpenRouter
- `meta-llama/llama-3.1-8b-instruct` — very cheap
- `anthropic/claude-3.5-sonnet` — more expensive but very capable

### Option C: Add WhatsApp Group Messaging

The bot can send daily briefings to your WhatsApp group using [Green API](https://green-api.com) (free tier).

#### Step 1: Create a Green API account

1. Go to [green-api.com](https://green-api.com) and sign up
2. Create a new instance (free tier = 1 instance)
3. Scan the QR code with your phone's WhatsApp (just like WhatsApp Web)
4. Copy your **Instance ID** and **API Token** from the dashboard

#### Step 2: Find your group's chat ID

```bash
# Replace YOUR_INSTANCE_ID and YOUR_API_TOKEN with your actual values
curl -s "https://api.green-api.com/waInstanceYOUR_INSTANCE_ID/getContacts/YOUR_API_TOKEN" \
  | python3 -c "import json,sys; [print(c['id'],c.get('name','')) for c in json.load(sys.stdin) if '@g.us' in c['id']]"
```

This prints your WhatsApp groups with their IDs. The ID looks like `120363XXXXX@g.us`.

#### Step 3: Test sending a message

```bash
curl -X POST "https://api.green-api.com/waInstanceYOUR_INSTANCE_ID/sendMessage/YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chatId": "YOUR_GROUP_ID", "message": "Hello from AI Research Agent!"}'
```

If you see the message in your WhatsApp group, it's working.

#### Step 4: Configure

Add to your `.env` (local) or Render environment variables:

```env
GREENAPI_INSTANCE_ID=your_instance_id
GREENAPI_API_TOKEN=your_api_token
GREENAPI_GROUP_ID=120363XXXXX@g.us
DAILY_BRIEFING_HOUR=7          # 7:00 AM UTC
DAILY_BRIEFING_SOURCES=news,crypto,stocks
```

The scheduler runs alongside the Telegram bot and sends a briefing to your group every morning.

---

## REST API Reference

The API runs on FastAPI with automatic Swagger docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns LLM provider and model info |
| `GET` | `/sources` | List all available data sources |
| `POST` | `/query` | Send a question — LLM auto-picks the best source |
| `POST` | `/query/{source}` | Force a specific source (e.g., `/query/weather`) |
| `GET` | `/webhook/whatsapp` | WhatsApp webhook verification (Meta) |
| `POST` | `/webhook/whatsapp` | Incoming WhatsApp message handler |

### Examples

```bash
# Ask a question (LLM picks the source)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Bitcoin doing today?"}'

# Force a specific source
curl -X POST http://localhost:8000/query/weather \
  -H "Content-Type: application/json" \
  -d '{"message": "Tokyo"}'

# List all sources
curl http://localhost:8000/sources
```

### Response format

```json
{
  "source": "crypto",
  "query": "Bitcoin",
  "items": [{"title": "Bitcoin", "content": "Price: $97,234...", "url": "...", "source": "crypto_coingecko"}],
  "analysis": "Key Takeaway: Bitcoin is trading at...",
  "response": "Full formatted briefing...",
  "tried_sources": ["crypto"],
  "model_used": "llama3:latest"
}
```

---

## How to Customize

### Add your own data source

Every data source is a collector class in `src/collectors/`. To add a new one:

**1. Create the collector** (`src/collectors/your_source.py`):

```python
import httpx
from src.collectors.base import BaseCollector, CollectedItem

class YourCollector(BaseCollector):
    async def _fetch(self, query: str, **kwargs) -> list[CollectedItem]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://your-api.com/search?q={query}")
            data = resp.json()

        results = []
        for item in data["results"]:
            results.append(CollectedItem(
                title=item["title"],
                content=item["description"],
                url=item["link"],
                source="your_source",
            ))
        return results

    async def close(self):
        pass
```

**2. Register it** in `src/collectors/__init__.py`:
```python
from src.collectors.your_source import YourCollector
COLLECTOR_REGISTRY["your_source"] = YourCollector
```

**3. Add the command** in `src/bot.py` (line ~30) and `src/whatsapp.py` (line ~19):
```python
SOURCE_COMMANDS["yoursource"] = "Your source description (free)"
# and
SOURCE_MAP["yoursource"] = "your_source"
```

**4. Add a fallback chain** in `src/graph.py`:
```python
FALLBACK_CHAIN["your_source"] = ["ddg", "news"]
```

### Change LLM models

**Locally (Ollama)** — use any model from [ollama.ai/library](https://ollama.ai/library):
```bash
ollama pull gemma2:9b
# Update .env:
OLLAMA_MODEL=gemma2:9b
OLLAMA_ANALYSIS_MODEL=gemma2:9b
```

**Cloud (OpenRouter)** — use any model from [openrouter.ai/models](https://openrouter.ai/models):
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### Change daily briefing schedule

```env
# What to include (comma-separated source names)
DAILY_BRIEFING_SOURCES=news,crypto,stocks,weather,reddit

# When to send (UTC)
DAILY_BRIEFING_HOUR=9
DAILY_BRIEFING_MINUTE=30
```

---

## Project Structure

```
ai-research-agent/
│
├── main.py                  # Entry point — starts Telegram bot + daily scheduler
├── api.py                   # FastAPI REST API + WhatsApp webhook endpoints
├── render.yaml              # Render deployment config (2 free services)
├── requirements.txt         # Python dependencies (11 packages, all free)
├── start.sh                 # Auto-start script (checks Ollama, starts bot)
├── .env.example             # Template for environment variables
│
├── src/
│   ├── graph.py             # LangGraph pipeline — the brain of the agent
│   │                        #   route → collect → retry → analyze → respond
│   │
│   ├── llm.py               # LLM clients — Ollama (local) + OpenRouter (cloud)
│   │                        #   Factory pattern: get_llm_client() picks the right one
│   │
│   ├── bot.py               # Telegram bot — commands, model picker, message handling
│   ├── whatsapp.py          # WhatsApp bot — Green API (groups) + Meta Cloud API (DMs)
│   ├── scheduler.py         # Daily briefing — generates + sends at scheduled time
│   │
│   ├── config.py            # Settings — all env vars in one place (pydantic-settings)
│   ├── storage.py           # Database — SQLite for query logs + WhatsApp subscribers
│   ├── logging_config.py    # Structured logging — JSON in production, pretty in dev
│   │
│   └── collectors/          # Data source plugins (each is ~50-100 lines)
│       ├── base.py          # BaseCollector — abstract class with auto-retry
│       ├── news.py          # Google News (RSS parsing)
│       ├── weather.py       # wttr.in (free weather API)
│       ├── crypto.py        # CoinGecko (prices, trending, market data)
│       ├── stocks.py        # Yahoo Finance (quotes, P/E, volume)
│       ├── reddit.py        # Reddit (JSON API, no auth needed)
│       ├── github.py        # GitHub (repo search, stars, descriptions)
│       ├── arxiv.py         # arXiv (research papers, abstracts)
│       ├── wiki.py          # Wikipedia (article summaries)
│       ├── ddg.py           # DuckDuckGo (web search + news)
│       ├── serper.py        # Google via Serper (paid API key)
│       ├── tmz.py           # TMZ (celebrity news scraping)
│       └── cryptonews.py    # CryptoPanic (crypto news feed)
│
└── tests/
    └── test_e2e.py          # 36 end-to-end tests covering all layers
```

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **AI Workflow** | LangGraph | Stateful graph with conditional edges, retries, branching |
| **LLM (local)** | Ollama | Free, private, runs on laptop |
| **LLM (cloud)** | OpenRouter | Access to 100+ models, pay-per-token |
| **Telegram Bot** | python-telegram-bot | Mature async library, supports inline keyboards |
| **WhatsApp** | Green API | Free tier, supports group messaging |
| **REST API** | FastAPI | Auto-generated docs, async, type-safe |
| **Database** | SQLite + aiosqlite | Zero config, async, file-based |
| **Config** | pydantic-settings | Type-safe env vars with validation |
| **Logging** | structlog | Structured JSON logs for production |
| **HTTP** | httpx | Modern async HTTP client |
| **Hosting** | Render | Free tier with Blueprint deployment |

## Configuration Reference

<details>
<summary>Click to expand full configuration table</summary>

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | (required) | Telegram bot token from @BotFather |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openrouter` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Routing model (fast, small) |
| `OLLAMA_ANALYSIS_MODEL` | `llama3:latest` | Analysis model (smart, large) |
| `OPENROUTER_API_KEY` | | OpenRouter API key |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat` | Cloud model |
| `GREENAPI_INSTANCE_ID` | | Green API instance ID |
| `GREENAPI_API_TOKEN` | | Green API token |
| `GREENAPI_GROUP_ID` | | WhatsApp group chat ID |
| `WHATSAPP_ACCESS_TOKEN` | | Meta Cloud API token |
| `WHATSAPP_PHONE_NUMBER_ID` | | Meta phone number ID |
| `WHATSAPP_VERIFY_TOKEN` | `whatsapp-verify-token` | Webhook verification secret |
| `DAILY_BRIEFING_HOUR` | `7` | Daily briefing hour (UTC) |
| `DAILY_BRIEFING_MINUTE` | `0` | Daily briefing minute |
| `DAILY_BRIEFING_SOURCES` | `news,crypto,stocks` | Sources for daily briefing |
| `GITHUB_TOKEN` | | GitHub PAT (higher rate limits) |
| `SERPER_API_KEY` | | Serper.dev Google search key |
| `DATABASE_URL` | `sqlite+aiosqlite:///./bot.db` | Database connection string |
| `LOG_LEVEL` | `INFO` | Logging level |

</details>

## License

MIT
