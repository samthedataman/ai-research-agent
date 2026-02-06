"""
Stock Market Collector - Yahoo Finance (FREE, no API key)
=========================================================
Uses Yahoo Finance's public query endpoints.
No API key required.
"""

from typing import Any

import httpx

from src.collectors.base import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class StocksCollector(BaseCollector):
    """Collects stock market data from Yahoo Finance (free, no API key)."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance"

    def __init__(self):
        super().__init__(name="stocks")
        self.client = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            },
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch stock data. Query can be:
        - A ticker symbol like "AAPL" or "TSLA"
        - Comma-separated tickers like "AAPL,GOOGL,MSFT"
        - "market" for major indices
        """
        if query.lower() == "market":
            symbols = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"]
        else:
            symbols = [s.strip().upper() for s in query.split(",")]

        return await self._fetch_quotes(symbols)

    async def _fetch_quotes(self, symbols: list[str]) -> list[CollectedItem]:
        """Fetch quotes for a list of symbols."""
        symbol_str = ",".join(symbols)

        try:
            response = await self.client.get(
                f"https://query1.finance.yahoo.com/v7/finance/quote",
                params={
                    "symbols": symbol_str,
                    "fields": "symbol,shortName,regularMarketPrice,regularMarketChange,"
                    "regularMarketChangePercent,regularMarketVolume,marketCap,"
                    "fiftyTwoWeekHigh,fiftyTwoWeekLow,regularMarketDayHigh,"
                    "regularMarketDayLow,regularMarketOpen",
                },
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPStatusError, Exception):
            # Fallback to chart endpoint for individual quotes
            return await self._fetch_quotes_fallback(symbols)

        items: list[CollectedItem] = []
        for quote in data.get("quoteResponse", {}).get("result", []):
            symbol = quote.get("symbol", "")
            name = quote.get("shortName", symbol)
            price = quote.get("regularMarketPrice", 0)
            change = quote.get("regularMarketChange", 0)
            change_pct = quote.get("regularMarketChangePercent", 0)
            volume = quote.get("regularMarketVolume", 0)
            market_cap = quote.get("marketCap", 0)
            high_52w = quote.get("fiftyTwoWeekHigh", 0)
            low_52w = quote.get("fiftyTwoWeekLow", 0)
            day_high = quote.get("regularMarketDayHigh", 0)
            day_low = quote.get("regularMarketDayLow", 0)

            direction = "up" if change >= 0 else "down"
            content = (
                f"{name} ({symbol}): ${price:,.2f} ({direction} {abs(change_pct):.2f}%). "
                f"Day range: ${day_low:,.2f} - ${day_high:,.2f}. "
                f"52-week range: ${low_52w:,.2f} - ${high_52w:,.2f}. "
                f"Volume: {volume:,.0f}."
            )
            if market_cap:
                if market_cap >= 1e12:
                    cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    cap_str = f"${market_cap/1e6:.2f}M"
                content += f" Market cap: {cap_str}."

            items.append(
                CollectedItem(
                    source="stocks_yahoo",
                    title=f"{symbol}: ${price:,.2f} ({change_pct:+.2f}%)",
                    content=content,
                    url=f"https://finance.yahoo.com/quote/{symbol}",
                    metadata={
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "change": change,
                        "change_pct": change_pct,
                        "volume": volume,
                        "market_cap": market_cap,
                    },
                )
            )
        return items

    async def _fetch_quotes_fallback(self, symbols: list[str]) -> list[CollectedItem]:
        """Fallback using chart API for individual symbols."""
        items: list[CollectedItem] = []
        for symbol in symbols[:5]:
            try:
                response = await self.client.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"interval": "1d", "range": "5d"},
                )
                response.raise_for_status()
                data = response.json()

                result = data.get("chart", {}).get("result", [{}])[0]
                meta = result.get("meta", {})
                price = meta.get("regularMarketPrice", 0)
                prev_close = meta.get("chartPreviousClose", 0)
                name = meta.get("shortName", symbol)

                change = price - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0

                content = f"{name} ({symbol}): ${price:,.2f} ({change_pct:+.2f}%)"

                items.append(
                    CollectedItem(
                        source="stocks_yahoo",
                        title=f"{symbol}: ${price:,.2f}",
                        content=content,
                        url=f"https://finance.yahoo.com/quote/{symbol}",
                        metadata={"symbol": symbol, "price": price, "change_pct": change_pct},
                    )
                )
            except Exception as e:
                logger.warning("stock_fetch_error", symbol=symbol, error=str(e))
        return items

    async def close(self) -> None:
        await self.client.aclose()
