"""
Crypto Collector - CoinGecko Free API + DexScreener (FREE, no API key)
======================================================================
CoinGecko public API: 10-30 calls/min, no key required.
DexScreener: completely free, no key required.
"""

from typing import Any

import httpx

from src.collectors.base_collector import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class CryptoCollector(BaseCollector):
    """Collects crypto market data from CoinGecko (free, no API key)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        super().__init__(name="crypto")
        self.client = httpx.AsyncClient(
            timeout=20.0,
            headers={"User-Agent": "ai-research-agent/1.0"},
        )

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """
        Fetch crypto data. Query can be:
        - "trending" for trending coins
        - "market" for top coins by market cap
        - a coin ID like "bitcoin" or "ethereum" for specific coin data
        """
        mode = kwargs.get("mode", "auto")

        if query.lower() == "trending" or mode == "trending":
            return await self._fetch_trending()
        elif query.lower() in ("market", "top") or mode == "market":
            return await self._fetch_market(kwargs.get("limit", 10))
        else:
            return await self._fetch_coin(query)

    async def _fetch_trending(self) -> list[CollectedItem]:
        """Fetch trending coins from CoinGecko."""
        response = await self.client.get(f"{self.BASE_URL}/search/trending")
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for coin_data in data.get("coins", []):
            coin = coin_data.get("item", {})
            name = coin.get("name", "")
            symbol = coin.get("symbol", "")
            market_cap_rank = coin.get("market_cap_rank", "N/A")
            price_btc = coin.get("price_btc", 0)

            content = (
                f"{name} ({symbol}) is trending on CoinGecko. "
                f"Market cap rank: #{market_cap_rank}. "
                f"Price in BTC: {price_btc:.8f}."
            )

            items.append(
                CollectedItem(
                    source="crypto_coingecko",
                    title=f"Trending: {name} ({symbol})",
                    content=content,
                    url=f"https://www.coingecko.com/en/coins/{coin.get('id', '')}",
                    metadata={
                        "coin_id": coin.get("id", ""),
                        "symbol": symbol,
                        "market_cap_rank": market_cap_rank,
                        "price_btc": price_btc,
                    },
                )
            )
        return items

    async def _fetch_market(self, limit: int = 10) -> list[CollectedItem]:
        """Fetch top coins by market cap."""
        response = await self.client.get(
            f"{self.BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": str(limit),
                "page": "1",
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        for coin in data:
            name = coin.get("name", "")
            symbol = coin.get("symbol", "").upper()
            price = coin.get("current_price", 0)
            market_cap = coin.get("market_cap", 0)
            change_24h = coin.get("price_change_percentage_24h", 0) or 0
            change_7d = coin.get("price_change_percentage_7d_in_currency", 0) or 0
            volume = coin.get("total_volume", 0)

            direction = "up" if change_24h > 0 else "down"
            content = (
                f"{name} ({symbol}): ${price:,.2f} ({direction} {abs(change_24h):.1f}% 24h). "
                f"Market cap: ${market_cap:,.0f}. "
                f"24h volume: ${volume:,.0f}. "
                f"7d change: {change_7d:+.1f}%."
            )

            items.append(
                CollectedItem(
                    source="crypto_coingecko",
                    title=f"{name} ({symbol}) - ${price:,.2f}",
                    content=content,
                    url=f"https://www.coingecko.com/en/coins/{coin.get('id', '')}",
                    metadata={
                        "coin_id": coin.get("id", ""),
                        "symbol": symbol,
                        "price_usd": price,
                        "market_cap": market_cap,
                        "change_24h": change_24h,
                        "change_7d": change_7d,
                        "volume_24h": volume,
                    },
                )
            )
        return items

    async def _fetch_coin(self, coin_id: str) -> list[CollectedItem]:
        """Fetch data for a specific coin."""
        # First try direct ID, then search
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/coins/{coin_id.lower()}",
                params={"localization": "false", "tickers": "false", "community_data": "false"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError:
            # Search for the coin
            search_resp = await self.client.get(
                f"{self.BASE_URL}/search", params={"query": coin_id}
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            coins = search_data.get("coins", [])
            if not coins:
                return [
                    CollectedItem(
                        source="crypto_coingecko",
                        title=f"Coin not found: {coin_id}",
                        content=f"No cryptocurrency found matching '{coin_id}'.",
                    )
                ]
            # Fetch the first match
            actual_id = coins[0]["id"]
            response = await self.client.get(
                f"{self.BASE_URL}/coins/{actual_id}",
                params={"localization": "false", "tickers": "false", "community_data": "false"},
            )
            response.raise_for_status()
            data = response.json()

        name = data.get("name", "")
        symbol = data.get("symbol", "").upper()
        market_data = data.get("market_data", {})
        price = market_data.get("current_price", {}).get("usd", 0)
        market_cap = market_data.get("market_cap", {}).get("usd", 0)
        change_24h = market_data.get("price_change_percentage_24h", 0) or 0
        ath = market_data.get("ath", {}).get("usd", 0)
        description = data.get("description", {}).get("en", "")[:500]

        content = (
            f"{name} ({symbol}): ${price:,.2f} ({change_24h:+.1f}% 24h). "
            f"Market cap: ${market_cap:,.0f}. ATH: ${ath:,.2f}. "
            f"{description}"
        )

        return [
            CollectedItem(
                source="crypto_coingecko",
                title=f"{name} ({symbol}) - ${price:,.2f}",
                content=content,
                url=f"https://www.coingecko.com/en/coins/{data.get('id', '')}",
                metadata={
                    "coin_id": data.get("id", ""),
                    "symbol": symbol,
                    "price_usd": price,
                    "market_cap": market_cap,
                    "change_24h": change_24h,
                    "ath_usd": ath,
                },
            )
        ]

    async def close(self) -> None:
        await self.client.aclose()


class DexScreenerCollector(BaseCollector):
    """Collects DEX token data from DexScreener (free, no API key)."""

    BASE_URL = "https://api.dexscreener.com"

    def __init__(self):
        super().__init__(name="dexscreener")
        self.client = httpx.AsyncClient(timeout=15.0)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """Search for tokens on DexScreener."""
        response = await self.client.get(
            f"{self.BASE_URL}/latest/dex/search", params={"q": query}
        )
        response.raise_for_status()
        data = response.json()

        items: list[CollectedItem] = []
        limit = kwargs.get("limit", 5)
        for pair in data.get("pairs", [])[:limit]:
            base = pair.get("baseToken", {})
            name = base.get("name", "Unknown")
            symbol = base.get("symbol", "?")
            price = pair.get("priceUsd", "N/A")
            chain = pair.get("chainId", "?")
            liquidity = pair.get("liquidity", {}).get("usd", 0)
            volume = pair.get("volume", {}).get("h24", 0)
            change = pair.get("priceChange", {}).get("h24", 0)

            content = (
                f"{name} ({symbol}) on {chain}: ${price}. "
                f"24h change: {change}%. "
                f"Liquidity: ${liquidity:,.0f}. "
                f"24h volume: ${volume:,.0f}. "
                f"DEX: {pair.get('dexId', 'unknown')}."
            )

            items.append(
                CollectedItem(
                    source="dexscreener",
                    title=f"{symbol} on {chain} - ${price}",
                    content=content,
                    url=pair.get("url", ""),
                    metadata={
                        "chain": chain,
                        "address": base.get("address", ""),
                        "price_usd": price,
                        "liquidity_usd": liquidity,
                        "volume_24h": volume,
                        "change_24h": change,
                    },
                )
            )
        return items

    async def close(self) -> None:
        await self.client.aclose()
