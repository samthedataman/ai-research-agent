"""
Weather Collector - wttr.in (FREE, no API key)
===============================================
Uses wttr.in for weather data. Completely free, no rate limits,
no API key required.
"""

from typing import Any

import httpx

from src.collectors.base import BaseCollector, CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class WeatherCollector(BaseCollector):
    """Collects weather data from wttr.in (free, no API key)."""

    BASE_URL = "https://wttr.in"

    def __init__(self):
        super().__init__(name="weather")
        self.client = httpx.AsyncClient(timeout=15.0)

    async def _fetch(self, query: str, **kwargs: Any) -> list[CollectedItem]:
        """Fetch weather for a location. Query = city name."""
        location = query.replace(" ", "+")
        fmt = kwargs.get("format", "j1")  # JSON format

        response = await self.client.get(
            f"{self.BASE_URL}/{location}",
            params={"format": fmt},
            headers={"User-Agent": "ai-research-agent/1.0"},
        )
        response.raise_for_status()

        if fmt == "j1":
            data = response.json()
            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            area_name = area.get("areaName", [{}])[0].get("value", query)
            country = area.get("country", [{}])[0].get("value", "")
            temp_c = current.get("temp_C", "?")
            temp_f = current.get("temp_F", "?")
            desc = current.get("weatherDesc", [{}])[0].get("value", "")
            humidity = current.get("humidity", "?")
            wind_speed = current.get("windspeedKmph", "?")
            feels_like = current.get("FeelsLikeC", "?")

            content = (
                f"Weather in {area_name}, {country}: {desc}. "
                f"Temperature: {temp_c}°C ({temp_f}°F), feels like {feels_like}°C. "
                f"Humidity: {humidity}%, Wind: {wind_speed} km/h."
            )

            # Build forecast summary
            forecasts = data.get("weather", [])
            forecast_parts = []
            for day in forecasts[:3]:
                date = day.get("date", "")
                max_c = day.get("maxtempC", "?")
                min_c = day.get("mintempC", "?")
                desc_day = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "") if day.get("hourly") else ""
                forecast_parts.append(f"{date}: {min_c}-{max_c}°C, {desc_day}")

            if forecast_parts:
                content += " 3-day forecast: " + "; ".join(forecast_parts)

            return [
                CollectedItem(
                    source="weather_wttr",
                    title=f"Weather: {area_name}, {country}",
                    content=content,
                    url=f"{self.BASE_URL}/{location}",
                    metadata={
                        "temp_c": temp_c,
                        "temp_f": temp_f,
                        "humidity": humidity,
                        "wind_kmph": wind_speed,
                        "description": desc,
                        "location": area_name,
                        "country": country,
                    },
                )
            ]
        else:
            # Plain text format
            return [
                CollectedItem(
                    source="weather_wttr",
                    title=f"Weather: {query}",
                    content=response.text.strip(),
                    url=f"{self.BASE_URL}/{location}",
                )
            ]

    async def close(self) -> None:
        await self.client.aclose()
