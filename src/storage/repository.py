import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.models import AnalysisResultDB, CollectedItemDB, ReportDB
from src.collectors.base_collector import CollectedItem
from src.logging_config import get_logger

logger = get_logger(__name__)


class Repository:
    def __init__(self, session: AsyncSession):
        self.session = session

    # --- Collected Items ---

    async def save_collected_items(self, items: list[CollectedItem]) -> list[str]:
        ids: list[str] = []
        for item in items:
            db_item = CollectedItemDB(
                source=item.source,
                title=item.title,
                content=item.content,
                url=item.url,
                published_at=item.published_at,
                metadata_json=item.metadata,
            )
            self.session.add(db_item)
            ids.append(db_item.id)
        await self.session.commit()
        logger.info("items_saved", count=len(ids))
        return ids

    async def get_pending_items(self, limit: int = 50) -> list[CollectedItemDB]:
        result = await self.session.execute(
            select(CollectedItemDB)
            .where(CollectedItemDB.analyzed == "pending")
            .order_by(CollectedItemDB.collected_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_item_analyzed(self, item_id: str, status: str = "completed") -> None:
        await self.session.execute(
            update(CollectedItemDB)
            .where(CollectedItemDB.id == item_id)
            .values(analyzed=status)
        )
        await self.session.commit()

    async def get_items_since(
        self, since: datetime.datetime, source: str | None = None
    ) -> list[CollectedItemDB]:
        query = select(CollectedItemDB).where(CollectedItemDB.collected_at >= since)
        if source:
            query = query.where(CollectedItemDB.source == source)
        result = await self.session.execute(query.order_by(CollectedItemDB.collected_at.desc()))
        return list(result.scalars().all())

    # --- Analysis Results ---

    async def save_analysis(
        self,
        item_id: str,
        analysis_type: str,
        result_text: str,
        sentiment: str = "",
        confidence: float = 0.0,
        topics: list[str] | None = None,
        model_used: str = "",
    ) -> str:
        analysis = AnalysisResultDB(
            item_id=item_id,
            analysis_type=analysis_type,
            result=result_text,
            sentiment=sentiment,
            confidence=confidence,
            topics=topics or [],
            model_used=model_used,
        )
        self.session.add(analysis)
        await self.session.commit()
        return analysis.id

    async def get_analyses_for_item(self, item_id: str) -> list[AnalysisResultDB]:
        result = await self.session.execute(
            select(AnalysisResultDB).where(AnalysisResultDB.item_id == item_id)
        )
        return list(result.scalars().all())

    async def get_recent_analyses(
        self, limit: int = 100, analysis_type: str | None = None
    ) -> list[AnalysisResultDB]:
        query = select(AnalysisResultDB)
        if analysis_type:
            query = query.where(AnalysisResultDB.analysis_type == analysis_type)
        result = await self.session.execute(
            query.order_by(AnalysisResultDB.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    # --- Reports ---

    async def save_report(
        self,
        report_type: str,
        title: str,
        content: str,
        items_analyzed: list[str] | None = None,
    ) -> str:
        report = ReportDB(
            report_type=report_type,
            title=title,
            content=content,
            items_analyzed=items_analyzed or [],
        )
        self.session.add(report)
        await self.session.commit()
        return report.id

    async def get_reports(
        self, report_type: str | None = None, limit: int = 10
    ) -> list[ReportDB]:
        query = select(ReportDB)
        if report_type:
            query = query.where(ReportDB.report_type == report_type)
        result = await self.session.execute(
            query.order_by(ReportDB.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
