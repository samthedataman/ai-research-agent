import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.storage.models import Base, CollectedItemDB, AnalysisResultDB
from src.storage.repository import Repository
from src.collectors.base_collector import CollectedItem


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


class TestRepository:
    @pytest.mark.asyncio
    async def test_save_and_get_items(self, db_session):
        repo = Repository(db_session)
        items = [
            CollectedItem(
                source="test",
                title="Test Article",
                content="Test content here",
                url="https://example.com",
            )
        ]

        ids = await repo.save_collected_items(items)
        assert len(ids) == 1

        pending = await repo.get_pending_items()
        assert len(pending) == 1
        assert pending[0].title == "Test Article"

    @pytest.mark.asyncio
    async def test_mark_item_analyzed(self, db_session):
        repo = Repository(db_session)
        items = [
            CollectedItem(source="test", title="Test", content="Content")
        ]
        ids = await repo.save_collected_items(items)

        await repo.mark_item_analyzed(ids[0], "completed")

        pending = await repo.get_pending_items()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_save_analysis(self, db_session):
        repo = Repository(db_session)
        items = [
            CollectedItem(source="test", title="Test", content="Content")
        ]
        ids = await repo.save_collected_items(items)

        analysis_id = await repo.save_analysis(
            item_id=ids[0],
            analysis_type="sentiment",
            result_text="positive sentiment detected",
            sentiment="positive",
            confidence=0.95,
            topics=["AI", "tech"],
            model_used="test-model",
        )

        analyses = await repo.get_analyses_for_item(ids[0])
        assert len(analyses) == 1
        assert analyses[0].sentiment == "positive"
        assert analyses[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_save_and_get_report(self, db_session):
        repo = Repository(db_session)

        report_id = await repo.save_report(
            report_type="daily",
            title="Daily Report",
            content="Report content here",
            items_analyzed=["item1", "item2"],
        )

        reports = await repo.get_reports(report_type="daily")
        assert len(reports) == 1
        assert reports[0].title == "Daily Report"
