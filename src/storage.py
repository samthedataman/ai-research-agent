import datetime
import uuid

from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """Logs every user query and bot response for history."""
    __tablename__ = "query_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, nullable=False, index=True)
    source = Column(String, nullable=False)
    query = Column(String, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def log_query(user_id: int, source: str, query: str, response: str) -> None:
    """Save a query + response to the database."""
    async with async_session() as session:
        entry = QueryLog(user_id=user_id, source=source, query=query, response=response)
        session.add(entry)
        await session.commit()
        logger.info("query_logged", user_id=user_id, source=source)


async def get_history(user_id: int, limit: int = 10) -> list[QueryLog]:
    """Get recent query history for a user."""
    async with async_session() as session:
        result = await session.execute(
            select(QueryLog)
            .where(QueryLog.user_id == user_id)
            .order_by(QueryLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
