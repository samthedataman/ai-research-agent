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


class WhatsAppSubscriber(Base):
    """WhatsApp users subscribed to daily briefings."""
    __tablename__ = "wa_subscribers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number = Column(String, nullable=False, unique=True, index=True)
    subscribed_at = Column(DateTime, default=datetime.datetime.utcnow)
    active = Column(Integer, default=1)  # 1 = active, 0 = unsubscribed
    preferences = Column(String, default="news,crypto,stocks")


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


# ── WhatsApp subscriber CRUD ────────────────────────────────────────────────


async def add_wa_subscriber(phone_number: str, preferences: str = "news,crypto,stocks") -> None:
    """Subscribe a WhatsApp number to daily briefings (upsert)."""
    async with async_session() as session:
        result = await session.execute(
            select(WhatsAppSubscriber).where(WhatsAppSubscriber.phone_number == phone_number)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.active = 1
            existing.preferences = preferences
        else:
            session.add(WhatsAppSubscriber(phone_number=phone_number, preferences=preferences))
        await session.commit()
        logger.info("wa_subscribed", phone=phone_number)


async def remove_wa_subscriber(phone_number: str) -> None:
    """Unsubscribe a WhatsApp number (soft delete)."""
    async with async_session() as session:
        result = await session.execute(
            select(WhatsAppSubscriber).where(WhatsAppSubscriber.phone_number == phone_number)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.active = 0
            await session.commit()
            logger.info("wa_unsubscribed", phone=phone_number)


async def get_wa_subscribers() -> list[WhatsAppSubscriber]:
    """Get all active WhatsApp subscribers."""
    async with async_session() as session:
        result = await session.execute(
            select(WhatsAppSubscriber).where(WhatsAppSubscriber.active == 1)
        )
        return list(result.scalars().all())
