import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, String, Text, JSON
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class CollectedItemDB(Base):
    __tablename__ = "collected_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(String, default="")
    published_at = Column(String, default="")
    metadata_json = Column(JSON, default=dict)
    collected_at = Column(DateTime, default=datetime.datetime.utcnow)
    analyzed = Column(String, default="pending")  # pending, completed, failed


class AnalysisResultDB(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    item_id = Column(String, nullable=False, index=True)
    analysis_type = Column(String, nullable=False)  # sentiment, topic, summary
    result = Column(Text, nullable=False)
    sentiment = Column(String, default="")
    confidence = Column(Float, default=0.0)
    topics = Column(JSON, default=list)
    model_used = Column(String, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ReportDB(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    report_type = Column(String, nullable=False)  # daily, weekly, alert
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    items_analyzed = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
