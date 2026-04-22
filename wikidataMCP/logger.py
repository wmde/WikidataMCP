"""Logging service for the FastAPI application."""

import asyncio
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import declarative_base, sessionmaker

"""
MySQL database setup for storing Wikidata labels in all languages.
"""

DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASS = os.environ["DB_PASS"]
DB_PORT = int(os.environ.get("DB_PORT", "3306"))

DATABASE_URL = f"mariadb+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Limit the number of open connections
    max_overflow=10,  # Allow extra connections beyond pool_size
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_pre_ping=True,
)

Base = declarative_base()
Session = sessionmaker(bind=engine, expire_on_commit=False)


def _utcnow_naive() -> datetime:
    """Return naive UTC datetime without using deprecated utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Logger(Base):
    """Logging model for user requests."""

    __tablename__ = "requests"
    __table_args__ = (
        Index("ix_requests_toolname_timestamp", "toolname", "timestamp"),
        Index("ix_requests_redaction_scan", "is_redacted", "timestamp", "id"),
        {"mysql_charset": "utf8mb4"},
    )

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=_utcnow_naive, index=True, nullable=False)
    toolname = Column(String(128), index=True, nullable=False, default="/")
    parameters = Column(JSON, default=dict, nullable=False)
    response_time = Column(Float, nullable=False)
    is_redacted = Column(Boolean, default=False, index=True, nullable=False)

    # User Agent
    user_agent = Column(String(255))
    user_agent_hash = Column(String(64), index=True, nullable=False)

    @staticmethod
    def add_request(
        toolname,
        start_time,
        parameters=None,
        user_agent='',
    ):
        """Add a new request log entry.

        Args:
            request (_type_): The incoming request object.
            start_time (_type_): The time when the request was received.
            toolname (str, optional): The logged tool name. Use "/" for home page views.
            parameters (dict | None, optional): Request parameters. Defaults to None.
        """
        with Session() as session:
            try:
                # Clean up old logs (older than 90 days)
                Logger.redact_old_requests(90, 1000)

                user_agent_hash = sha256(user_agent.encode("utf-8")).hexdigest()

                # Add new log entry
                log_entry = Logger(
                    toolname=toolname[:128] if toolname else "",
                    user_agent=user_agent,
                    user_agent_hash=user_agent_hash,
                    parameters=parameters,
                    response_time=time.time() - start_time,
                    is_redacted=False,
                )
                session.add(log_entry)
                session.commit()
            except Exception:
                session.rollback()
                traceback.print_exc()

    @staticmethod
    def add_request_async(**kwargs):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(
                Logger.add_request,
                **kwargs
            ))
        except:
            # no running loop (safe fallback)
            Logger.add_request(**kwargs)

    @staticmethod
    def redact_old_requests(days: int = 90, batch_size: int = 1000):
        """Redacts old request logs.

        Args:
            days (int, optional): The age of logs to redact in days. Defaults to 90.
            batch_size (int, optional): The number of logs to process in each batch. Defaults to 1000.
        """
        cutoff_date = _utcnow_naive() - timedelta(days=days)
        with Session() as session:
            try:
                old_requests = (
                    session.query(Logger)
                    .filter(Logger.timestamp < cutoff_date)
                    .filter((Logger.is_redacted.is_(None)) | (Logger.is_redacted.is_(False)))
                    .order_by(Logger.id.asc())
                    .yield_per(batch_size)
                )

                changed = False
                for row in old_requests:
                    row.user_agent = ""
                    row.parameters = {}
                    row.is_redacted = True
                    changed = True

                if changed:
                    session.commit()

            except Exception:
                session.rollback()
                traceback.print_exc()


def initialize_database():
    """Create tables if they do not already exist."""
    try:
        Base.metadata.create_all(engine)
        return True
    except Exception as e:
        print(f"Error while initializing labels database: {e}")
        return False

initialize_database()