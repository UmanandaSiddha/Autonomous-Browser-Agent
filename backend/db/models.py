import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStep(str, enum.Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    AUTHENTICATING = "authenticating"
    EXTRACTING_EMAILS = "extracting_emails"
    SUMMARIZING = "summarizing"
    VALIDATING = "validating"
    COMPLETED = "completed"


def _enum_column(enum_class):
    """
    SQLite has no native enum type, so store the
    value as VARCHAR and validate in Python.
    """
    return Enum(
        enum_class,
        native_enum=False,
        validate_strings=True,
        values_callable=lambda e: [
            member.value for member in e
        ],
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AutomationJob(Base):
    __tablename__ = "automation_jobs"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        index=True,
        nullable=False,
    )

    automation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    status: Mapped[JobStatus] = mapped_column(
        _enum_column(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED,
    )

    step: Mapped[JobStep | None] = mapped_column(
        _enum_column(JobStep),
        nullable=True,
    )

    progress: Mapped[int] = mapped_column(
        default=0,
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    result: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
