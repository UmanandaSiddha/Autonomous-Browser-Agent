from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStep(str, Enum):
    INITIALIZING = "initializing"
    AUTHENTICATING = "authenticating"
    EXTRACTING_EMAILS = "extracting_emails"
    SUMMARIZING = "summarizing"
    VALIDATING = "validating"
    COMPLETED = "completed"


class AutomationJob(BaseModel):
    id: str

    status: JobStatus
    step: JobStep

    progress: int

    created_at: datetime
    updated_at: datetime

    error: str | None = None

    result: dict | None = None