from pydantic import BaseModel

from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionType(str, Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    NONE = "none"


class EmailMessage(BaseModel):
    sender_name: str
    sender_email: str
    subject: str
    snippet: str
    timestamp: str
    thread_id: str | None = None
    link: str | None = None


class ActionItem(BaseModel):
    action: str
    reason: str
    priority: Priority
    action_type: ActionType


class EmailDigest(BaseModel):
    summary: str
    priority_items: list[ActionItem]
    action_items: list[str]