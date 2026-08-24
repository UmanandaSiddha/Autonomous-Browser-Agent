from pydantic import BaseModel


class EmailMessage(BaseModel):
    sender_name: str
    sender_email: str
    subject: str
    snippet: str
    timestamp: str
    thread_id: str | None = None


class ActionItem(BaseModel):
    action: str
    reason: str
    priority: str
    action_type: str


class EmailDigest(BaseModel):
    summary: str
    priority_items: list[ActionItem]
    action_items: list[str]