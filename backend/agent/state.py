from typing import TypedDict

from backend.services.models import EmailDigest, EmailMessage


class AgentState(TypedDict):
    authenticated: bool
    emails: list[EmailMessage]
    digest: EmailDigest | None
    error: str | None
    retry_count: int