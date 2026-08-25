from pydantic import BaseModel


class TrainingActionItem(BaseModel):
    action: str
    reason: str
    priority: str
    action_type: str


class TrainingExample(BaseModel):
    emails: list[dict]
    digest: dict