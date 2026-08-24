from pydantic import BaseModel

class AuthStatusResponse(BaseModel):
    authenticated: bool

class AuthConnectResponse(BaseModel):
    authenticated: bool
    message: str