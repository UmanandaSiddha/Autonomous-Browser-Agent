from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from pwdlib import PasswordHash
from pydantic_settings import BaseSettings, SettingsConfigDict


password_hash = PasswordHash.recommended()


class _AuthSettings(BaseSettings):
    """
    Reads backend/.env. A real environment variable still wins,
    so deployments can override without touching the file.
    """

    jwt_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        extra="ignore",
    )


JWT_SECRET = _AuthSettings().jwt_secret

# RFC 7518 requires >= 32 bytes for HS256. Failing loudly beats
# silently signing every token with a weak fallback.
if len(JWT_SECRET) < 32:
    raise RuntimeError(
        "JWT_SECRET missing or too short. Set a value of at "
        "least 32 characters in backend/.env"
    )

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    return password_hash.verify(
        password,
        hashed_password,
    )


def create_access_token(user_id: str) -> str:
    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    payload = {
        "sub": user_id,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )