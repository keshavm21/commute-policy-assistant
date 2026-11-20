"""JWT issuing (the mock identity provider) and verification.

The company an employee belongs to comes ONLY from a verified token. Nothing in a
request body, header or query string can change it.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError, field_validator

from app.config import Settings, get_settings
from app.ingestion import COMPANY_ID_RE

ALGORITHM = "HS256"


class Principal(BaseModel):
    """Who is calling, as proven by the token."""

    employee_id: str
    company_id: str
    role: Literal["employee", "admin"]

    @field_validator("company_id")
    @classmethod
    def _valid_company(cls, value: str) -> str:
        if not COMPANY_ID_RE.match(value):
            raise ValueError("invalid company_id")
        return value


class AuthError(Exception):
    pass


def issue_token(principal: Principal, settings: Settings, ttl_minutes: int | None = None) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": principal.employee_id,
        "company_id": principal.company_id,
        "role": principal.role,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes or settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str, settings: Settings) -> Principal:
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],  # pinned: rejects "alg": "none" and algorithm swaps
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "company_id", "role", "iss", "aud", "iat", "exp"]},
        )
        return Principal(
            employee_id=claims["sub"], company_id=claims["company_id"], role=claims["role"]
        )
    except (jwt.PyJWTError, ValidationError) as exc:
        raise AuthError(str(exc)) from None


def load_users(path: Path) -> dict[str, Principal]:
    """The mock user directory: employee_id -> Principal."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        r["employee_id"]: Principal(
            employee_id=r["employee_id"], company_id=r["company_id"], role=r["role"]
        )
        for r in rows
    }


# ---- FastAPI dependencies ----------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    try:
        return decode_token(credentials.credentials, settings)
    except AuthError:
        raise unauthorized from None


def admin_principal(principal: Principal = Depends(current_principal)) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required.")
    return principal
