"""Request and response shapes for the public API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    # extra="forbid": a client sending company_id (or anything else) gets a 422,
    # instead of the field being silently ignored. The company comes only from the JWT.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def _clean(cls, value: str) -> str:
        # Drop control characters (keep normal whitespace), collapse runs of spaces.
        value = "".join(ch for ch in value if ch.isprintable() or ch in "\n\t")
        value = " ".join(value.split())
        if not value:
            raise ValueError("question is empty")
        return value


class Citation(BaseModel):
    document: str
    version: int
    effective_date: str
    section: str
    heading: str
    excerpt: str  # the policy text itself, from the store, so the employee can check it


Status = Literal["answered", "not_covered", "no_policy", "try_again"]


class AskResponse(BaseModel):
    status: Status
    answer: str
    citations: list[Citation]
    request_id: str


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_id: str = Field(min_length=1, max_length=50)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    company_id: str
    role: str


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=200_000)


class IngestResponse(BaseModel):
    status: str
    company_id: str
    version: int
    chunk_count: int
    quarantined_sections: list[str] = []  # not indexed: text addressed to an AI
