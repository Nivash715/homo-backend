import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _validate_email(v: str) -> str:
    if not _EMAIL_RE.match(v):
        raise ValueError("Invalid email address")
    return v.lower().strip()


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8)
    role: str = Field(default="analyst", pattern="^(admin|analyst|viewer)$")
    organization_id: Optional[str] = None

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, v):
        return _validate_email(v)


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, v):
        return _validate_email(v)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., description="User email address")

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, v):
        return _validate_email(v)


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
