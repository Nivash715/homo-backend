from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.auth_schema import (
    RegisterRequest, LoginRequest, TokenResponse,
    ForgotPasswordRequest, ChangePasswordRequest
)
from app.services.auth_service import AuthService
from app.core.auth import get_current_user

router = APIRouter()
auth_service = AuthService()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest):
    return await auth_service.register(payload)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest):
    return await auth_service.login(payload)


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {k: v for k, v in current_user.items() if k != "password_hash"}


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    return await auth_service.forgot_password(payload.email)


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user)
):
    return await auth_service.change_password(current_user["id"], payload)


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    return {"message": "Logged out successfully"}
