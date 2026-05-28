from fastapi import HTTPException, status
from app.core.auth import hash_password, verify_password, create_access_token
from app.database.connection import db
from app.schemas.auth_schema import RegisterRequest, LoginRequest, ChangePasswordRequest


class AuthService:
    async def register(self, payload: RegisterRequest) -> dict:
        try:
            user = db.create_user(
                username=payload.username,
                email=payload.email,
                password_hash=hash_password(payload.password),
                role=payload.role,
                organization_id=payload.organization_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

        token = create_access_token({"sub": user["id"], "role": user["role"]})
        return {"access_token": token, "token_type": "bearer", "user": user}

    async def login(self, payload: LoginRequest) -> dict:
        user = db.get_user_by_email(payload.email)
        if not user or not verify_password(payload.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        token = create_access_token({"sub": user["id"], "role": user["role"]})
        safe_user = {k: v for k, v in user.items() if k != "password_hash"}
        return {"access_token": token, "token_type": "bearer", "user": safe_user}

    async def forgot_password(self, email: str) -> dict:
        # In production: send reset email
        return {"message": "If an account with that email exists, a reset link has been sent."}

    async def change_password(self, user_id: str, payload: ChangePasswordRequest) -> dict:
        user = db.get_user_by_id(user_id)
        if not verify_password(payload.current_password, user["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        db.update_user(user_id, {"password_hash": hash_password(payload.new_password)})
        return {"message": "Password changed successfully"}
