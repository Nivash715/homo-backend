"""
Custom middleware: request timing, logging, security headers.
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        logger.info(f"[{request_id}] ▶ {request.method} {request.url.path}")

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.error(f"[{request_id}] ✖ Unhandled error: {exc}")
            raise

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{request_id}] ◀ {response.status_code} — {elapsed:.1f}ms"
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        # Skip extra headers for CORS preflight — browser is sensitive here
        if request.method == "OPTIONS":
            return response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response
