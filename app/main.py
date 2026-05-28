"""
HomoFedAI CyberSec Platform — FastAPI entry point.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from app.utils.logger import logger

# ─── Routes ──────────────────────────────────────────────────────────────────
from app.routes.auth_routes import router as auth_router
from app.routes.organization_routes import router as org_router
from app.routes.threat_routes import router as threat_router
from app.routes.federated_routes import router as federated_router
from app.routes.encryption_routes import router as encryption_router
from app.routes.analytics_routes import router as analytics_router
from app.routes.admin_routes import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting…")
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs(settings.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs("./uploads", exist_ok=True)
    yield
    logger.info("Server shutting down — goodbye.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Privacy-Preserving Federated AI Cybersecurity Platform with Homomorphic Encryption",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────
# NOTE: Starlette applies middleware in reverse-add order (last added = outermost).
# CORSMiddleware MUST be outermost so it intercepts OPTIONS preflight before
# any other middleware touches the request.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev: allow all origins
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ─── Static files ────────────────────────────────────────────────────────────
os.makedirs("./uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ─── Routers ─────────────────────────────────────────────────────────────────
PREFIX = settings.API_PREFIX
app.include_router(auth_router,       prefix=f"{PREFIX}/auth",         tags=["Authentication"])
app.include_router(org_router,        prefix=f"{PREFIX}/organizations", tags=["Organizations"])
app.include_router(threat_router,     prefix=f"{PREFIX}/threats",       tags=["Threat Intelligence"])
app.include_router(federated_router,  prefix=f"{PREFIX}/federated",     tags=["Federated Learning"])
app.include_router(encryption_router, prefix=f"{PREFIX}/encryption",    tags=["Encryption"])
app.include_router(analytics_router,  prefix=f"{PREFIX}/analytics",     tags=["Analytics"])
app.include_router(admin_router,      prefix=f"{PREFIX}/admin",         tags=["Admin"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "platform": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": settings.APP_VERSION}
