"""Aggregates every v1 route under a single router."""

from fastapi import APIRouter

from app.api.v1.routes import face_verification, health

router = APIRouter()
router.include_router(health.router)
router.include_router(face_verification.router)
