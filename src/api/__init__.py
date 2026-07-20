"""API router aggregation."""

from fastapi import APIRouter

from .offline import router as offline_router
from .online import router as online_router
from .spk import router as spk_router
from .system import router as system_router

router = APIRouter()
router.include_router(offline_router)
router.include_router(online_router)
router.include_router(spk_router)
router.include_router(system_router)

__all__ = ["router"]
