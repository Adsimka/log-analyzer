"""
Сборка всех роутеров API v1.
"""

from fastapi import APIRouter

from src.api.v1.ingest import router as ingest_router
from src.api.v1.results import router as results_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(ingest_router)
api_v1_router.include_router(results_router)