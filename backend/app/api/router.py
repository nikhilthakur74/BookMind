"""Root API router combining individual route modules."""

from fastapi import APIRouter
from app.api.endpoints import books, evaluations, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(books.router)
api_router.include_router(evaluations.router)
