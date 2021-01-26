"""
Benedykt Kościński
"""
# 3rd party
from fastapi import APIRouter

from app.router import prediction, about

router = APIRouter()
router.include_router(prediction.router, tags=["prediction"], prefix="/predict")
router.include_router(about.router, tags=["about"], prefix="/about")
