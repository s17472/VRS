"""
Benedykt Kościński
"""
# 3rd party
from fastapi import APIRouter
from fastapi import status

# Local
from app.model.about import AboutResult

router = APIRouter()


@router.get("/", response_model=AboutResult, status_code=status.HTTP_200_OK, response_description="Details about module.")
async def about() -> AboutResult:
    return AboutResult()
