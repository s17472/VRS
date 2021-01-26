"""
Benedykt Kościński
"""
# Builtins
from typing import List

# 3rd party
from fastapi import APIRouter
from fastapi import File
from fastapi import UploadFile
from fastapi import status

from app.model.prediction import PredictionPayload
from app.model.prediction import PredictionResult
from app.service.prediction import PredictionService
from tensorflow.keras.applications.resnet50 import ResNet50

router = APIRouter()

model = ResNet50(weights='imagenet')
print('Model Loaded')

@router.post("/", response_model=PredictionResult, status_code=status.HTTP_201_CREATED, response_description="Files uploaded")
async def upload_files_for_prediction(files: List[UploadFile] = File(...)) -> PredictionResult:
    """
    Upload files for violence detection analysis.
    """
    # payload = PredictionPayload(data=files)
    service = PredictionService(model)
    return service.predict(payload=files)
