"""
Benedykt Kościński
"""
# Builtins
from enum import Enum
from typing import List

# 3rd party
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import Field


class PredictionPayload(BaseModel):
    data: List[UploadFile]


class PredictionProbability(str, Enum):
    low = 'Low'
    medium = 'Medium'
    high = 'High'
    very_high = 'Very High'


class PredictionResult(BaseModel):
    prediction: float = Field(0, ge=0, le=1)
    probablity: PredictionProbability
