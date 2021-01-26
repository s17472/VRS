"""
Benedykt Kościński
"""
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image
from fastapi import UploadFile
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import decode_predictions

from ..model.prediction import PredictionProbability
from ..model.prediction import PredictionResult


class PredictionService():
    def __init__(self, model):
        self.model = model

    def _set_probablity(self, prediction: float) -> PredictionProbability:
        if prediction < 0.5:
            return PredictionProbability.low
        elif prediction < 0.7:
            return PredictionProbability.medium
        elif prediction < 0.9:
            return PredictionProbability.high
        else:
            return PredictionProbability.very_high

    def prepare_image(self, img, target):
        # If the image mode is not RGB, convert it
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize the input image and preprocess it
        img = img.resize(target)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = imagenet_utils.preprocess_input(img)

        # Return the processed image
        return img


    def predict(self, payload: List[UploadFile]) -> PredictionResult:
        if payload is None:
            raise ValueError("No valid payload.")

        frame = payload[0].file.read()
        img = Image.open(BytesIO(frame))
        x = self.prepare_image(img, (224, 224))


        from timeit import default_timer as timer
        start = timer()
        preds = self.model.predict(x)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282

        results = decode_predictions(preds, top=1)[0]
        print('Predicted:', results)

        prediction = 0.50
        return PredictionResult(prediction=results[0][2], probablity=self._set_probablity(results[0][2]))
