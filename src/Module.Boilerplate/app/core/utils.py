"""
Benedykt Kościński
"""
from tensorflow.keras.applications.resnet50 import ResNet50


def load_model():
    model = ResNet50(weights='imagenet')
    return model
