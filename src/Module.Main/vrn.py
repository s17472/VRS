"""
Script responsible for reshaping, transforming and then sending data to container with VRN module served via TensorFlow Serving
- Benedykt Kościński
"""
import cv2
import numpy as np

from config import VRN_FRAME_COUNT, VRN_ADDRESS
from grpc_manager import grpc_predict


def vrn_reshape(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        reshaped_frames.append(frame)

    return reshaped_frames


def vrn_transform(frames):
    frames = np.array(frames)
    frames = (frames / 255.).astype(np.float16)
    data = vgg_predict(frames)
    data = np.reshape(data, [1, VRN_FRAME_COUNT, 4096])
    return data


def vgg_predict(data):
    return grpc_predict(data, VRN_ADDRESS, "input_1", "fc2", "vgg_base")
    

def vrn_predict(data):
    return grpc_predict(data, VRN_ADDRESS, "lstm_input", "activation_2", "vrn")[0]
