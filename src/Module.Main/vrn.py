import cv2
import numpy as np
from config import VRN_FRAME_COUNT, VRN_PATH
from grpc_manager import grpc_predict, grpc_prep, grpc_request


def reshape_vrn(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        reshaped_frames.append(frame)

    return reshaped_frames


def transform_vrn(frames):
    frames = np.array(frames)
    frames = (frames / 255.).astype(np.float16)
    data = predict_vgg(frames)
    data = np.reshape(data, [1, VRN_FRAME_COUNT, 4096])
    return data


def predict_vgg(data):
    return grpc_request(data, VRN_PATH, "input_1", "fc2", "vgg_base")
    

def predict_vrn(data):
    return grpc_request(data, VRN_PATH, "lstm_input", "activation_2", "vrn")[0]
