import cv2
import numpy as np

from config import VRN_FRAME_COUNT, VRN_PATH
from grpc_manager import grpc_prep, grpc_predict


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

    stub, grpc_request = grpc_prep(VRN_PATH, "input_1", "vgg_base", frames)
    transfer_values = grpc_predict(stub, grpc_request, "fc2")
    data = np.reshape(transfer_values, [1, VRN_FRAME_COUNT, 4096])
    return data


def get_prediction_vrn(data):
    stub, grpc_request = grpc_prep(VRN_PATH, "lstm_input", "vrn", data)
    score = grpc_predict(stub, grpc_request, "activation_2")
    return score[0]
