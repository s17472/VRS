import cv2
import numpy as np

from config import FGN_PATH
from fgn_data_transformation import get_optical_flow, set_optical_flow, normalize_respectively
from grpc_manager import grpc_predict, grpc_prep


def reshape_fgn(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (64, 64, 3))
        reshaped_frames.append(frame)

    return reshaped_frames


def transform_fgn(frames):
    collected_frames = np.array(frames)

    flows = get_optical_flow(collected_frames)

    data = set_optical_flow(collected_frames, flows)
    data = np.float32(data)

    data = normalize_respectively(data)
    data = np.array([data])
    return data


def get_prediction_fgn(data):
    stub, grpc_request = grpc_prep(FGN_PATH, "input_1", 'fgn', data)
    predict = grpc_predict(stub, grpc_request, "dense_2")[0]
    return predict
