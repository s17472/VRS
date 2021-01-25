import cv2
import numpy as np

from config import FGN_ADDRESS
from fgn_data_transformation import (get_optical_flow, normalize_respectively,
                                     set_optical_flow)
from grpc_manager import grpc_request


def fgn_reshape(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (64, 64, 3))
        reshaped_frames.append(frame)

    return reshaped_frames


def fgn_transform(frames):
    collected_frames = np.array(frames)

    flows = get_optical_flow(collected_frames)

    data = set_optical_flow(collected_frames, flows)
    data = np.float32(data)

    data = normalize_respectively(data)
    data = np.array([data])
    return data


def fgn_predict(data):
    return grpc_request(data, FGN_ADDRESS, "input_1", "dense_2", "fgn")[0]
