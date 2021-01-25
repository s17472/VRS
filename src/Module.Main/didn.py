import cv2
import numpy as np
import tensorflow as tf
from config import DIDN_PATH
from grpc_manager import grpc_predict, grpc_prep, grpc_request


def reshape_didn(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (416, 416))
        reshaped_frames.append(frame)

    return reshaped_frames


def transform_didn(frames):
    frames = np.array(frames)
    frames = (frames / 255.).astype(np.float16)
    data = frames[np.newaxis, ...]
    data = tf.constant(data)
    return data


def prediction_didn(data):
    pred_box = grpc_request(data[0], DIDN_PATH, "input_1", "tf_op_layer_concat_10", "didn")
    pred_conf = list(pred_box._values)[4::5]
    if len(pred_conf) > 0:
        return max(pred_conf)
    return 0
