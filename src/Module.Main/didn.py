import cv2
import numpy as np
import tensorflow as tf
from config import DIDN_PATH
from grpc_manager import grpc_predict, grpc_prep


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


def get_prediction_didn(transformed_frames):
    stub, grpc_request = grpc_prep(DIDN_PATH, "input_1", "didn", transformed_frames[0])
    pred_bbox = grpc_predict(stub, grpc_request, "tf_op_layer_concat_10")

    pred_conf = list(pred_bbox._values)[4::5]
    if len(pred_conf) > 0:
        return max(pred_conf)
    return 0
