"""
Script responsible for reshaping, transforming and then sending data to container with DIDN module served via TensorFlow Serving
- Benedykt Kościński
"""
import cv2
import numpy as np
import tensorflow as tf

from config import DIDN_ADDRESS
from grpc_manager import grpc_predict


def didn_reshape(frames):
    reshaped_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (416, 416))
        reshaped_frames.append(frame)

    return reshaped_frames


def didn_transform(frames):
    frames = np.array(frames)
    frames = (frames / 255.).astype(np.float16)
    data = frames[np.newaxis, ...]
    data = tf.constant(data)
    return data


def didn_predict(data):
    pred_box = grpc_predict(data[0], DIDN_ADDRESS, "input_1", "tf_op_layer_concat_10", "didn")
    pred_conf = list(pred_box._values)[4::5]
    if len(pred_conf) > 0:
        return max(pred_conf)
    return 0
