import argparse
import time
import cv2
import numpy as np
from data_transformation import reshape, get_optical_flow, set_optical_flow, normalize_respectively
from keras.models import load_model
from config import FRAMES_NO


def reshape_frames(frames):
    """
    Reshaped frames using method from data_transformation module
    Args:
        frames: frames to be reshaped

    Returns:
        reshaped frames
    """
    reshaped_frames = []
    for frame in frames:
        reshaped_frames.append(reshape(frame))
    return reshaped_frames


def transform_frames(frames):
    """
    Transform frames to be corresponding to model input shape
    Args:
        frames: frames to transform

    Returns:
        transformed frames (input data)
    """
    collected_frames = np.array(frames)

    flows = get_optical_flow(collected_frames)

    data = set_optical_flow(collected_frames, flows)
    data = np.float32(data)

    data = normalize_respectively(data)
    data = np.array([data])
    return data


def get_prediction(data):
    """
    classify the input
    Args:
        data: input data

    Returns:
        prediction of the fight in percentage
    """
    predict = model.predict(data)[0][0]
    return round(predict * 100, 2)


def get_frames(path):
    """
    Reads first FRAMES_NO frames
    Args:
        path: source of video

    Returns:
        collected frames
    """
    cap = cv2.VideoCapture(path)

    frames = []
    while len(frames) != FRAMES_NO:
        _, frame = cap.read()
        frames.append(frame)
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path of the model")
    ap.add_argument("-s", "--source", required=False, default="testset/fight.avi", help="source of video")
    args = vars(ap.parse_args())

    # load the trained network
    model = load_model(args["model"])
    frames = get_frames(args["source"])

    frames = reshape_frames(frames)
    data = transform_frames(frames)

    start_time = time.time()
    prediction = get_prediction(data)
    print("Fight:", prediction)
    print("--- %s seconds ---" % (time.time() - start_time))
