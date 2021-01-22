import argparse
import cv2
import imutils
import numpy as np
from fgn_data_transformation import reshape, get_optical_flow, set_optical_flow, normalize_respectively
from imutils import paths
from keras.models import load_model
from config import FRAMES_NO


def reshape_frames(frames):
    """
    Reshaped frames using method from fgn_data_transformation module
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
    predict = model.predict(data)[0]
    return predict * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path of the model")
    ap.add_argument("-t", "--test_set", required=False, default="testset/", help="path to test images")
    ap.add_argument("-s", "--show", required=False, default=False, action='store_true', help="frame resize number")
    args = vars(ap.parse_args())

    TESTSET_DIR = args["test_set"]
    MODEL_PATH = args["model"]
    SHOW = args["show"]

    # load the trained network
    model = load_model(MODEL_PATH)
    model.summary()

    video_paths = list(paths.list_files(TESTSET_DIR))
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        len_frames = int(cap.get(7))

        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()

            if SHOW:
                output = imutils.resize(frame, width=400)

            frames.append(frame)

            if len(frames) == FRAMES_NO:
                frames = reshape_frames(frames)
                data = transform_frames(frames)

                predict = get_prediction(data)

                if SHOW:
                    winners_indexes = np.argsort(predict)[::-1]

                    for n, index in enumerate(winners_indexes):
                        index_name = "Fight" if index == 0 else "NoFight"
                        label = "{}: {:.6f}%".format(index_name, round(predict[index], 2))

                        cv2.putText(output, label, (10, (n * 30) + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                frames.pop(0)

            if SHOW:
                cv2.putText(output, "{}/64".format(len(frames)+1), (350, 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow("Output", output)

                key = cv2.waitKey(500) & 0xFF
                if key == ord("q"):
                    break
