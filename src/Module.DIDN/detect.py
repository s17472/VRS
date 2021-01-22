import time
import tensorflow as tf
from absl import app
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = 416
    video_path = './test_set/video.mp4'
    saved_model_loaded = tf.saved_model.load('./model/model_dir_name', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # start video capturing process
    try:
        video = cv2.VideoCapture(int(video_path))
    except:
        video = cv2.VideoCapture(video_path)

    while True:
        return_value, frame = video.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        else:
            print('Error during video reading')
            break

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        print(pred_bbox)
        for key, value in pred_bbox.items():
            pred_conf = value[:, :, 4:]
        scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])).numpy()

        fps = 1.0 / (time.time() - start_time)

        # return value in % and fps
        if scores.size > 0: print(np.amax(scores))
        print("FPS: %.2f" % fps)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
