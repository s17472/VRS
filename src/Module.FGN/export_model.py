import argparse
import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to model")
    ap.add_argument("-sd", "--save_dir", required=False, default="model/", help="model save directory")
    ap.add_argument("-t", "--test_set", required=False, default="../testset/", help="path to test images")
    ap.add_argument("-r", "--resize", required=False, default=64, help="size of the frame")
    ap.add_argument("-s", "--show", required=False, default=False, action='store_true', help="frame resize number")
    args = vars(ap.parse_args())

    MODEL_DIR = args["save_dir"]
    MODEL_PATH = args["model"]
    MODEL_VERSION = 1

    export_path = os.path.join(MODEL_DIR, str(MODEL_VERSION))
    print('export_path = {}\n'.format(export_path))

    model = load_model(MODEL_PATH)
    model.summary()

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
