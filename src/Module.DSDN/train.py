import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import nn_arch
import numpy as np
from keras_preprocessing.image import img_to_array
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.applications import imagenet_utils

from tools import create_dir


def prepare_data(spec_dir: str) -> (list, list):
    """
    Prepare data for future learning
    Args:
        spec_dir: path to directory with spectrograms

    Returns:
        list of collected data nd labels
    """
    spectrogram_paths = sorted(list(os.listdir(spec_dir)))
    random.shuffle(spectrogram_paths)

    # collect all data and labels
    data = []
    labels = []
    for spectrogram_filename in spectrogram_paths:
        spectrogram_path = spec_dir + spectrogram_filename
        spectrogram = cv2.imread(spectrogram_path)
        spectrogram = resize(spectrogram, (RESIZE, RESIZE))
        spectrogram = img_to_array(spectrogram)

        data.append(spectrogram)
        label = ''.join(spectrogram_path.split('/')[1].split()[0])
        labels.append(label)

    # transform
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    data = np.array(data, dtype="float") / 255.0
    data = imagenet_utils.preprocess_input(data, mode='tf')

    return data, labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spectrograms_dir", required=False, default="spec_set/", help="path to spectograms dir")
    ap.add_argument("-r", "--resize", required=False, default=224, help="image resize")
    ap.add_argument("-e", "--epochs", required=False, default=10, help="number of epochs")
    ap.add_argument("-bs", "--batch_size", required=False, default=10, help="batch size")
    ap.add_argument("-cn", "--classes_number", required=False, default=2, help="number of classes")
    args = vars(ap.parse_args())

    spectrograms_dir = args["spectrograms_dir"]

    RESIZE = int(args["resize"])
    EPOCHS = int(args["epochs"])
    BS = int(args["batch_size"])
    NO_CLASSES = int(args["classes_number"])
    IMG_DEPTH = 3

    data, labels = prepare_data(spectrograms_dir)

    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, random_state=42, test_size=0.25)

    model = nn_arch.SimpleVGG16.build(RESIZE, RESIZE, IMG_DEPTH, NO_CLASSES)

    loss = 'binary_crossentropy' if NO_CLASSES is 2 else "categorical_crossentropy"
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    print("---Fit generator---")
    H = model.fit(train_data, train_labels, batch_size=BS, epochs=EPOCHS, validation_data=(valid_data, valid_labels))

    # save model to disk
    create_dir("models")
    model.save("models\\audio.h5")

    # plot the training loss and accuracy and save fig
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("models\\audio.png")
