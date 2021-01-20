import argparse
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tools import create_dir
import librosa.display


def create_spectrogram(audio_file: str, filename: str):
    """
    Create spectrogram from audio file and save as picture
    Args:
        audio_file: path to audio file
        filename: name of the file to be saved
    """
    clip, sample_rate = librosa.load(audio_file)

    fig = plt.figure(figsize=[5, 5])

    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)


def extract_spectrograms(audio_dir: str):
    """
    Extract spectrograms from all audio file in directory
    Args:
        audio_dir: path to directory
    """
    for audio_file in os.listdir(audio_dir):
        filename = audio_file.split('.')[0]
        spec_file = spec_dir + filename + ".jpg"

        create_spectrogram(audio_dir + audio_file, spec_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--audio_dir", required=False, default="audio_set/", help="path to target audio dataset dir")
    ap.add_argument("-s", "--spectograms_dir", required=False, default="spec_set/", help="path to spectrograms dir")
    args = vars(ap.parse_args())

    audio_dir = args["audio_dir"]
    spec_dir = args["spectograms_dir"]

    create_dir(spec_dir)

    extract_spectrograms(audio_dir)
