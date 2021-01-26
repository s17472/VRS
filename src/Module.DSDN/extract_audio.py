import argparse
import csv

from moviepy.editor import *
from yt_downloader import YTDownloader

from tools import create_dir, refactor, remove_dir, status_message


class AudioSetCollector:
    """
    Audio dataset collector from AudioSet site
    Args:
        csv_data_path: path to csv file with video annotations downloaded from AydioSet site
        csv_labels_path: path to csv file with annotations information downloaded from AudioSet site
        target_label: name of the label to downloaded
        dataset_dir: path to the folder where audio will be saved
    """
    def __init__(self, csv_data_path: str, csv_labels_path: str, target_label: str, dataset_dir: str = "set/"):
        self.csv_data_path = csv_data_path
        self.csv_labels_path = csv_labels_path
        self.dataset_dir = dataset_dir
        self.trim_dir = "trim_" + dataset_dir
        self.audio_dir = "audio_" + dataset_dir
        self.target_label = target_label

        # create dirs if not exist
        create_dir(self.dataset_dir)
        create_dir(self.trim_dir)
        create_dir(self.audio_dir)

        self.labels = self.get_labels()
        self.target_label_code = self.labels[target_label]

        self.data = self.get_data()

    def get_labels(self) -> dict:
        """
        Extract all labels form csv file

        Returns: dictionary with label name and corresponding code
        """
        with open(self.csv_labels_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            labels = {row[2]: row[1] for row in csv_reader}
        return labels

    def get_data(self) -> dict:
        """
        Gets all data from csv file to dictionary

        Returns: dictionary with all collected data
        """
        with open(self.csv_data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader = list(csv_reader)

            data = {}
            for i, row in enumerate(csv_reader):
                status_message(i, len(csv_reader), "extracted data")
                if self.target_label_code in "".join(row):
                    data[row[0]] = list(map(lambda x: int(float(x)), row[1:3]))
        return data

    def download_videos(self):
        """
        Download all videos in collected data
        """
        yt_downloader = YTDownloader(self.dataset_dir)

        for i, id in enumerate(self.data.keys()):
            # prints status message
            status_message(i, len(self.data), "downloaded videos")
            yt_downloader.download(id, name=self.target_label)

    def cut_videos(self):
        """
        Trim all downloaded videos
        """
        videos = os.listdir(self.dataset_dir)

        for video in videos:
            # get id of yt video from filename
            id = video.split()[-1][:-4]

            # get start and end time
            start, end = self.data[id]

            clip = VideoFileClip(self.dataset_dir + video)
            # check for error in end time
            if end > clip.duration:
                end = clip.duration
            clip = clip.subclip(start, end)
            clip.write_videofile(self.trim_dir + video)

    def save_audio(self):
        """
        Save audio from all trimmed videos with wav extension
        """
        videos = os.listdir(self.trim_dir)

        for video in videos:
            # change extension
            audio_save = self.audio_dir + refactor(video, "wav")
            video = self.trim_dir + video

            video = VideoFileClip(video)
            audio = video.audio

            if audio is None:
                continue

            audio.write_audiofile(audio_save)

    def extract_audio(self):
        """
        Extract audio from all videos
        """
        self.download_videos()
        self.cut_videos()
        self.save_audio()

        remove_dir(self.dataset_dir)
        remove_dir(self.trim_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_dir", required=False, default="set/", help="path to target dataset dir")
    ap.add_argument("-lp", "--labels_path", required=True, help="path to labels file")
    ap.add_argument("-dp", "--data_path", required=True, help="path to data file")
    ap.add_argument("-l", "--label", required=True, help="label name")
    args = vars(ap.parse_args())

    audio_set = AudioSetCollector(args["data_path"], args["labels_path"], args["label"], args["dataset_dir"])

    audio_set.extract_audio()
