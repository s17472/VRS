import os
import numpy as np
from keras.utils import Sequence
from keras.utils import np_utils
from data_transformation import video_2_npy, color_jitter, uniform_sampling, random_flip, normalize, \
    normalize_respectively
from config import SIZE, FRAMES_NO, BATCH_SIZE


class DataGenerator(Sequence):
    """
    Data generator for Keras fit_generator
    Args:
        directory: path do data directory
        batch_size: batch size number
        shuffle: do shuffle of images
        data_augmentation: do augmentation of images
    """
    def __init__(self, directory, batch_size=BATCH_SIZE, shuffle=True, data_augmentation=True):
        # initialize the params
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_aug = data_augmentation

        # sub-folders
        self.dirs = sorted(os.listdir(self.directory))

        # gets data
        self.X_path, self.Y_dict = self.search_data()

        # basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))

    def search_data(self):
        """
        Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        Returns:
            list of paths and dictionary with pair "data:label"
        """
        x_path = []
        y_dict = {}

        # list all kinds of sub-folders
        categorical = np_utils.to_categorical(range(len(self.dirs)))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                # append the each file path, and keep its label
                x_path.append(file_path)
                y_dict[file_path] = categorical[i]
        return x_path, y_dict

    def __len__(self):
        """
        Calculate the iterations of each epoch
        Returns:
            Iterations of each epoch
        """
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """
        Gets batch data
        Args:
            index: index

        Returns:
            batch data
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        """
        Shuffle the data at each end of epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        """
        Generates batch data
        Args:
            batch_path: patch of current batch

        Returns:
            batch of data and labels
        """
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def load_data(self, path):
        """
        Transform video to correct format
        Args:
            path: patch of the video

        Returns:
            transformed data
        """
        # load np array with 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = video_2_npy(file_path=path)
        data = np.float32(data)

        data = uniform_sampling(video=data, target_frames=FRAMES_NO)
        # whether to utilize the data augmentation
        if self.data_aug:
            data[..., :3] = color_jitter(data[..., :3])
            data = random_flip(data, prob=0.5)

        # normalize rgb images and optical flows, respectively
        data = normalize_respectively(data)

        return data
