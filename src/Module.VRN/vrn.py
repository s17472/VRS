"""
Python script for learning VRN network (with RAFT OF)
- Jakub Kulaszewicz
"""

import cv2
import os
import sys
import h5py
import random
import numpy as np
from random import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model
from tensorflow import keras
from raft_optical_flow import convert_to_optical_flow



def print_progress(count, max_count):
    """
    Prints progress based on current given value (count) and it's maximum value (max_count)
    Args:
        count: numerator for counting progress
        max_count: denominator for counting progress

    Returns: -
    """
    pct = count / max_count

    msg = "\r- Progress: {0:.1%}".format(pct)

    # Print to console
    sys.stdout.write(msg)
    sys.stdout.flush()


"""
Getting access to dataset file lists for both classes
"""
dir_fight = "/path/to/fight/directory"
dir_not_fight = "/path/to/nonFight/directory"
list_fight = os.listdir(dir_fight)
list_no_fight = os.listdir(dir_not_fight)


"""
Num of videos in dataset per class (initial dataset must have 50/50 ratio for binary classification)
"""
SINGLE_CLASS_FILE_COUNT = 997

"""
Shuffle the cards ;)
"""
fight_final = random.sample(list_fight, FILE_COUNT)
no_fight_final = random.sample(list_no_fight, SINGLE_CLASS_FILE_COUNT)

"""
Labelling
"""
fight_labels = []
no_fight_labels = []
for i in range(SINGLE_CLASS_FILE_COUNT):
    fight_labels.append([1, 0])
    no_fight_labels.append([0, 1])

final = fight_final + no_fight_final

labels = fight_labels + no_fight_labels

"""
Shuffle with labels
"""
c = list(zip(final, labels))
shuffle(c)

names, labels = zip(*c)

"""
Image size to resize to and in tuple format
"""
img_size = 224
img_size_tuple = (img_size, img_size)

"""
Number of output classes for classification (Violence / NonViolence)
"""
num_classes = 2

"""
Number of frames extracted per video
"""
_images_per_file = 20

"""
Datasets names for training and testing
"""
train_dataset_name = 'training_dataset'
test_dataset_name = 'test_dataset'


def get_frames(current_dir, file_name):
    """
    Takes video file from given directory, makes transformations on each frame including OF
    Args:
        current_dir: Video file directory
        file_name: File name

    Returns:
    Extracted and transformed frames
    """
    in_file = os.path.join(current_dir, file_name)

    vidcap = cv2.VideoCapture(in_file)

    _, first_image = vidcap.read()
    prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

    flows = []
    count = 0
    while count < _images_per_file:
        _, image = vidcap.read()
        curr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        flow = convert_to_optical_flow(prev, curr, img_size)

        flows.append(flow)

        prev = curr
        count += 1

    return (np.array(flows) / 255.).astype(np.float16)


image_model = load_model('/path/to/VGG16/model/file')

"""
Change to fc2 if using classic VGG16 from keras
"""
transfer_layer = image_model.get_layer('dense2')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

transfer_values_size = K.int_shape(transfer_layer.output)[1]


def process_transfer(vid_names, labels):
    """
    Takes list of video file names and list of mapped labels, gets extracted and transformed video frames,
     makes VGG16 prediction on them and at last labels them
    Args:
        vid_names: video file directory path
        labels: video file name
    Returns:
    Yields VGG16 predictions from each video extracted frames with labels
    """
    count = 0

    tam = len(vid_names)

    while count < tam:

        video_name = vid_names[count]

        if labels[count] == [0, 1]:
            in_dir = dir_not_fight
        else:
            in_dir = dir_fight

        image_batch = get_frames(in_dir, video_name)

        transfer_values = image_model_transfer.predict(image_batch)

        labels1 = labels[count]

        aux = np.ones([_images_per_file, num_classes])

        labelss = labels1 * aux

        yield transfer_values, labelss

        count += 1


def make_files(n_files, names, labels, dataset_name):
    """
    Prepares training dataset and saves it in in h5 format, prints progress while doing that
    Args:
        dataset_name: dataset name that will be saved in h5 format
        labels: list of video file labels
        names: list of video file names
        n_files: File count for training set

    Returns:
    -
    """
    gen = process_transfer(names, labels)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]

    with h5py.File(f'{dataset_name}.h5', 'w') as f:
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]

        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)

        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)

        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:

            if numer == n_files:
                break

            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]

            print_progress(numer, n_files)

            numer += 1


"""
Train/Test split 80%/20%
"""
training_set = int(len(names) * 0.8)
test_set = int(len(names) * 0.2)

"""
Training/test file names and labels lists
"""
names_training = names[0:training_set]
names_test = names[training_set:]

labels_training = labels[0:training_set]
labels_test = labels[training_set:]

"""
Preparing train and test data
"""
make_files(training_set, names_training, labels_training, train_dataset_name)
make_files(test_set, names_test, labels_test, test_dataset_name)


def process_all_data(dataset_name):
    """
    Reads dataset h5 file, loads data and its labels, returns in arrays
    Args:
        dataset_name: name of the dataset that has to be processed

    Returns:
    Arrays with dataset data and labels
    """
    joint_transfer = []
    frames_num = _images_per_file
    count = 0

    with h5py.File(f'{dataset_name}.h5', 'r') as f:

        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch) / frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc], y_batch[count]])
        count = inc

    data = []
    target = []

    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))

    return data, target


data, target = process_all_data(train_dataset_name)
data_test, target_test = process_all_data(test_dataset_name)

"""
LSTM hyper params
"""
chunk_size = 2
n_chunks = 20
rnn_size = 512


"""
Creating model with LSTM layer and some dense layers to lower outgoing neurons till binary classification output
"""
model = Sequential()
model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(2))

"""
Final activation, compiling model with loss function and chosen optimizer
"""
from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.0005)
model.add(Activation('softmax'))
model.compile(loss=keras.losses.mean_squared_error, optimizer=opt, metrics=['accuracy'])

"""
Num of epochs to run with batch size
"""
epoch = 1000
batchS = 500

"""
Per epoch checkpoints setup, early stop callback to save some machine time, 
tensor board config for learning process logs and finally running learning process
"""
checkpoint = ModelCheckpoint(filepath='/path/to/save/each/checkpoint/vrn.model.{epoch:02d}.h5',
                             monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=80, verbose=1, mode='auto')
logs = TensorBoard(log_dir='/path/to/save//logs')
history = model.fit(np.array(data[0:SINGLE_CLASS_FILE_COUNT]), np.array(target[0:SINGLE_CLASS_FILE_COUNT]),
                    epochs=epoch,
                    validation_data=(
                    np.array(data[SINGLE_CLASS_FILE_COUNT:]), np.array(target[SINGLE_CLASS_FILE_COUNT:])),
                    batch_size=batchS, verbose=2, callbacks=[checkpoint, early, logs])
