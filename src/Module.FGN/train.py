import argparse

import config
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from data import DataGenerator
from keras.callbacks import CSVLogger, LearningRateScheduler
from keras.optimizers import SGD
from model import flow_gated_network_model


def plot(H):
    """
    Plots the training loss and accuracy and save the fig
    Args:
        H: history object from training
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, config.EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, config.EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, config.EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, config.EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("plots/" + config.MODEL_NAME + ".png")
    plt.show()


def scheduler(epoch: int):
    """
    Scheduler for learning rate optimization
    Args:
        epoch: epohc number

    Returns:
        Numpy array
    """
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.7)
    return K.get_value(model.optimizer.lr)


class SaveModelCallback(keras.callbacks.Callback):
    """
    Saves model every on epoch end
    Args:
        model: model to save
    """
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, save_dir="logs/"):
        """

        Args:
            epoch: epoch number
            save_dir: path to save firectory
        """
        self.model_to_save.save(save_dir + config.MODEL_NAME + '_at_epoch_%d.h5' % (epoch + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="name of the model")
    ap.add_argument("-d", "--dataset", required=True, help="path to dataset folder")
    ap.add_argument("-s", "--size", required=False, default=64, help="frame size")
    ap.add_argument("-fn", "--frames_number", required=False, default=64, help="number of frames")
    ap.add_argument("-e", "--epochs", required=False, default=30, help="number of epochs")
    ap.add_argument("-bs", "--batch_size", required=False, default=8, help="batch size")
    ap.add_argument("-wn", "--workers_number", required=False, default=16, help="number of workers")
    args = vars(ap.parse_args())

    config.SIZE = int(args["size"])
    config.FRAMES_NO = int(args["frames_number"])

    config.EPOCHS = int(args["epochs"])
    config.WORKERS_NO = int(args["workers_number"])
    config.BATCH_SIZE = int(args["batch_size"])

    config.MODEL_NAME = args["model"]
    config.DATASET_DIR = args["dataset"]

    model = flow_gated_network_model()
    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    reduce_lr = LearningRateScheduler(scheduler)
    check_point = SaveModelCallback(model)
    csv_logger = CSVLogger('logs/logs.csv', separator=',', append=True)

    callbacks_list = [check_point, csv_logger, reduce_lr]

    train_generator = DataGenerator(directory='{}/train'.format(config.DATASET_DIR),
                                    batch_size=config.BATCH_SIZE,
                                    data_augmentation=True)

    val_generator = DataGenerator(directory='{}/val'.format(config.DATASET_DIR),
                                  batch_size=config.BATCH_SIZE,
                                  data_augmentation=False)

    history = model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        callbacks=callbacks_list,
        verbose=1,
        epochs=config.EPOCHS,
        workers=config.WORKERS_NO,
        max_queue_size=4,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator))

    plot(history)
