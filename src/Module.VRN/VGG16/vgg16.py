#!/usr/bin/env python
# coding: utf-8

"""
Python script for learning base VGG16 from ground up or Keras version with imagenet weights for fine-tuning
(for binary classification)
- Jakub Kulaszewicz
"""

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import tensorflow as tf

"""
GPU session config
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

"""
Dataset load-up
"""
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="path/to/train/data",
                                       target_size=(224, 224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="path/to/validation/data",
                                      target_size=(224, 224))

"""
Uncomment if you want fine-tune VGG16 with imagenet weights
"""
# model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# # Freeze four convolution blocks
# for layer in model.layers[:15]:
#     layer.trainable = False
# # Make sure you have frozen the correct layers
# for i, layer in enumerate(model.layers):
#     print(i, layer.name, layer.trainable)


"""
Comment out if you want fine-tune VGG16 with imagenet weights 
"""
model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

"""
Closing layers for fine-tuning VGG16
Uncomment if you want fine-tune VGG16 with imagenet weights
"""
# x = model.output
# x = Flatten()(x)
# x = Dense(units=4096,activation="relu")(x)
# x = Dense(units=4096,activation="relu")(x)
# x = Dense(units=2,activation="softmax")(x)
#
# model = Model(inputs=model.input, outputs=x)

"""
Base VGG16 closing layers
Comment out if you want fine-tune VGG16 with imagenet weights
"""
model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="sigmoid"))

"""
Learning related configuration
"""
from tensorflow.keras.optimizers import SGD

opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpoint = ModelCheckpoint(filepath='/path/to/save/checkpoints/vgg16.model.{epoch:02d}.h5',
                             monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
logs = TensorBoard(log_dir='/path/to/save/logs')
hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata, validation_steps=10,
                           epochs=100, callbacks=[checkpoint, early, logs])
