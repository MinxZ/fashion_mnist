from __future__ import absolute_import, division, print_function

import argparse
import os
import random
from datetime import datetime

import cv2
import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.datasets import fashion_mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.vis_utils import model_to_dot
from scipy import misc
from tqdm import tqdm


def run(model_name, lr, optimizer, epoch, patience, batch_size, weights, test=None):
    def load_data(height=128, width=128, use_imagenet=None):
        num_classes = 10
        (train, train_l), (test, test_l) = fashion_mnist.load_data()

        y = keras.utils.to_categorical(train_l, num_classes)
        y_test = keras.utils.to_categorical(test_l, num_classes)

        if use_imagenet:
            train = (train.reshape((-1, 28, 28)) / 255. - 0.5) * 2
            train = np.array(
                [misc.imresize(x, (height, width)) for x in tqdm(iter(train))])
            test = (test.reshape((-1, 28, 28)) / 255. - 0.5) * 2
            test = np.array(
                [misc.imresize(x, (height, width)) for x in tqdm(iter(test))])

            x = np.stack((train, train, train), axis=3)
            x_test = np.stack((test, test, test), axis=3)
        else:
            x = (train.reshape((train.shape[0], 28, 28, 1)) / 255. - 0.5) * 2
            x_test = (test.reshape(
                (test.shape[0], 28, 28, 1)) / 255. - 0.5) * 2

        # devide into train and validation
        dvi = int(train.shape[0] * 0.9)
        x_train = x[:dvi, :, :, :]
        y_train = y[:dvi, :]
        x_val = x[dvi:, :, :, :]
        y_val = y[dvi:, :]

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_val.shape[0], 'validation samples')
        print(x_test.shape[0], 'test samples')

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # Loading Datasets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    n_class = y_test.shape[1]
    input_shape = x_train.shape[1:]

    if weights == 'None':
        weights = None
        print("\n Training on " + model_name + ": \n")
    else:
        weights = 'imagenet'
        print("\n Fine tune on " + model_name + ": \n")

    print('Weights are ' + str(weights))

    def get_features(MODEL, data=x_train):
        cnn_model = MODEL(
            include_top=False, input_shape=input_shape, weights=weights)

        inputs = Input(input_shape)
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs, x)

        features = cnn_model.predict(data, batch_size=32, verbose=1)
        return features

    def fine_tune(MODEL,
                  model_name,
                  optimizer,
                  lr,
                  epoch,
                  patience,
                  batch_size,
                  weights,
                  X=x_train,
                  test=None):
        # Fine-tune the model

        from random_eraser import get_random_eraser
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_h=60, pixel_level=True))

        val_datagen = ImageDataGenerator()

        inputs = Input(input_shape)
        x = inputs
        cnn_model = MODEL(
            include_top=False, input_shape=input_shape, weights=None)
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='sim')(x)
        x = Dropout(0.5)(x)
        x = Dense(n_class, activation='softmax', name='predictions')(x)
        model = Model(inputs=inputs, outputs=x)

        # Loading weights
        try:
            model.load_weights(model_name + '.h5')
            print('Load ' + model_name + '.h5 successfully.')
        except:
            if weights == 'imagenet':
                print(
                    'Start computing ' + model_name + ' bottleneck feature: ')
                features = get_features(MODEL, X)

                # Training models
                inputs = Input(features.shape[1:])
                x = inputs
                x = Dropout(0.5)(x)
                x = Dense(128, activation='relu', name='sim')(x)
                x = Dropout(0.5)(x)
                x = Dense(n_class, activation='softmax', name='predictions')(x)
                model_fc = Model(inputs, x)
                model_fc.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                h = model_fc.fit(
                    features,
                    y_train,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1)
                model_fc.save('fc_' + model_name + '.h5')
                model.load_weights('fc_' + model_name + '.h5', by_name=True)

        print("Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
        if optimizer == "Adam":
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
        elif optimizer == "SGD":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                metrics=['accuracy'])

        if not test:
            datagen.fit(x_train)
            val_datagen.fit(x_val)

            class LossHistory(keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.losses = []

                def on_epoch_end(self, batch, logs={}):
                    self.losses.append((logs.get('loss'),
                                        logs.get("val_loss")))

            history = LossHistory()
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=patience, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(
                filepath=model_name + '.h5', verbose=0, save_best_only=True)
            reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
            if optimizer == "Adam":
                callbacks = [history, early_stopping, checkpointer]
            else:
                callbacks = [history, early_stopping, checkpointer, reduce_lr]
            h = model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) / batch_size,
                validation_data=val_datagen.flow(
                    x_val, y_val, batch_size=batch_size),
                validation_steps=len(x_val) / batch_size,
                epochs=epoch,
                callbacks=callbacks)
            return h
        else:
            print('Evalute on test set')
            val_datagen.fit(x_test)
            score = model.evaluate_generator(
                val_datagen.flow(x_test, y_test, batch_size=batch_size),
                len(x_test) / batch_size)
            print(score)
            return score

    list_model = {
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2,
        "VGG16": VGG16,
        "MobileNet": MobileNet
    }
    fine_tune(list_model[model_name], model_name, optimizer, lr, epoch,
              patience, batch_size, weights, x_train, test)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="Xception", type=str)
    parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
    parser.add_argument(
        "--optimizer", help="Optimizer to use", default="Adam", type=str)
    parser.add_argument(
        "--epoch", help="Number of epochs", default=1e5, type=int)
    parser.add_argument(
        "--batch_size", help="Batch size", default=64, type=int)
    parser.add_argument(
        "--patience",
        help="Patience to wait to early stopping",
        default=7,
        type=int)
    parser.add_argument(
        "--weights",
        help="to use pretrained weights or not",
        default='imagenet',
        type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.lr, args.optimizer, args.epoch, args.patience,
        args.batch_size, args.weights)

