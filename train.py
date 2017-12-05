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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.datasets import fashion_mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm


def run(model_name, lr, optimizer, epoch, patience, batch_size):
    # Loading Datasets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    width = 28
    # Compute the bottleneck feature

    def get_features(MODEL, data=x_train):
        cnn_model = MODEL(
            include_top=False,
            input_shape=(width, width, 3),
            weights='imagenet')

        inputs = Input((width, width, 3))
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
                  X=x_train):
        # Fine-tune the model
        print("\n\n Fine tune " + model_name + " : \n")

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2)

        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        inputs = Input((width, width, 3))
        x = inputs
        cnn_model = MODEL(
            include_top=False,
            input_shape=(width, width, 3),
            weights='imagenet')
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(n_class, activation='softmax', name='predictions')(x)
        model = Model(inputs=inputs, outputs=x)

        # for layer in model.layers[:114]:
        #     layer.trainable = False

        try:
            model.load_weights(model_name + '.h5')
            print('Load ' + model_name + '.h5 successfully.')
        except:
            try:
                model.load_weights('fc_' + model_name + '.h5', by_name=True)
                print('Fail to load ' + model_name + '.h5, load fc_' +
                      model_name + '.h5 instead.')
            except:
                print(
                    'Start computing ' + model_name + ' bottleneck feature: ')
                features = get_features(MODEL, X)

                # Training models
                inputs = Input(features.shape[1:])
                x = inputs
                x = Dropout(0.5)(x)
                x = Dense(n_class, activation='softmax', name='predictions')(x)
                model_fc = Model(inputs, x)
                model_fc.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                h = model_fc.fit(
                    features,
                    y,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1)

                model_fc.save('fc_' + model_name + '.h5', 'w')

        print("\n " + "Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
        optimizer
        if optimizer == "Nadam":
            model.compile(
                optimizer=Nadam(lr=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        elif optimizer == "SGD":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                metrics=['accuracy'])

        class LossHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                # self.val_losses = []
                self.losses = []

            def on_epoch_end(self, batch, logs={}):
                # self.val_losses.append(logs.get("val_loss"))
                self.losses.append((logs.get('loss'), logs.get("val_loss")))

        history = LossHistory()

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(
            filepath=model_name + '.h5', verbose=1, save_best_only=True)
        h2 = model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            validation_data=val_datagen.flow(
                x_val, y_val, batch_size=batch_size),
            validation_steps=len(x_val) / batch_size,
            epochs=epoch,
            callbacks=[early_stopping, checkpointer, history])
        #         print(history.losses)
        # np.savetxt("val_losses_" + str(optimizer) + "_" + str(lr) + ".csv",
        #            history.losses)
        with open(model_name + ".csv", 'a') as f_handle:
            np.savetxt(f_handle, history.losses)
        # with open(model_name + "2.csv", 'a') as f_handle:
        #     np.savetxt(f_handle, model_name + str(optimizer) + "_" + str(lr))
        #     np.savetxt(f_handle, history.losses)
        # model.save(model_name + '.h5', 'w')

    # list_model = [Xception, InceptionV3, InceptionResNetV2]
    # list_name_model = ["Xception", "InceptionV3", "InceptionResNetV2"]
    #
    # for x in range(3):
    list_model = {
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "InceptionResNetV2": InceptionResNetV2
    }
    print(list_model[model_name])
    fine_tune(list_model[model_name], model_name, optimizer, lr, epoch,
              patience, batch_size, x_train)
    # fine_tune(list_model[0], list_name_model[0], optimizer, lr, epoch,
    #           patience, batch_size, X)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="Xception", type=str)
    parser.add_argument(
        "--lr", help="learning rate", default=0.0001, type=float)
    parser.add_argument(
        "--optimizer", help="optimizer", default="Nadam", type=str)
    parser.add_argument(
        "--epoch", help="Number of epochs", default=100, type=int)
    parser.add_argument(
        "--patience", help="Patience to wait", default=2, type=int)
    parser.add_argument(
        "--batch_size", help="Batch size", default=16, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.lr, args.optimizer, args.epoch, args.patience,
        args.batch_size)

