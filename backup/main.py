from mimetypes import init
import os
import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from tensorflow import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import pathlib
from datetime import datetime
import logging

FLAGGY_DIR = '/media/luka/89683ab3-c0a6-4837-be8c-03edb0c1d685/Projects/Programming/Flaggy/2022'


class ModelSaver(keras.callbacks.Callback):
    def __init__(self, class_name):
        self.class_name = class_name

    def on_train_end(self, logs=None):
        print('Saving model checkpoint...')
        self.model.save(FLAGGY_DIR + '/models/model_' + self.class_name + '.h5')


@tf.autograph.experimental.do_not_convert
def main():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    dataset_name = 'flags-diverse-demo'
    dataset_url = 'file://' + FLAGGY_DIR + '/dataset/' + dataset_name + ".tar.gz"
    data_dir = tf.keras.utils.get_file(
        dataset_name, origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    img_height, img_width = 256, 256

    try:
        model_path = os.path.join(FLAGGY_DIR + '/models/', 'model_' + dataset_name + '.h5')
        resnet_model = keras.models.load_model(model_path)
        print('Found an existing H5 model.')
    except:
        print('Model not found. Initializing the training process...')

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode='categorical',
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=16)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode='categorical',
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=16)

        resnet_model = Sequential()

        pretrained_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(256, 256, 3),
            pooling='avg',
            classes=5,
            weights='imagenet'
        )

        for layer in pretrained_model.layers:
            layer.trainable = False

        resnet_model.add(pretrained_model)
        resnet_model.add(Flatten())
        resnet_model.add(Dropout(0.2))
        resnet_model.add(Dense(512, activation='relu'))
        resnet_model.add(Dense(5, activation='softmax'))

        resnet_model.summary()

        # train
        callbacks = [ModelSaver(dataset_name), TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)]
        resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        resnet_model.fit(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=25)

    # multi inference
    for image_name in sorted(os.listdir(FLAGGY_DIR + '/test_images/')):
        image_path = os.path.join(FLAGGY_DIR + '/test_images/', image_name)

        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (img_height, img_width))

        x = np.expand_dims(image_resized, axis=0)
        y_prob = resnet_model.predict(x)

        output_class = np.argmax(y_prob)
        print(image_path)
        print("The predicted class is", output_class)
        print()

    # single inference
    # path = '/media/luka/89683ab3-c0a6-4837-be8c-03edb0c1d685/Projects/Programming/Flaggy/2022/test_images/it2.jpg'
    # image = cv2.imread(path)
    # image_resized = cv2.resize(image, (img_height, img_width))

    # x = np.expand_dims(image_resized, axis=0)
    # y_prob = resnet_model.predict(x)

    # output_class = np.argmax(y_prob)
    # print("The predicted class is", output_class)


if __name__ == "__main__":
    main()
