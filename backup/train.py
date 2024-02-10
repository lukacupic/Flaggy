
from siamese_network import build_siamese_model
import utils
import config
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.datasets import mnist
import numpy as np

def main():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    
    (pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
    (pairTest, labelTest) = utils.make_pairs(testX, testY)

    imgA = Input(shape=config.IMG_SHAPE)
    imgB = Input(shape=config.IMG_SHAPE)
    featureExtractor = build_siamese_model(config.IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    distance = Lambda(utils.euclidean_distance)([featsA, featsB])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]],
        labelTrain[:],
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS
    )

    model.save(config.MODEL_PATH)

if __name__ == "__main__":
	main()