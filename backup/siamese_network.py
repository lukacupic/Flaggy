# import the necessary packages
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D

def build_siamese_model(input_shape, embedding_dim=48):
	inputs = Input(input_shape)

	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)
    
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)

	pooled_output = GlobalAveragePooling2D()(x)
	outputs = Dense(embedding_dim)(pooled_output)
	model = Model(inputs, outputs)
    
	return model