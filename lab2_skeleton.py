import tensorflow as tf
import keras
import numpy as np

"""
import tensorflow.keras
from tensorflow.keras.utile import to_categorical
"""

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255
num_pixels = X_train.shape[1] * X_train.shape[2]

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==5.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==5.0)[0]] = 1
y_test = y_new

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#To complete

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(2, input_dim=num_pixels, activation='softmax'), kernel_initializer="normal")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#epochs = 100
#batch_size = 128
model.fit(X_train, y_train, epochs=10)

