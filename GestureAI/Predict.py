import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


model = keras.Sequential([
Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28, 1) ),
MaxPooling2D(pool_size = (2, 2)),

Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
MaxPooling2D(pool_size = (2, 2)),

Conv2D(64, kernel_size = (3, 3), activation = 'relu'),
MaxPooling2D(pool_size = (2, 2)),

Flatten(),
Dense(128, activation = 'relu'),
Dropout(0.20),
Dense(24, activation = 'softmax')
])

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

model.load_weights("weights.h5")

image = cv2.imread("b.jpg", 0)   
image = cv2.resize(image, (28, 28))

cv2.imshow("sign", image)
cv2.waitKey(0)
image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


image = (np.expand_dims(image,0))
image = np.array([np.reshape(i, (28, 28, 1)) for i in image])

predictions = model.predict(image)

print(np.argmax(predictions[0]))

i = 0