import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def plot_history(h):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])
    plt.show()

csvTrain = pd.read_csv('sign-language-mnist\sign_mnist_train.csv')
csvTest = pd.read_csv('sign-language-mnist\sign_mnist_test.csv')

Trainlabels = csvTrain['label'].values
Testlabels = csvTest['label'].values

# show distribution
plt.figure(figsize = (18,8))
sns.countplot(x = Trainlabels)
plt.show()

# drop label from the images
csvTrain.drop('label', axis = 1, inplace = True)
csvTest.drop('label', axis = 1, inplace = True)

# reshaping the images and labels
Trainimages = csvTrain.values
Trainimages = np.array([np.reshape(i, (28, 28, 1)) for i in Trainimages])

Testimages = csvTest.values
Testimages = np.array([np.reshape(i, (28, 28, 1)) for i in Testimages])

# split training data 
Trainimages, Validationimages, Trainlabels, Validationlabels = train_test_split(Trainimages, Trainlabels, test_size = 0.3)

# normalize data
Trainimages = Trainimages / 255
ValidationImages = Validationimages / 255
Testimages = Testimages / 255

# vectorize label
labelBinrizer = LabelBinarizer()
Trainlabels = labelBinrizer.fit_transform(Trainlabels)
Validationlabels = labelBinrizer.fit_transform(Validationlabels)
Testlabels = labelBinrizer.fit_transform(Testlabels)

# build network
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

#history = model.fit(Trainimages, Trainlabels, validation_data = (Validationimages, Validationlabels), epochs=50, batch_size=128)

# save weights
model.save_weights("weights.h5")

model.load_weights("weights.h5")

# plot graph
plot_history(history)

# verify with test data
predictions = model.predict(Testimages)

