import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

img_dir = 'dataset/'

# classifying our dataset into two variables
no_tumour_img = os.listdir(img_dir+'no/')
yes_tumour_img = os.listdir(img_dir+'yes/')

dataset = []
label = []

INPUT_SIZE = 64

# reading and appending all the jpg images with a label which represents the classification
for i, img_name in enumerate(no_tumour_img):
    if(img_name.split('.')[1] == 'jpg'):
        img = cv2.imread(img_dir+'no/'+img_name)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(img))
        label.append(0)

for i, img_name in enumerate(yes_tumour_img):
    if(img_name.split('.')[1] == 'jpg'):
        img = cv2.imread(img_dir+'yes/'+img_name)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(img))
        label.append(1)

# converting our dataset into a numpy array for us to work
dataset = np.array(dataset)
label = np.array(label)

# splitting our dataset into test and train
X_train, x_test, Y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)


X_train = normalize(X_train, axis=1)
x_test = normalize(x_test, axis=1)

Y_train = to_categorical(Y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train,
batch_size=16,
verbose=1,
epochs=10,
validation_data=(x_test, y_test),
shuffle=False)

model.save('BrainTumour10EpochsCategorical.h5')

