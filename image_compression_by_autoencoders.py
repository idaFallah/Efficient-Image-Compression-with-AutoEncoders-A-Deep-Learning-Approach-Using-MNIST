# importinng libs

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

tf.__version__

# loading the dataset & splitting it into train/ test sets

from tensorflow.keras.datasets import mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

x_train.shape, y_train.shape

x_test.shape, y_test.shape

x_train[0], y_train[0]

# visualizing images

i = np.random.randint(0, 59999)
print(y_train[i])
plt.imshow(x_train[i], cmap='gray')

# creating grid to show more pictures
width = 10
height = 10
fig, axes = plt.subplots(height, width, figsize=(15,15))
axes = axes.ravel()  # matrix of (10, 10) -> vector of [100]
print(axes.shape)
for i in np.arange(0, width * height):
  index = np.random.randint(0, 59999)
  axes[i].imshow(x_train[index], cmap='gray')
  axes[i].set_title(y_train[index], fontsize=8)
  axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

# preprocessing the images

x_train[0].min(), x_train[0].max()

x_train = x_train / 255
x_test = x_test / 255

x_train[0].min(), x_train[0].max()

x_train.shape, x_test.shape

# we have to flatten the matrix of images to vectors to set them as inputs

x_train.shape[0], x_train.shape[1], x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_train.shape  # 0 -> number of pictures, 1 -> number of pixels

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_test.shape

# building & training the linear AutoEncoder

# 784 input = size of the images, + we have the same amount of units in output layer
# 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784

autoencoder = Sequential()

#Encoding layers
autoencoder.add(Dense(units=128, activation='relu', input_dim=784))
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=32, activation='relu'))  # encoded image(compressed layer)

#Decoding layers
autoencoder.add(Dense(units=64, activation='relu'))
autoencoder.add(Dense(units=128, activation='relu'))
autoencoder.add(Dense(units=784, activation='sigmoid'))  # simoid cause after normalization the value of images are between 0 & 1 which is compatible with sigmoid func.

autoencoder.summary( )

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train, x_train, epochs=50)  # both are x_train cause we are comparing the original image with the decoded image which is supposed to be the same

autoencoder.input

autoencoder.summary()

# encoding the image

autoencoder.input

autoencoder.get_layer('dense_3').output

encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('dense_3').output)

encoder.summary()

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')

x_test[0].shape

x_test[0].reshape(1, -1).shape

encoded_image = encoder.predict(x_test[0].reshape(1, -1))

encoded_image

encoded_image.shape

plt.imshow(encoded_image.reshape(8, 4), cmap='gray')

# decoding the picture

input_layer_for_decoder = Input(shape=(32,))
decoder_layer1 = autoencoder.layers[4]
decoder_layer2 = autoencoder.layers[5]
decoder_layer3 = autoencoder.layers[6]
decoder.Model(inputs=input_layer_for_decoder, outputs=decoder_layer3(decoder_layer2(decoder_layer1(input_layer_for_decoder))))

decoded_image = decoder.predict(encoded_image)

decoded_image.shape

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')

plt.imshow(decoded_image.reshape(28, 28), cmap='gray')

# testing the test set

x_test.shape[0]

n_images = 10
test_images = np.random.randint(0, x_test.shape[0] - 1, size = n_images)
print(test_images)
plt.figure(figsize=(18, 18))
for i, image_index in enumerate(test_images):  # enumerate for when the i definig the loop is calculated randomly and differs each time
  print(i, image_index)
  #original image
  ax = plt.subplot(10, 10, i + 1)
  plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
  plt.xticks(())  # to hide the excess numbers
  plt.yticks(())

  #coded images
  ax = plt.subplot(10, 10, i + 1 + n_images)  # to be represented in a different order
  encoded_image = encoder.predict(x_test[image_index].reshape(1, -1))
  plt.imshow(encoded_image.reshape(8, 4), cmap='gray')
  plt.xticks(())
  plt.yticks(())

  #decoded images
  ax = plt.subplot(10, 10, i + 1 + n_images * 2)
  plt.imshow(decoder.predict(encoded_image).reshape(28, 28), cmap='gray')
  plt.xticks(())
  plt.yticks(())













