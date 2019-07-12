from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure() # Create new figure in matploblib.pyplot
# plt.imshow(train_images[0]) # Show images
# plt.colorbar() # Adds a colorbar to the plot
# plt.grid(False) # Configure if showing grid lines or not
# plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building the neural network requires configuring the layers of the model and building it

# Basic building block of a neural network is a layer
# Layers extract representations from the data fed into them
# Most of deep learning consists of chaining together layers
# Most layers have parameters that are learned during training

# Keras is a high level API to build and train deep learning models
# Can put these together and extend functionality

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flattens from 2d array to on 1d array
    keras.layers.Dense(128, activation=tf.nn.relu),  # 128 nodes or neuron densely connected neural layer
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 10 node or neuron softmax layer. Returns array  of 10 probability

# Input shape has to be defined on the first layer in sequential model


# Compile step below:
# Optimizer: This is how the the model is updated based on data it sees and loss function

# Loss function: measures how accurate the model is during training.
# Goal is to minimize this function to steer the model in right direction

# Metrics: self explanatory. This one measures how many are labeled correctly
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


#####
predictions = model.predict(test_images)
predictions[0]
#
np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)  # No grid lines
    plt.xticks([])  # Get or set current tick and labels from the x-axis
    plt.yticks([])  # Set y-axis

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)  # Returns max in the array
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})". format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")  # Make bar plot

    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 0
# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

img = test_images[0]
img = (np.expand_dims(img,0))

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
