# MNIST Two-Layer Dense Network (Linear Classifier)

import tensorflow as tf
from keras.layers import Input, Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
from keras.datasets import cifar10

# Determine whether a CPU or GPU is being used by TensorFlow
device_name = tf.test.gpu_device_name()
print(device_name)

# Load MNIST datasets for training (60,000 exemplars) and testing (10,000 exemplars)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize inputs to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

num_epochs = 200                        # One epoch of training means processing each examplar in the training set exactly once.
batch_size = 100                        # Number of exemplars processed at once. Larger batches speed up computation but need more memory.
size_input = (32, 32, 3)
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of individual classes (here: 10 classes, one for each digit)

num_train_exemplars = x_train.shape[0]

# Build the model (computational graph)
cifar_model = Sequential(
    [Input(shape=size_input, name='input_layer'),
    Flatten(name='flat_layer'),         # Flatten the input, i.e., turn the 28x28 2D array into a 784-element vector
    Dense(size_output, activation='softmax', name='output_layer')])     # Output layer uses softmax activation, which is a good choice for classification tasks

cifar_4_layer = Sequential([
    Input(shape=size_input, name='input_layer'),
    Flatten(name='flat_layer'),
    Dense(512, activation='relu', name='hidden_layer_1'),
    Dense(256, activation='relu', name='hidden_layer_2'),
    Dense(size_output, activation='softmax', name='output_layer')])

cifar_4_layer_higher_neurons = Sequential([
    Input(shape=size_input, name='input_layer'),
    Flatten(name='flat_layer'),
    Dense(1024, activation='relu', name='hidden_layer_1'),
    Dense(512, activation='relu', name='hidden_layer_2'),
    Dense(size_output, activation='softmax', name='output_layer')])

cifar_5_layer = Sequential([
    Input(shape=size_input, name='input_layer'),
    Flatten(name='flat_layer'),
    Dense(1024, activation='relu', name='hidden_layer_1'),
    Dense(512, activation='relu', name='hidden_layer_2'),
    Dense(256, activation='relu', name='hidden_layer_3'),
    Dense(size_output, activation='softmax', name='output_layer')])

models = [cifar_4_layer, cifar_4_layer_higher_neurons, cifar_5_layer]

for model in models:
    layer = models.index(model) + 1
    print(f"Training variant {layer}...")
    model.summary()
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics='accuracy')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# The 5-layer model performs better with accuracy reaching around 96% because of more layers, more neurons and non-linear activation function 'reLU'. This helps
# network learn complex patterns and relationships in data.
