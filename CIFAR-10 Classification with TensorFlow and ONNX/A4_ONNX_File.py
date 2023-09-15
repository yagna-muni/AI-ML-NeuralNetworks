import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import tf2onnx

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

size_input = (32, 32, 3)
size_output = 10
num_epochs = 200
batch_size = 100

# Create the best-performing network
cifar_5_layer = Sequential([
    Input(shape=size_input, name='input_layer'),
    Flatten(name='flat_layer'),
    Dense(1024, activation='relu', name='hidden_layer_1'),
    Dense(512, activation='relu', name='hidden_layer_2'),
    Dense(256, activation='relu', name='hidden_layer_3'),
    Dense(size_output, activation='softmax', name='output_layer')])

cifar_5_layer.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics='accuracy')

cifar_5_layer.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# Define the input signature
spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
# Convert the trained Keras model to ONNX format using input_signature
tf2onnx.convert.from_keras(cifar_5_layer,  input_signature=spec, output_path='cifar_5_layer.onnx')
