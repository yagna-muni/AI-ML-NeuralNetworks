import cv2
import numpy as np
import onnxruntime as ort
from io import BytesIO
import requests
import tensorflow as tf

cifar10_class_names  = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

img_urls = [
    'https://png.pngitem.com/pimgs/s/534-5342414_thumb-image-japan-airlines-plane-png-transparent-png.png',
    'https://cdn.pixabay.com/photo/2017/09/01/00/15/png-2702691__480.png',
    'https://cdn.pixabay.com/photo/2016/03/02/13/59/bird-1232416__480.png',
    'https://w7.pngwing.com/pngs/414/106/png-transparent-enzo-ferrari-sports-car-luxury-vehicle-ferrari-compact-car-car-performance-car.png'
]

for img_url in img_urls:
    response = requests.get(img_url)
    img_bytes = response.content
    img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (32, 32))  # Resize the input image to 32x32 pixels (CIFAR-10 input size)
    img_resized = img_resized.astype(np.float32) / 255.0  # Normalize the input to be between 0 and 1

    network_input = np.expand_dims(img_resized, axis=0)  # Inputs are of the form (<# images>, height, width, 3)

    network_sess = ort.InferenceSession('cifar_5_layer.onnx', providers=['CUDAExecutionProvider'])  # Load the network. Replace with 'CPUExecutionProvider' if using a CPU.

    inputName = network_sess.get_inputs()[0].name
    outputName = network_sess.get_outputs()[0].name

    result = network_sess.run([outputName], {inputName: network_input})  # Run the network on the input
    result = result[0]
    result = tf.nn.softmax(result).numpy()

    # Find the class with the highest probability
    predicted_class = np.argmax(result)
    predicted_class_name = cifar10_class_names[predicted_class]

    print(f"Image URL: {img_url}")
    print(f"Output activations: {np.round(result, 3)}")
    print(f"Predicted class: {predicted_class_name}\n")

