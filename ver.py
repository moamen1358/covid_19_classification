import PIL
import cv2
import numpy as np
import tensorflow as tf

# Create a dictionary to store library names and their versions
libraries = {
    "PIL": PIL.__version__,
    "OpenCV": cv2.__version__,
    "NumPy": np.__version__,
    "TensorFlow": tf.__version__
}

# Iterate over the dictionary and print the output
for name, version in libraries.items():
    print(f"{name} == {version}")
