import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the trained model
model = keras.models.load_model('image_sr_model.h5')

# Load the low-resolution input image
lr_image = Image.open('input_image.jpg')

# Upscale the image by a factor of 8
lr_image = lr_image.resize((lr_image.width // 8, lr_image.height // 8))

# Convert the image to a numpy array and normalize its values
lr_array = np.asarray(lr_image, dtype=np.float32) / 255.0

# Add a batch dimension to the input
lr_array = np.expand_dims(lr_array, axis=0)

# Generate the high-resolution output image
sr_array = model.predict(lr_array)

# Convert the output array to an image and save it
sr_image = Image.fromarray(np.uint8(sr_array[0] * 255.0))
sr_image.save('output_image.jpg')
