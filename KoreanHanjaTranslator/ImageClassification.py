import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.tensorboard as tf
import Images
import DataSet

dataset = DataSet.KoreanHanjaCharacterDataset(r'HanjaCharacters.csv', Images)
image_count = len(dataset)
print(image_count)

sunflower_url = DataSet.KoreanHanjaCharacterDataset.__init__(r'HanjaCharacters.csv', Images)
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)