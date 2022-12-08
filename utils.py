import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import cv2
import numpy as np


with open('tokenizer.pkl', 'rb') as token:
    tokenizer = pickle.load(token)

# util functions


def get_sequences(tokenizer, descriptions):
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded = pad_sequences(sequences, truncating='post',
                           padding='post', maxlen=50)
    return padded


def image_preprocess(filename, input_size: list):
    image = tf.keras.preprocessing.image.load_img(
        filename, grayscale=False, color_mode="rgb", target_size=input_size, interpolation="nearest")
    array = tf.keras.preprocessing.image.img_to_array(image)
    return np.array(array)


# models
model_multi = tf.keras.models.load_model("multi_input")
model_txt = tf.keras.models.load_model("redo")
model_cv = tf.keras.models.load_model('CV_model')

index_to_class = {0: 'Kitchen & Dining',
                  1: 'Baby Care',
                  2: 'Watches',
                  3: 'Home Furnishing',
                  4: 'Beauty and Personal Care',
                  5: 'Home Decor & Festive Needs',
                  6: 'Computers'}
