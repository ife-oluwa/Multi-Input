import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


with open('tokenizer.pkl', 'rb') as token:
    tokenizer = pickle.load(token)


def get_sequences(tokenizer, descriptions):
    sequences = tokenizer.texts_to_sequences(descriptions)
    padded = pad_sequences(sequences, truncating='post',
                           padding='post', maxlen=50)
    return padded


model = tf.keras.models.load_model("redo")


index_to_class = {0: 'Kitchen & Dining',
                  1: 'Baby Care',
                  2: 'Watches',
                  3: 'Home Furnishing',
                  4: 'Beauty and Personal Care',
                  5: 'Home Decor & Festive Needs',
                  6: 'Computers'}
