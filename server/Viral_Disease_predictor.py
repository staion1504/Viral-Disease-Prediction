from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from flask import Flask, request, jsonify, redirect, url_for, render_template
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from collections import Counter


import h5py 
import pickle

from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

vocab_size = 1396
max_length = 10

model = load_model("./ADASYN_MODEL_4CLASS.h5") 

# Load the one-hot encoder
with open('tokenizer.pickle', 'rb') as handle:
    word_index = pickle.load(handle)


tokenizer = Tokenizer(char_level=False)
tokenizer.word_index = word_index

# print(word_index)

def divide_into_3mers(input_string):
    
    def generate_3mers(s):
        return [s[i:i+3] for i in range(len(s) - 2)]

    
    mers = generate_3mers(input_string)
    mer_freq = Counter(mers)
    sorted_mers = sorted(mer_freq.items(), key=lambda x: x[1], reverse=True)
    new_string = ' '.join([mer[0] for mer in sorted_mers])

    return new_string



def preprocess_sequence(sequence):
    sequencesNew =divide_into_3mers(sequence)
    encoded_sequences = tokenizer.texts_to_sequences([sequencesNew]) 
    padded_sequence = pad_sequences(encoded_sequences, maxlen=max_length, padding='post',truncating='post')
    
    return padded_sequence


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = data['sequence']
    preprocessed_sequence = preprocess_sequence(sequence)
    # print(preprocessed_sequence)
    prediction = model.predict(preprocessed_sequence)
    print(np.argmax(prediction[0])+1)
    return jsonify({'prediction':int(np.argmax(prediction[0])+1)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run the Flask app on port 5000

