

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import ADASYN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding,LSTM
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

from google.colab import drive
drive.mount('/content/drive')

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    return data

def remove_non_ATCG_characters(sequence):
    valid_characters = {'A', 'C', 'T', 'G'}
    filtered_sequence = ''
    for char in sequence:
        if char not in valid_characters:
            char = 'N'
        filtered_sequence += char
    return filtered_sequence

def divide_into_3mers(input_string):
    def generate_3mers(s):
        return [s[i:i+3] for i in range(len(s) - 2)]
    mers = generate_3mers(input_string)
    mer_freq = Counter(mers)
    sorted_mers = sorted(mer_freq.items(), key=lambda x: x[1], reverse=True)
    new_string = ' '.join([mer[0] for mer in sorted_mers])
    return new_string

def process_sequences(sequence_list):
    sequences_new = []
    for seq in sequence_list:
        sequences_new.append(divide_into_3mers(seq))
    return sequences_new

def get_num_classes(datalabels):
    unique_classes = set(datalabels)
    num_classes = len(unique_classes)
    return num_classes

def tokenize_sequences(sequences):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(sequences)
    encoded_sequences = tokenizer.texts_to_sequences(sequences)
    vocab_size = len(tokenizer.word_index)
    return encoded_sequences, vocab_size

def encode_labels(datalabels, num_classes):
    encoder = LabelEncoder()
    encoder.fit(datalabels)
    encoded_labels = encoder.transform(datalabels)
    encoded_labels_to_categorical = to_categorical(encoded_labels, num_classes=num_classes)
    return  encoded_labels_to_categorical

def preprocess_data(data):
    datalabels = data["Class"].tolist()
    Sequence = data["DnaSequence"].tolist()
    Sequence_with_ACTGN = [remove_non_ATCG_characters(seq) for seq in Sequence]
    mer_3_sequence=process_sequences(Sequence_with_ACTGN)
    num_classes = get_num_classes(datalabels)
    encoded_sequences, vocab_size = tokenize_sequences(mer_3_sequence)
    encoded_labels_to_categorical = encode_labels(datalabels, num_classes)
    return encoded_sequences,encoded_labels_to_categorical,num_classes,vocab_size

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,stratify=y,random_state=random_state)
    return X_train, X_test, y_train, y_test

def oversample_data(X_train, y_train):
    ada = ADASYN(sampling_strategy='minority')
    X_train_res, y_train_res = ada.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def pad_encoded_sequences(encoded_sequences,max_length=10):
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

def train_model(X_train, y_train,vocab_size,num_classes,max_length):
    embeded_vector_size = 50
    model = Sequential()
    model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_length,name="embedding"))
    model.add(LSTM(80))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train,epochs=10, batch_size=16, validation_split=0.2,shuffle=True)
    return model

def evaluate_model(model, X_test, y_test, num_classes):
    model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    x = list(range(num_classes))
    y = list(range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x, yticklabels=y)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main(file_path):
    data = load_data(file_path)
    X, y,num_classes,vocab_size = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_padded_sequences = pad_encoded_sequences(X_train, max_length=10)
    X_test_padded_sequences = pad_encoded_sequences(X_test, max_length=10)
    X_train_res, y_train_res = oversample_data(X_train_padded_sequences, y_train)
    print(len(X_train_res))
    model = train_model(X_train_res,y_train_res,vocab_size,num_classes,max_length=10)
    evaluate_model(model, X_test_padded_sequences, y_test,num_classes)

if __name__ == "__main__":
    file_path = "give here filepath"

