import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pickle
import keras
from keras import layers, Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, Flatten, MaxPooling1D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import np_utils
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics

# function for extracting features
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


def noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise

    
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift(data, sampling_rate, shift_max, shift_direction):
    """
    shift the spectogram in a direction

    Parameters
    ----------
    data : np.ndarray, audio time series
    sampling_rate : number > 0, sampling rate
    shift_max : float, maximum shift rate
    shift_direction : string, right/both

    """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0

    return augmented_data


emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def load_data(test,dataset_path="dataset"): # give RAVDESS dataset location in place of dataset
    x, y = [], []

    data_path = os.path.join(dataset_path, "Actor_*", "*.wav")
    for file in glob.glob(data_path):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]

        data, sr = librosa.load(file)

        # Extract features from original data
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

        # Adding noise data
        n_data = noise(data, 0.001)
        n_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(n_feature)
        y.append(emotion)

        # adding shift data
        s_data = shift(data, sr, 0.25, 'right')
        s_feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(s_feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test, random_state=9)


x_train, x_test, y_train, y_test = load_data(0.25)

print("x_train shape:",x_train.shape)

print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# One-hot encode the labels using Keras
y_train_encoded = keras.utils.to_categorical(y_train_encoded, 8)
y_test_encoded = keras.utils.to_categorical(y_test_encoded, 8)

# Normalize the input features
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

print("x_train shape:",x_train.shape)

print("x_train",x_train)

import matplotlib.pyplot as plt
from collections import Counter

# Combine training and testing sets
x_combined = np.concatenate((x_train, x_test))
y_combined = y_train + y_test

# Plot a bar graph or histogram showing the number of samples in each emotion
emotion_counts_combined = Counter(y_combined)

emotions_list = list(emotions.values())
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

fig, ax = plt.subplots(figsize=(10, 6))

# Combined set
ax.bar(emotions_list, [emotion_counts_combined[emotion] for emotion in emotions_list], color=colors)
ax.set_title('Combined Emotion Distribution')
ax.set_ylabel('Number of Samples')
ax.set_xticklabels(emotions_list, rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("x_train shape[1]",x_train.shape[1])
# Build the model

model = Sequential()

model.add(Conv1D(256, 5, padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

# Flattening
model.add(Flatten())

model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(x_train, y_train_encoded, epochs=50, batch_size=32, validation_data=(x_test, y_test_encoded),
                    callbacks=[early_stopping])

model.save('ser_saved_model.h5')

accuracy = model.evaluate(x_test, y_test_encoded)[1]
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot the training history (accuracy and loss over epochs)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
