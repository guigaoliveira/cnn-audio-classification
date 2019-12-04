from comet_ml import Experiment
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import os
from scipy.io import wavfile as wav
import struct
import matplotlib.pyplot as plt
import matplotlib
import librosa.display
import librosa
import pandas as pd
import numpy as np
import IPython.display as ipd

# Comet variables
API_KEY = "<HIDDEN>"
PROJECT = "test"
WORKSPACE = "guigaoliveira"

# Create Comet Experiment
experiment = Experiment(api_key=API_KEY,
                        project_name=PROJECT, workspace=WORKSPACE)

experiment.add_tag('ConvNet')

# params
params = {
    'random_state': 42,
    'epochs': 100,
    'batch_size': 128,
    'dropout': 0.4,
    'max_pad_length': 174
}

NUM_ROWS = 40
NUM_COLUMNS = 174
NUM_CHANNELS = 1

experiment.log_parameters(params)

matplotlib.use('agg')


def extract_audio_features(file_name):

    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = params['max_pad_length'] - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs


def build_model_graph(input_shape=(NUM_ROWS,
                                   NUM_COLUMNS,
                                   NUM_CHANNELS)):

    model = Sequential()
    # Layer 1
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(
        NUM_ROWS, NUM_COLUMNS, NUM_CHANNELS), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    # Layer 2
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    # Layer 3
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    # Layer 4
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')

    return model


# Set the path to the full UrbanSound dataset
fulldatasetpath = 'UrbanSound8K/audio/'

metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

features = []

for index, row in metadata.iterrows():

    file_name = os.path.join(os.path.abspath(
        fulldatasetpath), 'fold'+str(row["fold"])+'/', str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_audio_features(file_name)
    features.append([data, class_label])


# Features --> dataframe
feat_df = pd.DataFrame(features, columns=['feature', 'class_label'])

# Convert features and labels to numpy arrays
X = np.array(feat_df.feature.tolist())
y = np.array(feat_df.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, yy, test_size=0.2, random_state=params['random_state'])

x_train = x_train.reshape(
    x_train.shape[0], NUM_ROWS, NUM_COLUMNS, NUM_CHANNELS)
x_test = x_test.reshape(x_test.shape[0], NUM_ROWS, NUM_COLUMNS, NUM_CHANNELS)

num_labels = yy.shape[1]

# Build model
model = build_model_graph()

# Fit model
model.fit(x_train, y_train, batch_size=params['batch_size'],
          epochs=params['epochs'], validation_data=(x_test, y_test), verbose=1)

score = model.evaluate(x_train, y_train, verbose=0)
print('train accuracy: {}'.format(score))
# experiment.log_metric("train_acc", score)

score = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy: {}'.format(score))
# experiment.log_metric("val_acc", score)
