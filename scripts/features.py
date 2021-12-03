from librosa.util.utils import frame
import pandas as pd
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


SILENCE_THRESHOLD = .01
RATE =24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10

def to_mfcc(wav):
    return librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)

def segments(mfccs, labels):
    segments = []
    seg_labels = []
    for mfcc, label in zip(mfccs, labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start+1) * COL_SIZE])
            seg_labels.append(label)
    return segments, seg_labels

def get_audio(file:str):
    data, sr = librosa.load(file)
    return librosa.core.resample(y=data, orig_sr=sr, target_sr=RATE, scale=True)

def remove_silence(wav, threshold = 0.04, chunk=5000):
    frame_list = []
    for x in range(int(len(wav) / chunk)):
        if(np.any(wav[chunk * x:chunk * (x + 1)] >= threshold) or np.any(wav[chunk * x: chunk * (x+1)] <= -threshold)):
            frame_list.extend([True] * chunk)
        else:
            frame_list.extend([False] * chunk)
    frame_list.extend((len(wav) - len(frame_list)) * [False])
    return wav[frame_list]

def normalize(mfcc):
    norm = MinMaxScaler()
    return norm.fit_transform(np.abs(mfcc))

def get_features(df, test_size = 0.2):
    feats = []
    for idx, row in df.iterrows():
        audio = get_audio(df.loc[idx,['Audio']][0])
        mute_audio = remove_silence(audio)
        features = to_mfcc(mute_audio)
        feats.append(features)

    y = df.loc[:,['spanish']]
    feats, y = segments(feats, y)
    return train_test_split(feats,y,test_size=test_size)

def save_state(model, filename):
    print('Saving module')
    model.save(f'../{filename}.h5')

def train_model(x_train, y_train, x_val, y_val, batch_size=128):
    rows = x_train[0].shape[0]
    cols = x_train[0].shape[1]
    val_rows = x_val[0].shape[0]
    val_cols = x_val[0].shape[1]
    num_classes = len(y_train[0])

    input_shape = (rows, cols, 1)

    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_val = x_val.reshape(x_val.shape[0], val_rows, val_cols, 1)

    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'training samples')

    modelo = Sequential()
    modelo.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                data_format='channels_last',
                input_shape=input_shape))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.25))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(num_classes, activation='softmax'))

    modelo.compile(loss='binary_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])

    board = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None)

    gen = ImageDataGenerator(width_shift_range=0.05)
    stop = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    modelo.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / 32
            , epochs=EPOCHS, 
            callbacks=[stop, board], validation_data=(x_val, y_val))
    return modelo

if __name__ == '__main__':
    filename = './data.csv'
    df = pd.read_csv(filename)
    x_train, x_test, y_train, y_test = get_features(df)
    # print(test.shape)

    print('Entering postprocessing')
    mode = train_model(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
    
    save_state(mode, 'mode')
    print('End of the road')


