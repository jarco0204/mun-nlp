import pandas as pd

import librosa.display
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pydub import AudioSegment

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

SILENCE_THRESHOLD = .01
RATE =24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10

def get_wiggles(mfcc, root):
    fig, ax = plt.subplots()
    ax.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    librosa.display.waveshow(mfcc, sr=RATE, ax=ax)
    ax.set_ylim([-450, 50])
    image = FigureCanvasTkAgg(fig, root)
    return image
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

def create_segmented_mfccs(X_train):
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return segmented_mfccs

def get_audio(file:str):
    data, sr = librosa.load(file)
    return librosa.core.resample(y=data, orig_sr=sr, target_sr=RATE, scale=True)

def segment_one(mfcc):
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return np.array(segments)

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

def get_features(df, test_size = 0.2, debug=False):
    feats = []
    for idx, row in df.iterrows():
        if idx%100 == 0:
            print('Working on:', idx)
        audio = get_audio(df.loc[idx,['Audio']][0])
        mute_audio = remove_silence(audio)
        features = to_mfcc(mute_audio)
        feats.append(features)
        if debug and idx == 10:
            break

    if debug:
        y = np.array(df.loc[:11,['spanish', 'english']])
        # y = np.array([i[0] for i in y])
    else:
        y = np.array(df.loc[:,['spanish', 'english']])
    feats, y = segments(feats, y)
    return train_test_split(feats,y,test_size=test_size)


if __name__ == '__main__':
    filename = './data.csv'
    df = pd.read_csv(filename)
    debug = False
    x_train, x_test, y_train, y_test = get_features(df, debug=debug)
    # print(test.shape)

    


