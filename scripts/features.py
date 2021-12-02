from librosa.util.utils import frame
import pandas as pd
import sys

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


if __name__ == '__main__':
    filename = './data.csv'
    df = pd.read_csv(filename)
    print(df.loc['Audio',0])