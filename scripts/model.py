from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from features import get_features, create_segmented_mfccs
import pandas as pd
import numpy as np
import accuracy
from collections import Counter

SILENCE_THRESHOLD = .01
RATE =24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10


def save_state(model, filename):
    print('Saving module')
    model.save(f'./{filename}03.h5')

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

    modelo.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])

    board = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None)

    gen = ImageDataGenerator(width_shift_range=0.05)
    stop = EarlyStopping(monitor='accuracy', min_delta=.005, patience=10, verbose=1, mode='auto')

    modelo.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / 32
            , epochs=EPOCHS, 
            callbacks=[stop, board], validation_data=(x_val, y_val))
    return modelo

if __name__ == '__main__':
    print('Entering postprocessing')
    filename = './data.csv'
    df = pd.read_csv(filename)
    debug = False
    x_train, x_test, y_train, y_test = get_features(df, debug=debug)
    
    # Get statistics
    train_count = len(y_train)
    test_count = len(y_test)

    print("Entering main")

    # import ipdb;
    # ipdb.set_trace()


    # acc_to_beat = test_count.most_common(1)[0][1] / float(np.sum(list(test_count.values())))

    mode = train_model(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
    
    save_state(mode, 'mode1')
    y_predicted = accuracy.predict_class_all(create_segmented_mfccs(x_test), mode)

    print('Training samples:', train_count)
    print('Testing samples:', test_count)
    # print('Accuracy to beat:', acc_to_beat)
    print('Accuracy:', accuracy.get_accuracy(y_predicted,y_test))

    print('End of the road')