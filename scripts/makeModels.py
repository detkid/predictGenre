import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Activation, LSTM
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras import regularizers, models, layers
import numpy as np
import pickle

# エラー回避
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

num_classes = 11
maxlen = 20
max_features = 5173  # total words
embedding_dims = 50
batch_size = 1000
cnn_epochs = 8
rnn_epochs = 8
kernel_size = 3
filters = 64
lstm_units = 100
hidden_dims = 50
MODEL = 0  # 0: CNN, 1: RNN
DATA_TYPE = 'char'  # word or char

# 入力データのベクトル化を行う関数


def vectorized_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results


def make_sample_model():
    # 訓練データのワンホットベクトル化
    x_train = vectorized_sequences(x_train)
    # x_test  = vectorized_sequences(test_data)

    # ラベルの変換
    y_train = np.asarray(y_train).astype('float32')
    # y_test = np.asarray(test_labels).astype('float32')

    # sample model
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    return history


def make_embedding_CNN(x_train, y_train, max_features, embedding_dims, maxlen):
    print('Build CNN model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    #### softmaxを使用するので改良 ###
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=0, verbose=1)
    model.fit(x_train, y_train, validation_split=0.1,
              batch_size=batch_size,
              epochs=cnn_epochs,
              callbacks=[early_stopping])

    return model


def make_embedding_RNN(x_train, y_train, max_features, embedding_dims, maxlen):
    print('Build RNN model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_units, return_sequences=False))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    #### softmaxを使用するので改良 ###
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=0, verbose=1)
    model.fit(x_train, y_train, validation_split=0.1,
              batch_size=batch_size,
              epochs=rnn_epochs,
              callbacks=[early_stopping])

    return model


if __name__ == "__main__":

    data = np.load('data/dataset/clean_dataset_char.npy')

    np.random.shuffle(data)
    x_train = data[:, 0]
    y_train = data[:, 1]

    x_train = pad_sequences(x_train, maxlen=maxlen)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    val_data = np.load('data/dataset/val_dataset_char.npy')
    x_val = val_data[:, 0]
    y_val = val_data[:, 1]

    x_val = pad_sequences(x_val, maxlen=maxlen)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    print('x_train shape = ' + str(x_train.shape))
    print('x_val shape = ' + str(x_val.shape))

    if(MODEL == 0):
        model = make_embedding_CNN(
            x_train, y_train, max_features, embedding_dims, maxlen)
        json_string = model.to_json()
        open('model/' + DATA_TYPE + '_learning/cnn/cnn_model.json',
             'w').write(json_string)
        model.save_weights('model/' + DATA_TYPE +
                           '_learning/cnn/cnn_weights.h5')
    elif(MODEL == 1):
        model = make_embedding_RNN(
            x_train, y_train, max_features, embedding_dims, maxlen)
        json_string = model.to_json()
        open('model/' + DATA_TYPE + '_learning/rnn/rnn_model.json',
             'w').write(json_string)
        model.save_weights('model/' + DATA_TYPE +
                           '_learning/rnn/rnn_weights.h5')

    # evaluate_model(model, x_val, y_val)
    score = model.evaluate(x_val, y_val)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    print('end\n')
