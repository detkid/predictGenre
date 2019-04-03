import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Activation, LSTM
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model, Sequential, model_from_json
from keras import regularizers, models, layers
from sklearn.metrics import classification_report
import numpy as np
import pickle

# エラー回避
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

num_classes = 11
maxlen = 20

SPLIT_NUMBER = 90000
EVAL_KIND = 1
MODEL = 1  # 0:CNN, 1:RNN


def evaluate_by_sentence(model, x_val, y_val):
    print('evaluating sentence...')
    result = model.predict_classes(x_val)

    for index, group in enumerate(result):
        if group != y_val[index]:
            print(str(index) + ': predict = ' + str(group) +
                  ', label = ' + str(y_val[index]))

    y_val = keras.utils.to_categorical(y_val, num_classes)
    score = model.evaluate(x_val, y_val)
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    return model


def evaluate_by_class(model, x_val, y_val):
    print('evaluating class...')
    y_pred = model.predict(x_val)

    y_label = []
    for y in y_pred:
        y_label.append(np.argmax(y))

    y_label = np.array(y_label)

    print(classification_report(y_val, y_label))

    y_val = keras.utils.to_categorical(y_val, num_classes)
    score = model.evaluate(x_val, y_val)
    print('test loss:', score[0])
    print('test accuracy:', score[1])

    return model


if __name__ == "__main__":

    if MODEL == 0:
        json_file = open('model/cnn/cnn_model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights('model/cnn/cnn_weights.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif MODEL == 1:
        json_file = open('model/rnn/rnn_model.json', 'r')
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights('model/rnn/rnn_weights.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    val_data = np.load('data/dataset/val_dataset.npy')
    x_val = val_data[:, 0]
    y_val = val_data[:, 1]

    y_val = y_val.astype(np.int64)

    x_val = pad_sequences(x_val, maxlen=maxlen)
    # y_val = keras.utils.to_categorical(y_val, num_classes)

    if EVAL_KIND == 1:
        evaluate_by_class(model, x_val, y_val)
    elif EVAL_KIND == 2:
        evaluate_by_sentence(model, x_val, y_val)
