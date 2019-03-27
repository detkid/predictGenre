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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

num_classes = 2
maxlen = 10
max_features = 7521
embedding_dims = 50
batch_size = 1000
epochs = 4
kernel_size = 3
filters = 250
MODEL = 1


def create_model(self):
    model = Sequential()
    model.add(Embedding(self.input_dim, self.output_dim,
                        input_length=1, embeddings_initializer=uniform(seed=20170719)))
    model.add(Flatten())
    model.add(Dense(self.input_dim, use_bias=False,
                    kernel_initializer=glorot_uniform(seed=20170719)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="RMSprop", metrics=['categorical_accuracy'])
    print('#2')
    return model


# 学習
def train(self, x_train, t_train, batch_size, epochs, maxlen, emb_param):
    early_stopping = EarlyStopping(
        monitor='categorical_accuracy', patience=1, verbose=1)
    print('#1', t_train.shape)
    model = self.create_model()
    # model.load_weights(emb_param)    # 埋め込みパラメーターセット。ファイルをロードして学習を再開したいときに有効にする
    print('#3')
    model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
              shuffle=True, callbacks=[early_stopping], validation_split=0.0)
    return model


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


def make_embedding_CNN(x_train, y_train, x_test, y_test, max_features, embedding_dims, maxlen):
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

    # # We add a vanilla hidden layer:
    # model.add(Dense(hidden_dims))
    # model.add(Dropout(0.2))
    # model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    #### softmaxを使用するので改良 ###
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    return model


def make_embedding_RNN(x_train, y_train, x_test, y_test, max_features, embedding_dims, maxlen):
    print('Build RNN model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(LSTM(10, return_sequences=False))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    #### softmaxを使用するので改良 ###
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    return model


if __name__ == "__main__":
    # with codecs.open('data/dataset/clean_dataset.pickle', mode='rb', encoding='utf-8') as f:
    #     data = pickle.loads(f)

    data = np.load('data/dataset/clean_dataset.npy')

    np.random.shuffle(data)
    x_train = data[:, 0]
    y_train = data[:, 1]

    x_train = pad_sequences(x_train, maxlen=maxlen)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    split_number = 9000
    x_val = x_train[split_number:]
    part_x_train = x_train[:split_number]
    y_val = y_train[split_number:]
    part_y_train = y_train[:split_number]

    print('x_train shape = ' + str(part_x_train.shape))
    print('x_val shape = ' + str(x_val.shape))

    if(MODEL == 0):
        model = make_embedding_CNN(
            part_x_train, part_y_train, x_val, y_val, max_features, embedding_dims, maxlen)
    elif(MODEL == 1):
        model = make_embedding_RNN(
            part_x_train, part_y_train, x_val, y_val, max_features, embedding_dims, maxlen)

    print('end')
