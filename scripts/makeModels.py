from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers, models, layers
import numpy as np
import pickle


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


if __name__ == "__main__":
    # with codecs.open('data/dataset/clean_dataset.pickle', mode='rb', encoding='utf-8') as f:
    #     data = pickle.loads(f)

    data = np.load('data/dataset/clean_dataset.npy')

    np.random.shuffle(data)
    x_train = data[:, 0]
    y_train = data[:, 1]

    # 訓練データのワンホットベクトル化
    x_train = vectorized_sequences(x_train)
    # x_test  = vectorized_sequences(test_data)

    # ラベルの変換
    y_train = np.asarray(y_train).astype('float32')
    # y_test = np.asarray(test_labels).astype('float32')

    split_number = 9000
    x_val = x_train[:split_number]
    partial_x_train = x_train[split_number:]
    y_val = y_train[:split_number]
    partial_y_train = y_train[split_number:]

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

    print('end')
