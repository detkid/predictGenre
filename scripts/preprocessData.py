import numpy as np
from gensim import corpora, models, similarities
import pandas as pd
import pickle
import glob
import MeCab
import re

# # TODO
# LDA model
# 1. import csv data as pandas
# 2. format as numpy array [string data, label]
# 3. segmentate string data by mecab
# 4. wash data (discarding symbol, mark, non-independent words)
# 5. make one-hot vector
# 6. make corpus
# 7. make dictionary
# 8. make LDA model

# CNN model
# 1. import csv data as pandas
# 2. format as numpy array [string data, label]
# 3. segmentate string data by mecab
# 4. wash data (discarding symbol, mark, non-independent words)
# 5. make one-hot vector
# 6. make CNN model by keras

# RNN model
# 1. import csv data as pandas
# 2. format as numpy array [string data, label]
# 3. segmentate string data by mecab
# 4. wash data (discarding symbol, mark, non-independent words)
# 5. make one-hot vector
# 6. make LSTM model by keras


def csv_to_l_data(file):
    data = []
    dict_items = []
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    raw_data = np.genfromtxt(file, delimiter=',', names=True, dtype=None)
    for row in raw_data:
        text = wash_data(row[2])
        node = m.parseToNode(text)
        seg_txt = []

        while node:
            word = node.surface
            if node.next:
                word = word.replace(node.next.surface, '')
            if node.feature.split(',')[0] in ['名詞', '動詞'] and word != row[0]:
                dict_items.append(word)
                seg_txt.append(word)
            node = node.next

        label = row[0]
        data.append([seg_txt, label])

    data = np.array(data)

    return data, dict_items


def extract_nouns():

    return data


def wash_data(text):
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub('RT', "", text)
    text = re.sub('お気に入り', "", text)
    text = re.sub('まとめ', "", text)
    text = re.sub(r'[!-~]', "", text)  # 半角記号,数字,英字
    text = re.sub(r'[︰-＠]', "", text)  # 全角記号

    return text


def make_dictionary(words_list):
    dictionary = corpora.Dictionary(words_list)


if __name__ == "__main__":
    # datafiles = glob.glob()
    # clean_data_list = []

    # for filepath in datafiles:
    #     with open(filepath) as file:
    #         data = csv_to_l_data(file)
    #     data = segmentate_string(data)
    #     data = extract_nouns(data)
    #     clean_data_list.append(data)

    with open('./data/sports_data.csv', encoding='utf-8') as file:
        csv_to_l_data(file)

    print('end')

    # pickle.dump(clean_data_list, 'data/clean_dataset')
