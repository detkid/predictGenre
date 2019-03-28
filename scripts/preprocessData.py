import numpy as np
import numpy.random as nr
from gensim import corpora, models, similarities
import pandas as pd
import pickle
import glob
import MeCab
import re
import glob


INPUT_TRAINING_DATA = glob.glob('./data/tweet/*.csv')
INPUT_TEST_DATA = './data/eval/test.csv'
OUTPUT_TRAINING_PATH = './data/dataset/clean_dataset.npy'
OUTPUT_TEST_PATH = './data/dataset/val_dataset.npy'

GENRE = ['スポーツ', '食べ物', '地名', '家族', '本マンガアニメ', '恋愛', '映画', '人間関係', '芸能人', 'テレビ', '仕事']


def csv_to_l_data(file):
    print(file.name + '- importing as data')
    data = []
    dict_items = []
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    node = m.parseToNode('初期化')  # 初期化しないと最初のnode.surfaceが空になる

    raw_data = np.genfromtxt(
        file, delimiter=',', names=True, dtype=None, encoding='utf-8')
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

    return data, dict_items


def csv_to_v_data(file):
    print(file.name + '- importing as val data')
    data = []
    dict_items = []
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    node = m.parseToNode('初期化')  # 初期化しないと最初のnode.surfaceが空になる

    raw_data = np.genfromtxt(
        file, delimiter=',', names=True, dtype=None, encoding='utf-8')
    for row in raw_data:
        node = m.parseToNode(row[1])
        seg_txt = []

        while node:
            word = node.surface
            if node.next:
                word = word.replace(node.next.surface, '')
            if node.feature.split(',')[0] in ['名詞', '動詞']:
                seg_txt.append(word)
            node = node.next

        label = row[0]
        data.append([seg_txt, label])

    return data


def wash_data(text):
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub('RT', "", text)
    text = re.sub('お気に入り', "", text)
    text = re.sub('まとめ', "", text)
    text = re.sub(r'[!-~]', "", text)  # 半角記号,数字,英字
    text = re.sub(r'[︰-＠]', "", text)  # 全角記号

    return text


def make_indexdict(words_list):
    print('making dictionary.')
    mat = np.array(words_list)
    words = sorted(list(set(mat)))
    cnt = np.zeros(len(words))

    print('total words:', len(words))
    word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
    indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

    # 単語の出現数をカウント
    for j in range(0, len(mat)):
        cnt[word_indices[mat[j]]] += 1

    # 出現頻度の少ない単語を「UNK」で置き換え
    words_unk = []                           # 未知語一覧

    for k in range(0, len(words)):
        if cnt[k] <= 3:
            words_unk.append(words[k])
            words[k] = 'UNK'

    print('低頻度語数:', len(words_unk))    # words_unkはunkに変換された単語のリスト

    words = sorted(list(set(words)))
    print('total words:', len(words))
    word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
    indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

    return word_indices

    # 以下、Word2Vec向け
    # # 前後の語数
    # maxlen = 10

    # mat_urtext = np.zeros((len(mat), 1), dtype=int)
    # for i in range(0, len(mat)):
    #     # 出現頻度の低い単語のインデックスをunkのそれに置き換え
    #     if mat[i] in word_indices:
    #         mat_urtext[i, 0] = word_indices[mat[i]]
    #     else:
    #         mat_urtext[i, 0] = word_indices['UNK']

    # print(mat_urtext.shape)

    # len_seq = len(mat_urtext) - maxlen
    # data = []
    # target = []
    # for i in range(maxlen, len_seq):
    #     data.append(mat_urtext[i])
    #     target.extend(mat_urtext[i-maxlen:i])
    #     target.extend(mat_urtext[i+1:i+1+maxlen])

    # x_train = np.array(data).reshape(len(data), 1) # indexのリスト
    # t_train = np.array(target).reshape(len(data), maxlen*2) # 前後の単語のindex (中心の単語は含まない)

    # z = list(zip(x_train, t_train))
    # nr.seed(12345)
    # nr.shuffle(z)                 # シャッフル
    # x_train, t_train = zip(*z)

    # x_train = np.array(x_train).reshape(len(data), 1)
    # t_train = np.array(t_train).reshape(len(data), maxlen*2)

    # print(x_train.shape, t_train.shape)


if __name__ == "__main__":

    all_data = []
    all_words_list = []
    for file_path in INPUT_TRAINING_DATA:
        with open(file_path, encoding='utf-8') as file:
            data, words_list = csv_to_l_data(file)
        all_data.extend(data)
        all_words_list.extend(words_list)
    all_data = np.array(all_data)

    index_dict = make_indexdict(all_words_list)

    np.save('data/dict/word_indices_with_verb.npy', index_dict)

    print('reforming as training dataset.')
    data_list = []
    dataset = []
    for l_data in all_data:
        sentence = l_data[0]
        label = GENRE.index(l_data[1])
        data_list = [index_dict[sentence[i]]
                     for i in range(len(sentence)) if index_dict.get(sentence[i])]

        dataset.append([data_list, label])

    dataset = np.array(dataset)
    np.save(OUTPUT_TRAINING_PATH, dataset)

    # with open('data/dict/word_indices_with_verb.npy') as file:
    #     index_dict = np.load(file)

    print('reforming as test dataset.')
    val_dataset = []
    with open(INPUT_TEST_DATA) as file:
        data = csv_to_v_data(file)

    for v_data_row in data:
        sentence = v_data_row[0]
        label = GENRE.index(v_data_row[1])
        val_list = [index_dict[sentence[i]]
                    for i in range(len(sentence)) if index_dict.get(sentence[i])]
        val_dataset.append([val_list, label])

    val_dataset = np.array(val_dataset)
    np.save(OUTPUT_TEST_PATH, val_dataset)
