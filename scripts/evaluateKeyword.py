import pandas as pd
from preprocessData import csv_to_v_data


GENRE = ['スポーツ', '食べ物', '地名', '家族', '本マンガアニメ',
         '恋愛', '映画', '人間関係', '芸能人', 'テレビ', '仕事']
INPUT_TEST_DATA = './data/eval/test.csv'


if __name__ == "__main__":
    count = 0
    keyword_dict = {}
    with open('./data/talk_genre.csv') as f:
        talk_genre_csv = pd.read_csv(f, sep=' ')

    for genre in GENRE:
        genre_bool_list = talk_genre_csv['genre'] == genre

        all_keywords = talk_genre_csv[genre_bool_list].trigger.values
        for value in all_keywords:
            keywords = value.split(',')
            for word in keywords:
                keyword_dict[word] = genre

    print(len(keyword_dict))

    with open(INPUT_TEST_DATA) as file:
        all_data = csv_to_v_data(file, False)
    
    for data in all_data:
        for noun in data[0]:
            if keyword_dict.get(noun):
                # print(keyword_dict[noun])
                count += 1

    print(str(count) + ' / 110')
    print('accuracy : ' + str(count/110))
