import re
import glob


def divide_by_sentence(paragraph):
    sentence_list = re.split('。|\n', paragraph)
    pattern = 'http|20|\n'
    result_list = []
    for sentence in sentence_list:
        matchOB = re.match(pattern, sentence)
        if not matchOB and sentence:
            result_list.append(sentence)

    return result_list


if __name__ == "__main__":
    texts = glob.glob('data/movie-enter/m*')
    for textfile in texts:
        with open(textfile, 'r') as file:
            news = file.read()
            sentences = divide_by_sentence(news)
        with open('data/movie-enter_data.csv', 'a') as file:
            for sentence in sentences:
                file.writelines('映画' + ',livedoor,' + sentence + '\n')
