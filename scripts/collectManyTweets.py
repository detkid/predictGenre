from requests_oauthlib import OAuth1Session
import json
import datetime
import time
import sys
from abc import ABCMeta, abstractmethod
import os
import pandas as pd

CK = os.environ['T_Consumer_key']                             # Consumer Key
CS = os.environ['T_Consumer_secret']    # Consumer Secret
AT = os.environ['T_Access_token']    # Access Token
AS = os.environ['T_Access_secret']         # Accesss Token Secert

COUNT = 150  # < 200
KEY_TOTAL = 1500
GENRE = '芸能人'
FILE_NAME = 'data/tweet/' + 'talent_data' + '.csv'

class TweetsGetter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.session = OAuth1Session(CK, CS, AT, AS)

    @abstractmethod
    def specifyUrlAndParams(self, keyword):
        '''
        呼出し先 URL、パラメータを返す
        '''

    @abstractmethod
    def pickupTweet(self, res_text, includeRetweet):
        '''
        res_text からツイートを取り出し、配列にセットして返却
        '''

    @abstractmethod
    def getLimitContext(self, res_text):
        '''
        回数制限の情報を取得 （起動時）
        '''

    def collect(self, total=-1, onlyText=False, includeRetweet=False):
        '''
        ツイート取得を開始する
        '''

        # ----------------
        # 回数制限を確認
        # ----------------
        self.checkLimit()

        # ----------------
        # URL、パラメータ
        # ----------------
        url, params = self.specifyUrlAndParams()
        params['include_rts'] = str(includeRetweet).lower()
        # include_rts は statuses/user_timeline のパラメータ。search/tweets には無効

        # ----------------
        # ツイート取得
        # ----------------
        cnt = 0
        unavailableCnt = 0
        while True:
            res = self.session.get(url, params=params)
            if res.status_code == 503:
                # 503 : Service Unavailable
                if unavailableCnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)

                unavailableCnt += 1
                print('Service Unavailable 503')
                self.waitUntilReset(time.mktime(
                    datetime.datetime.now().timetuple()) + 30)
                continue

            unavailableCnt = 0

            if res.status_code != 200:
                raise Exception('Twitter API error %d' % res.status_code)

            tweets = self.pickupTweet(json.loads(res.text))
            if len(tweets) == 0:
                # len(tweets) != params['count'] としたいが
                # count は最大値らしいので判定に使えない。
                # ⇒  "== 0" にする
                # https://dev.twitter.com/discussions/7513
                break

            for tweet in tweets:
                if (('retweeted_status' in tweet) and (includeRetweet is False)):
                    pass
                else:
                    if onlyText is True:
                        yield tweet['text']
                    else:
                        yield tweet

                    cnt += 1
                    if cnt % 100 == 0:
                        print('%d件 ' % cnt)

                    if total > 0 and cnt >= total:
                        return

            params['max_id'] = tweet['id'] - 1

            # ヘッダ確認 （回数制限）
            # X-Rate-Limit-Remaining が入ってないことが稀にあるのでチェック
            if ('X-Rate-Limit-Remaining' in res.headers and 'X-Rate-Limit-Reset' in res.headers):
                if (int(res.headers['X-Rate-Limit-Remaining']) == 0):
                    self.waitUntilReset(int(res.headers['X-Rate-Limit-Reset']))
                    self.checkLimit()
            else:
                print('not found  -  X-Rate-Limit-Remaining or X-Rate-Limit-Reset')
                self.checkLimit()

    def checkLimit(self):
        '''
        回数制限を問合せ、アクセス可能になるまで wait する
        '''
        unavailableCnt = 0
        while True:
            url = "https://api.twitter.com/1.1/application/rate_limit_status.json"
            res = self.session.get(url)

            if res.status_code == 503:
                # 503 : Service Unavailable
                if unavailableCnt > 10:
                    raise Exception('Twitter API error %d' % res.status_code)

                unavailableCnt += 1
                print('Service Unavailable 503')
                self.waitUntilReset(time.mktime(
                    datetime.datetime.now().timetuple()) + 30)
                continue

            unavailableCnt = 0

            if res.status_code != 200:
                raise Exception('Twitter API error %d' % res.status_code)

            remaining, reset = self.getLimitContext(json.loads(res.text))
            if (remaining == 0):
                self.waitUntilReset(reset)
            else:
                break

    def waitUntilReset(self, reset):
        '''
        reset 時刻まで sleep
        '''
        seconds = reset - time.mktime(datetime.datetime.now().timetuple())
        seconds = max(seconds, 0)
        print('\n     =====================')
        print('     == waiting %d sec ==' % seconds)
        print('     =====================')
        sys.stdout.flush()
        time.sleep(seconds + 10)  # 念のため + 10 秒

    @staticmethod
    def bySearch(keyword):
        return TweetsGetterBySearch(keyword)

    @staticmethod
    def byUser(screen_name):
        return TweetsGetterByUser(screen_name)


class TweetsGetterBySearch(TweetsGetter):
    '''
    キーワードでツイートを検索
    '''

    def __init__(self, keyword):
        super(TweetsGetterBySearch, self).__init__()
        self.keyword = keyword

    def specifyUrlAndParams(self):
        '''
        呼出し先 URL、パラメータを返す
        '''
        global COUNT
        url = 'https://api.twitter.com/1.1/search/tweets.json'
        params = {'q': self.keyword, 'count': COUNT}
        return url, params

    def pickupTweet(self, res_text):
        '''
        res_text からツイートを取り出し、配列にセットして返却
        '''
        results = []
        for tweet in res_text['statuses']:
            results.append(tweet)

        return results

    def getLimitContext(self, res_text):
        '''
        回数制限の情報を取得 （起動時）
        '''
        remaining = res_text['resources']['search']['/search/tweets']['remaining']
        reset = res_text['resources']['search']['/search/tweets']['reset']

        return int(remaining), int(reset)


class TweetsGetterByUser(TweetsGetter):
    '''
    ユーザーを指定してツイートを取得
    '''

    def __init__(self, screen_name):
        super(TweetsGetterByUser, self).__init__()
        self.screen_name = screen_name

    def specifyUrlAndParams(self):
        '''
        呼出し先 URL、パラメータを返す
        '''
        url = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
        params = {'screen_name': self.screen_name, 'count': COUNT}
        return url, params

    def pickupTweet(self, res_text):
        '''
        res_text からツイートを取り出し、配列にセットして返却
        '''
        results = []
        for tweet in res_text:
            results.append(tweet)

        return results

    def getLimitContext(self, res_text):
        '''
        回数制限の情報を取得 （起動時）
        '''
        remaining = res_text['resources']['statuses']['/statuses/user_timeline']['remaining']
        reset = res_text['resources']['statuses']['/statuses/user_timeline']['reset']

        return int(remaining), int(reset)


def set_keyword(genre):
    # genre, row
    # 地名　食べ物　映画　テレビ　スポーツ　芸能人　本マンガアニメ　人間関係　家族　仕事　恋愛  趣味
    # 1,2  3~23  24,25  26      27     28       29,30       31     32   33   34   35

    keyword_list = []
    with open('./data/talk_genre.csv') as f:
        talk_genre_csv = pd.read_csv(f, sep=' ')

    genre_bool_list = talk_genre_csv['genre'] == genre

    all_keywords = talk_genre_csv[genre_bool_list].trigger.values
    for value in all_keywords:
        keywords = value.split(',')
        for word in keywords:
            keyword_list.append(word)

    print('keyword number = ' + str(len(keyword_list)))
    print('total and COUNT should be ' + str(10000/len(keyword_list)))
    return keyword_list


if __name__ == '__main__':

    # ユーザーを指定して取得 （screen_name）
    #getter = TweetsGetter.byUser('AbeShinzo')
    genre = GENRE
    keyword_list = set_keyword(genre)
    for key in keyword_list:
        # キーワードで取得
        getter = TweetsGetter.bySearch(key)
        cnt = 0
        for tweet in getter.collect(total=KEY_TOTAL):
            cnt += 1
            print('------ %d' % cnt)
            print('{} {} {}'.format(
                tweet['id'], tweet['created_at'], '@'+tweet['user']['screen_name']))
            t = ''.join(tweet['text'].splitlines())
            print(t)
            t = t.replace(',', ' ')
            t = t.replace('#', ' ')
            with open(FILE_NAME, 'a') as f:
                f.writelines(genre + ',' + key + ',' + t + '\n')
