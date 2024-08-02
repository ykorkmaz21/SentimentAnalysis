import datetime
import GetOldTweets3 as got


def get_tweets_with_query(query, num_tweets):
    criteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                           .setSince("2023-01-01")\
                                           .setUntil(str(datetime.datetime.today()).split()[0])\
                                           .setMaxTweets(num_tweets)
    tweets = got.manager.TweetManager.getTweets(criteria)
    texts = [[tweet.text] for tweet in tweets]
    print(texts)


def get_tweets_with_username(username, num_tweets):
    criteria = got.manager.TweetCriteria().setUsername(username)\
                                           .setMaxTweets(num_tweets)
    tweets = got.manager.TweetManager.getTweets(criteria)
    texts = [[tweet.text] for tweet in tweets]
    print(texts)


# Etiket ya da kullanıcı adı kullanarak tweetleri alabiliyoruz
# query veya username'den aynı anda yalnızca biri kullanılmalı

query = input("Tweetleri almak için bir etiket girin (e.g. #trump) : #")
#username = input("Tweetleri almak için bir kullanıcı adı girin : @")
num_tweets = input("Kaç tweet görüntülemek istersiniz ? ")

get_tweets_with_query(query, int(num_tweets))
#get_tweets_with_username(username, int(num_tweets))
