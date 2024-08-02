from pprint import pprint
import praw
import pandas as pd
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns

reddit = praw.Reddit(client_id='YIpWKBeAlziDwetjZIB6Aw', client_secret='R-N3NddeWdvyWBCHBdbQf1VgSngFVQ', user_agent="Scraper 1.0 by /u/python_engineer")
subreddit = str(input("Analiz edilecek subreddit'i girin : "))

headlines = set()
try:
    for submission in reddit.subreddit(subreddit).hot(limit=None):
        headlines.add(submission.title)
except praw.exceptions.RedditAPIException as e:
    print(f"Reddit API hatası: {e}")
except praw.exceptions.PRAWException as e:
    print(f"PRAW hatası: {e}")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu, sebebi subreddit bulunmaması olabilir: {e}")
print(len(headlines))

df_reddit = pd.DataFrame(headlines)
df_reddit.head()

df_reddit.to_csv('data/reddit.csv', index=False, encoding='utf-8')

sia = SIA()

results = []
for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df_reddit = pd.DataFrame.from_records(results)
df_reddit.head()

df_reddit['label'] = 0
df_reddit.loc[df_reddit['compound'] > 0.2, 'label'] = 1
df_reddit.loc[df_reddit['compound'] < -0.2, 'label'] = -1
df_reddit.head()

df_reddit2 = df_reddit[['headline', 'label']]
df_reddit2.to_csv('data/reddit_labels.csv', index=False, encoding='utf-8')
df_reddit2.value_counts(normalize=True) * 100

print("Positive headlines:\n")
pprint(list(df_reddit[df_reddit['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df_reddit[df_reddit['label'] == -1].headline)[:5], width=200)

fig, ax = plt.subplots(figsize=(8, 8))
counts = df_reddit.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
plt.show()
